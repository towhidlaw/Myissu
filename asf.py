python
# =====================================================================
# TECH SHORTS FACTORY v5  (Final Monetization & Affiliate Edition)   brand: OPsblueprint
# Flow: Google Tech News + Hacker News + Wikipedia facts -> Gemini (GitHub Models fallback) script
#   -> Kokoro-82M voice -> faster-whisper word timings -> Pixabay B-roll + local SDXL-Lightning art
#   -> FFmpeg (.ass karaoke, grain, progress bar, watermark, loudnorm) -> thumbnail + post package -> Telegram
# GPU order (never two models at once): Kokoro -> Whisper -> SDXL -> FFmpeg.
# Test with zero GPU / network / keys:  selftest(CFG)   or   python this_file.py --selftest
#
# ---------------- COLAB CELL 1 (run once; does NOT reinstall torch or touch NumPy/CUDA) ----------------
# !apt-get -qq update && apt-get -qq install -y espeak-ng ffmpeg fonts-dejavu-core
# !pip freeze | grep -iE '^(numpy|torch|torchvision|torchaudio)==' > /tmp/keep.txt
# !pip install -q -c /tmp/keep.txt "kokoro>=0.9.4" soundfile faster-whisper diffusers transformers accelerate peft safetensors huggingface_hub requests pillow
# ---------------- COLAB CELL 2: paste this file, then run  run_pipeline()  (or  selftest(CFG) ) ----------------
# Colab secrets: GEMINI_API_KEY, GITHUB_TOKEN (fallback), PIXABAY_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL
# =====================================================================
import difflib
import gc
import html
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import wave
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from io import BytesIO
from itertools import zip_longest
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RESAMPLE = getattr(Image, "Resampling", Image)
SR = 44100


# =====================================================================
# 0. SETTINGS
# =====================================================================
@dataclass
class Config:
    brand: str = "OPsblueprint"
    num_shorts: int = 3
    render_workers: int = 1             # keep 1: prevents RAM crashes on free Colab
    out_dir: str = "/content/shorts" if os.path.isdir("/content") else "./shorts_out"
    send_to_telegram: bool = True
    delete_after_send: bool = True      # deletes only the .mp4, after a SUCCESSFUL sendVideo
    seed: int = None

    topic_source: str = "tech"          # "tech" (Google Tech News + Hacker News) | "google_trends" | "evergreen"
    trends_geo: str = "US"

    # ---- cloud script brain (no local LLM ever touches the GPU) ----
    gemini_models: tuple = ("gemini-3.5-flash", "gemini-3.5-flash-lite")   # next model is tried on 429 / 404 / 503
    github_models: tuple = (("https://models.inference.ai.azure.com/chat/completions", "gpt-4o-mini"),
                            ("https://models.github.ai/inference/chat/completions", "openai/gpt-4o-mini"))
    script_words: tuple = (125, 145)    # TOTAL spoken words incl. the bio nudge
    script_attempts: int = 3
    scenes_per_script: int = 8
    ctas: tuple = (
        "If you want the exact AI and security tools I actually run, they are at the first link in my bio.",
        "The tools I actually use to stay ahead of stories like this are at the first link in my bio.",
        "I keep my favorite AI and tech tools in one place, the first link in my bio.")
    disclosure: str = ("Disclosure: some links in my bio are affiliate links, so I may earn a commission if you buy through them, "
                       "at no extra cost to you. This video uses an AI-generated voice and AI-generated imagery alongside stock footage.")

    # ---- voice (Kokoro-82M, tried in order) ----
    kokoro_voices: tuple = ("am_michael", "am_adam", "af_heart")
    voice_speed: float = 1.05

    # ---- music / mix ----
    bgm_file: str = None
    bgm_db: float = -10.0
    duck_db: float = -5.0
    lead_in: float = 0.4
    tail: float = 1.0

    # ---- visuals ----
    n_clips: tuple = (3, 5)             # real Pixabay MP4 clips per short
    n_ai: tuple = (4, 6)                # local SDXL-Lightning images per short
    clip_max_s: float = 6.5
    local_image_size: tuple = (768, 1344)
    cut_range: tuple = (1.5, 3.2)       # seconds per cut, synced to sentence / word ends
    zoom: float = 0.10

    # ---- video ----
    width: int = 1080
    height: int = 1920
    fps: int = 30
    caption_words: int = 3
    caption_size: int = 78
    caption_uppercase: bool = True
    caption_center_y: float = 0.60
    font_path: str = None
    video_crf: int = 20
    encode_preset: str = "superfast"
    max_file_mb: int = 45


CFG = Config()


# =====================================================================
# 1. UTILITIES
# =====================================================================
def info(msg): print(f"[*] {msg}", flush=True)
def ok(msg): print(f"[+] {msg}", flush=True)
def warn(msg):
    msg = str(msg)
    print(f"[-] {msg if len(msg) <= 330 else msg[:327] + '...'}", flush=True)


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ShortFactory/5.0 (OPsblueprint)"})


def http_get(url, *, params=None, headers=None, timeout=(8, 20), retries=2):
    last = None
    for attempt in range(retries + 1):
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            return r
        except requests.RequestException as e:
            last = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last


def get_secret(name):
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name) or None


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = find_ffmpeg()


def run_ff(args, cwd=None):
    r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], capture_output=True, cwd=cwd)
    if r.returncode:
        raise RuntimeError("ffmpeg: " + r.stderr.decode(errors="ignore")[-400:])


def decode_audio(path):
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype="<f4").copy()


STOP = set("""a an and are as at be but by for from has have how i in into is it its of on or our that the
their this to was we what when where which who why will with you your new says say said after over about
more than up out us not can just now amid vs""".split())


def keywords(text, k=4):
    toks = [t.strip(".-'") for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9+.\-']*", text or "")]
    toks = [t for t in toks if t and t.lower() not in STOP]
    strong = [t for t in toks if t[0].isupper() or any(c.isdigit() for c in t)]
    out, seen = [], set()
    for t in strong + [t for t in toks if t not in strong]:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:k]


_EMOJI = re.compile("[\U00010000-\U0010ffff\u2600-\u27bf\u2b00-\u2bff\ufe0f\u200d]")


def clean_for_speech(text):
    t = _EMOJI.sub("", str(text or ""))
    t = t.translate({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2026: "...", 0x2013: "-", 0x2014: ", "})
    t = re.sub(r"(?im)(^\s*|[.!?]\s+)(?:hook|script|cta|outro|intro|narrator|voice[- ]?over)\s*[:\-]\s*", r"\1", t)
    t = re.sub(r"\((?:pause|beat|music|sfx|laughs?|sighs?)[^)]*\)", " ", t, flags=re.I)
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"[*`~>|]+", "", t).replace("&", " and ")
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t


def parse_json_loose(s):
    s = re.sub(r"^```(?:json)?|```$", "", (s or "").strip(), flags=re.M).strip()
    for cand in (s, (re.search(r"\{.*\}", s, re.S) or re.match("", "")).group(0)):
        try:
            return json.loads(cand, strict=False)
        except Exception:
            pass
    return None


_SCENE_BAN = re.compile(r"\b(text|texts|logo|logos|caption|captions|title|titles|words?|letters?|watermark|brands?|slogan|writing)\b", re.I)


def clean_scene(s):
    """Image prompts must not ask for text/logos (models draw gibberish)."""
    s = re.sub(r'"[^"]*"|“[^”]*”', " ", str(s))
    s = re.sub(r"\b(?:with|showing|displaying|reading|saying|bearing|featuring)\b[^,.;]*?"
               r"\b(?:text|words?|logos?|letters?|captions?|titles?|slogans?|brands?|writing)\b[^,.;]*", " ", s, flags=re.I)
    s = _SCENE_BAN.sub("", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(?:and|with|the|a|an|of|on|in)\s*(?=[,.;]|$)", "", s, flags=re.I)
    s = re.sub(r"\s+([,.;:])", r"\1", s).strip(" \"'.,;:")
    return " ".join(s.split()[:24])


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def normalize_tags(raw, extra=("#Tech", "#Shorts", "#AI")):
    items = re.findall(r"#?\w+", raw) if isinstance(raw, str) else [str(x) for x in raw] if isinstance(raw, (list, tuple)) else []
    tags = []
    for x in items:
        x = re.sub(r"\W", "", x)[:30]
        if x and ("#" + x).lower() not in [t.lower() for t in tags]:
            tags.append("#" + x)
    for e in extra:
        if len(tags) < 4 and e.lower() not in [t.lower() for t in tags]:
            tags.append(e)
    return " ".join(tags[:6])


def cover(img, size):
    """Scale + center-crop to exactly `size` without stretching."""
    tw, th = size
    s = max(tw / img.size[0], th / img.size[1])
    nw, nh = max(tw, math.ceil(img.size[0] * s)), max(th, math.ceil(img.size[1] * s))
    img = img.resize((nw, nh), RESAMPLE.LANCZOS)
    l, t = (nw - tw) // 2, (nh - th) // 2
    return img.crop((l, t, l + tw, t + th))


FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"]
_FONTS = {}


def get_font(cfg, size):
    if size not in _FONTS:
        cands = [cfg.font_path] + FONT_PATHS
        try:
            import matplotlib
            cands.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf"))
        except Exception:
            pass
        font = None
        for p in cands:
            if p and os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, size)
                    break
                except Exception:
                    pass
        if font is None:
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:
                font = ImageFont.load_default()
        _FONTS[size] = font
    return _FONTS[size]


# =====================================================================
# 2. TOPICS + REAL FACTS (Google Tech News RSS + Hacker News + Wikipedia)
# =====================================================================
TECH_THEMES = [
    # (theme, headline regex, Pixabay terms, Pixabay category (computer|science only), built-in SDXL scenes)
    ("ai", r"\b(ai|artificial intelligence|openai|chatgpt|gpt-?\d*|llms?|gemini|claude|anthropic|copilot|deepseek|grok|chatbots?|agents?|machine learning|neural)\b",
     ["artificial intelligence", "robot", "neural network", "machine learning"], "computer",
     ["glowing neural network of light filaments floating in a dark void", "humanoid robot face made of circuitry with cyan eyes"]),
    ("chips", r"\b(chips?|nvidia|amd|intel|cpus?|gpus?|processors?|semiconductors?|tsmc|silicon|qualcomm|snapdragon|blackwell|rtx|geforce)\b",
     ["microchip", "circuit board", "computer processor"], "computer",
     ["macro shot of a glowing silicon chip on a dark circuit board", "silicon wafer reflecting rainbow light inside a clean room"]),
    ("phone", r"\b(iphone|android|samsung|galaxy|pixel|smartphones?|phones?|ios|smartwatch|apple watch|foldables?)\b",
     ["smartphone", "mobile phone technology", "wearable technology"], "computer",
     ["flagship smartphone floating above a dark reflective surface with rim light", "exploded view of smartphone components hovering in mid air"]),
    ("computers", r"\b(macbooks?|laptops?|pcs?|windows|chromebooks?|ipad|tablets?|keyboards?|monitors?)\b",
     ["laptop computer", "desktop computer", "keyboard"], "computer",
     ["sleek laptop open on a dark desk with glowing keyboard backlight", "desktop workstation with glowing RGB components in a dark room"]),
    ("gaming", r"\b(gta|playstation|ps[45]|xbox|nintendo|steam|games?|gaming|rockstar|esports|vr|consoles?|headsets?)\b",
     ["video game", "gaming controller", "virtual reality"], "computer",
     ["glowing game controller on a dark desk with neon rim light", "virtual reality headset floating in a neon lit studio"]),
    ("security", r"\b(hack(?:ed|ers?|ing)?|breach(?:es)?|cyber\w*|ransomware|malware|security|vulnerabilit\w+|data leaks?|privacy|encryption|deepfakes?|scams?|phishing)\b",
     ["cyber security", "hacker", "encryption"], "computer",
     ["glowing padlock made of light over a dark digital grid", "hooded silhouette in front of cascading green code"]),
    ("cloud", r"\b(cloud|aws|azure|data ?centers?|servers?|storage|networks?|5g|6g|wi-?fi|internet|starlink|broadband|outage)\b",
     ["data center", "server room", "cloud computing"], "computer",
     ["endless rows of server racks with glowing blue lights in a data center", "fiber optic cables with bright pulses of light in the dark"]),
    ("robots", r"\b(robots?|robotics|humanoids?|drones?|automation|boston dynamics|optimus|exoskeletons?)\b",
     ["robot", "humanoid robot", "drone"], "computer",
     ["humanoid robot standing in a dark futuristic lab with rim light", "swarm of small drones flying over a neon city at night"]),
    ("auto", r"\b(tesla|evs?|electric vehicles?|self-driving|autonomous|waymo|batter(?:y|ies)|charging)\b",
     ["electric car technology", "autonomous vehicle", "battery technology"], "computer",
     ["sleek electric concept car in a dark futuristic garage with neon light", "close-up of a battery pack glowing with energy cells"]),
    ("space", r"\b(spacex|nasa|rockets?|satellites?|starship|space|moon|mars)\b",
     ["space technology", "rocket launch", "satellite"], "science",
     ["rocket launching at night with a bright exhaust plume", "satellite orbiting Earth with the sun rising behind it"]),
    ("quantum", r"\b(quantum)\b",
     ["quantum computer", "futuristic technology", "circuit board"], "science",
     ["quantum computer chandelier of golden cables in a cryogenic lab", "glowing qubits floating as spheres of light on a dark background"]),
    ("crypto", r"\b(bitcoin|crypto\w*|blockchain|ethereum|nfts?)\b",
     ["bitcoin", "blockchain", "cryptocurrency"], "computer",
     ["glowing golden coin with circuit patterns floating in a dark void", "blockchain network of linked glowing cubes over a dark grid"]),
    ("social", r"\b(tiktok|instagram|meta|facebook|twitter|youtube|social media|streaming|netflix|apps?|startups?|musk)\b",
     ["social media", "smartphone apps", "digital technology"], "computer",
     ["smartphone screen glowing with abstract social icons in a dark room", "network of glowing connected nodes forming a globe"]),
    ("software", r"\b(software|code|coding|developers?|github|programming|linux|open[- ]source|browsers?|chrome|api)\b",
     ["programming code", "software development", "computer code"], "computer",
     ["glowing lines of code floating in dark space", "developer silhouette at a desk with several glowing monitors of code"]),
]
GENERIC_TECH = ["technology", "artificial intelligence", "futuristic technology", "circuit board", "data center", "robot", "computer"]
_THEME_RX = [(n, re.compile(rx, re.I), terms, cat, scenes) for n, rx, terms, cat, scenes in TECH_THEMES]


@dataclass
class Topic:
    title: str
    context: str
    source: str = "news"

    def themes(self):
        text = self.title + " " + self.context.split("\n")[0]
        scored = sorted((-len(rx.findall(text)), i, n, t, c, s) for i, (n, rx, t, c, s) in enumerate(_THEME_RX) if rx.search(text))
        out = [s[2:] for s in scored[:1]]
        if len(scored) > 1 and -scored[1][0] >= min(2, -scored[0][0]):
            out.append(scored[1][2:])
        return out

    def photo_queries(self):
        """[(search term, Pixabay category computer|science), ...]"""
        th = self.themes()
        qs = []
        if th:
            a, b = th[0][1], (th[1][1] if len(th) > 1 else [])
            cat_of = {t: th[0][2] for t in a}
            cat_of.update({t: th[1][2] for t in b if t not in cat_of})
            qs = [(t, cat_of[t]) for t in a[:3] + b[:2] + a[3:] + b[2:]]
        qs += [(g, "computer") for g in GENERIC_TECH]
        out, seen = [], set()
        for q, c in qs:
            if q.lower() not in seen:
                seen.add(q.lower())
                out.append((q, c))
        return out

    def theme_scenes(self):
        return [s for th in self.themes() for s in th[3]]

    def visual_hint(self):
        th = self.themes()
        return th[0][1][0] if th else "technology"


class TopicEngine:
    NEWS_URL = "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en"
    TRENDS_URL = "https://trends.google.com/trending/rss"
    HN_TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
    HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{}.json"
    EVERGREEN = [
        "Artificial Intelligence Agents Revolutionizing Workflows", "Humanoid Robots Entering Global Manufacturing",
        "Next-Generation Quantum Computing Breakthroughs", "Cybersecurity Threats and Deepfakes in Generative AI",
        "Brain-Computer Interfaces Moving Out of the Lab", "Why Data Centers Are Becoming the New Power Plants",
    ]

    def __init__(self, cfg):
        self.cfg = cfg

    def _tech_news(self, limit):
        r = http_get(self.NEWS_URL)
        r.raise_for_status()
        out = []
        for item in ET.fromstring(r.content).findall(".//item"):
            raw = (item.findtext("title") or "").strip()
            if not raw:
                continue
            headline, _, source = raw.rpartition(" - ")
            if not headline:
                headline, source = raw, ""
            desc = item.findtext("description") or ""
            related = [html.unescape(re.sub(r"<[^>]+>", "", m)).strip() for m in re.findall(r"<a [^>]*>(.*?)</a>", desc, re.S)]
            related = [x for x in related if x and x.lower() not in (raw.lower(), headline.lower())][:3]
            ctx = f"Google Tech News headline: {headline}" + (f" (via {source})" if source else "")
            if related:
                ctx += ". Other coverage of the same story: " + " | ".join(related)
            out.append(Topic(headline, ctx, "tech-news"))
            if len(out) >= limit:
                break
        return out

    def _hacker_news(self, limit):
        out = []
        for i in http_get(self.HN_TOP).json()[:40]:
            if len(out) >= limit:
                break
            try:
                it = http_get(self.HN_ITEM.format(i), timeout=(5, 8), retries=1).json()
            except Exception:
                continue
            if not it or it.get("type") != "story":
                continue
            title = (it.get("title") or "").strip()
            if not title or re.match(r"(?i)^(ask|tell) hn", title):
                continue
            title = re.sub(r"(?i)^show hn:\s*", "", title)
            domain = re.sub(r"^www\.", "", re.sub(r"https?://([^/]+).*", r"\1", it.get("url") or ""))
            out.append(Topic(title, f"Hacker News front-page story: {title}" +
                             (f" (from {domain}, {it.get('score', 0)} points, {it.get('descendants', 0)} comments)" if domain else ""), "hacker-news"))
        return out

    def _google_trends(self, limit):
        r = http_get(self.TRENDS_URL, params={"geo": self.cfg.trends_geo})
        r.raise_for_status()
        ns = {"ht": "https://trends.google.com/trending/rss"}
        out = []
        for item in ET.fromstring(r.content).findall(".//item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            news = [html.unescape(ni.findtext("ht:news_item_title", namespaces=ns) or "") for ni in item.findall("ht:news_item", ns)]
            ctx = f"Trending Google search: {title}" + (". Related news: " + " | ".join(n for n in news[:2] if n) if news else "")
            out.append(Topic(title, ctx, "google-trends"))
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def wikipedia_background(title):
        kw = keywords(title, 3)
        if not kw:
            return ""
        try:
            hits = http_get("https://en.wikipedia.org/w/api.php", timeout=(5, 8), retries=1,
                            params={"action": "query", "list": "search", "srsearch": " ".join(kw), "srlimit": 3,
                                    "format": "json"}).json().get("query", {}).get("search", [])
        except Exception:
            return ""
        topic_words = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", title)}
        for h in hits:
            if not any(w.lower() in topic_words for w in re.findall(r"[A-Za-z0-9]+", h["title"])):
                continue
            try:
                s = http_get("https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(h["title"].replace(" ", "_"), safe=""),
                             timeout=(5, 8), retries=1).json()
            except Exception:
                continue
            ext = (s.get("extract") or "").strip()
            if s.get("type") == "standard" and ext:
                if len(ext) > 350:
                    ext = ext[:350].rsplit(". ", 1)[0].rstrip(".") + "."
                return f"Background (Wikipedia, may be older than the news): {ext}"
        return ""

    def get_topics(self, n):
        srcs = {"tech": [self._tech_news, self._hacker_news], "google_trends": [self._google_trends], "evergreen": []}
        pools, topics = [], []
        for fn in srcs.get(self.cfg.topic_source, srcs["tech"]):
            try:
                found = fn(n * 4)
                ok(f"{fn.__name__.strip('_')}: {len(found)} candidates")
                pools.append(found)
            except Exception as e:
                warn(f"{fn.__name__.strip('_')} failed ({type(e).__name__}: {e})")

        def is_dup(t):
            a = set(re.findall(r"\w+", t.title.lower()))
            return any(a and (b := set(re.findall(r"\w+", o.title.lower()))) and len(a & b) / len(a | b) >= 0.5 for o in topics)

        for t in (x for grp in zip_longest(*pools) for x in grp if x):   # interleave news + Hacker News
            if len(topics) < n and not is_dup(t):
                topics.append(t)
        if len(topics) < n:
            warn(f"only {len(topics)} live topics - filling with evergreen ones")
            pool = [e for e in self.EVERGREEN if not is_dup(Topic(e, ""))]
            random.shuffle(pool)
            for e in pool[: n - len(topics)]:
                topics.append(Topic(e, f"Evergreen topic: {e}. No specific news - keep claims general, no dates or statistics.", "evergreen"))
        for t in topics:
            if t.source != "evergreen":
                bg = self.wikipedia_background(t.title)
                if bg:
                    t.context += "\n" + bg
        return topics[:n]


# =====================================================================
# 3. SCRIPT BRAIN - Gemini (Google AI Studio) -> GitHub Models fallback. No local LLM, no Ollama.
# =====================================================================
@dataclass
class Script:
    text: str          # everything the voice says: body + bio nudge
    body: str
    title: str
    tags: str
    scenes: list
    description: str
    pinned: str
    hook: str          # 3-4 word thumbnail banner
    from_llm: bool = True


CTA_RX = re.compile(r"\b(follow|subscribe|comments?|like and|share this|link in (?:my |the )?bio|"
                    r"(?:check|see) (?:out )?(?:my |the )?bio|hit the|notifications?)\b", re.I)


def strip_trailing_cta(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    while len(sents) > 3 and CTA_RX.search(sents[-1]):
        sents.pop()
    return " ".join(sents)


HOOK_STYLES = [
    "a bold claim that sounds almost too big to be true (but is backed by the facts)",
    "a sharp question that speaks straight to the viewer",
    "a 'nobody is talking about this' reveal",
    "a warning about what the viewer is about to miss or get wrong",
    "a fast 'here is what I just found' tease that promises a twist",
]

SYSTEM_PROMPT = (
    "You are the writer behind OPsblueprint, a first-person investigative tech creator on YouTube Shorts and TikTok. "
    "You research first, then explain in a sharp, confident, slightly skeptical voice: 'I dug into this', 'here is what I found', "
    "'what caught my eye'. You never invent facts, numbers, quotes, hands-on tests or personal experiences: you only state what the "
    "FACTS block supports, and when facts are thin you explain why the news matters. Reply with one valid JSON object and nothing else.")


class CloudLLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gem, self.gh = get_secret("GEMINI_API_KEY"), get_secret("GITHUB_TOKEN")
        self.used = ""

    def prepare(self):
        return bool(self.gem or self.gh)

    def label(self):
        return " -> ".join(x for x, k in (("Gemini (Google AI Studio)", self.gem), ("GitHub Models gpt-4o-mini", self.gh)) if k)

    def _scrub(self, e):
        s = str(e)
        for k in (self.gem, self.gh):
            if k:
                s = s.replace(k, "***")
        return s

    def _gemini(self, system, user):
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0}}
        for model in self.cfg.gemini_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gem}"
            r = SESSION.post(url, json=body, timeout=(10, 90))
            if r.status_code in (429, 404, 503):
                warn(f"Gemini {model}: HTTP {r.status_code} - trying the next model")
                continue
            if r.status_code != 200:
                raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:150]}")
            self.used = f"Gemini / {model}"
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        raise RuntimeError("all Gemini models rate-limited or unavailable")

    def _github(self, system, user):
        errs = []
        for url, model in self.cfg.github_models:
            r = SESSION.post(url, headers={"Authorization": f"Bearer {self.gh}", "Content-Type": "application/json"},
                             json={"model": model, "temperature": 1.0, "max_tokens": 1800,
                                   "response_format": {"type": "json_object"},
                                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                             timeout=(10, 90))
            if r.status_code == 200:
                self.used = f"GitHub Models / {model}"
                return r.json()["choices"][0]["message"]["content"]
            errs.append(f"HTTP {r.status_code} {r.text[:80]}")
        raise RuntimeError("GitHub Models failed: " + " | ".join(errs))

    def chat_json(self, system, user):
        errs = []
        for key, fn, name in ((self.gem, self._gemini, "Gemini"), (self.gh, self._github, "GitHub Models")):
            if not key:
                continue
            try:
                return parse_json_loose(fn(system, user))
            except Exception as e:
                errs.append(f"{name}: {self._scrub(e)[:140]}")
                warn(errs[-1] + (" - falling back" if name == "Gemini" and self.gh else ""))
        raise RuntimeError("; ".join(errs) or "no cloud key available")


class ScriptEngine:
    def __init__(self, cfg, rng):
        self.cfg, self.rng, self.llm = cfg, rng, CloudLLM(cfg)

    def setup(self):
        if not self.llm.prepare():
            raise RuntimeError("add GEMINI_API_KEY and/or GITHUB_TOKEN to Colab Secrets (there is no local-LLM fallback)")
        ok(f"script brain: {self.llm.label()}")

    def _prompt(self, topic, hook, blo, bhi, note):
        n = self.cfg.scenes_per_script
        return f"""TOPIC: {topic.title}

FACTS (the only facts you may state - never invent numbers, dates, names, quotes or hands-on tests):
{topic.context}

Write the voice-over BODY of a short video: {blo}-{bhi} words, first person, investigative creator voice. The closing line about my bio is added separately - do NOT write one.
Structure:
1. HOOK (1-2 sentences): {hook}. It must stop the scroll in two seconds.
2. WHAT I FOUND (2-3 sentences): the story in plain words ("I dug into...", "here is what I found").
3. WHY IT MATTERS (3-4 sentences): what changes for normal people, with one concrete everyday example.
4. THE CATCH (2 sentences): the risk or the part nobody mentions. Open a loop.
5. CLOSING THOUGHT (1 sentence): sharp and memorable.
Style: spoken English, contractions, talk to "you", sentences of 6-14 words with a few very short ones. No filler like "in today's video". \
No emojis, hashtags, markdown, stage directions or labels. Never tell people to follow, subscribe, comment or check anything.

Return one JSON object with these keys:
"script": the body as one plain-text paragraph (no line breaks, no double quotes inside),
"title": curiosity-driven, honest, under 60 characters,
"hook": a 3-4 word ALL-CAPS thumbnail banner,
"description": two sentences for the video description (no links),
"pinned_comment": one friendly line that points to the first link in my bio,
"tags": 5 relevant hashtags,
"scenes": exactly {n} image prompts for an AI image model in story order, 10-16 words each (subject + setting + lighting). Technology only \
(devices, chips, robots, code, data centers, futuristic interfaces) tied to the story. No faces, brand names, logos or readable text.
{note}"""

    def generate(self, topic, hook):
        cfg = self.cfg
        lo, hi = cfg.script_words
        cta = clean_for_speech(self.rng.choice(cfg.ctas))
        blo, bhi = lo - len(cta.split()), hi - len(cta.split())
        mid, best, note = (blo + bhi) / 2, None, ""
        for attempt in range(1, cfg.script_attempts + 1):
            try:
                data = self.llm.chat_json(SYSTEM_PROMPT, self._prompt(topic, hook, blo, bhi, note))
            except Exception as e:
                warn(f"script call failed ({e})")
                time.sleep(3)
                continue
            if not isinstance(data, dict):
                note = "\nYour last reply was not valid JSON. Reply with the JSON object only."
                continue
            body = strip_trailing_cta(clean_for_speech(data.get("script")))
            n = len(body.split())
            if best is None or abs(n - mid) < abs(len(best[0].split()) - mid):
                best = (body, data)
            if blo - 8 <= n <= bhi + 8:
                ok(f"script via {self.llm.used}: {n + len(cta.split())} words incl. bio nudge (attempt {attempt})")
                return self._pack(topic, body, data, cta)
            warn(f"script body had {n} words, want {blo}-{bhi} (attempt {attempt})")
            note = f"\nYour previous script had {n} words. It MUST be {blo}-{bhi} words."
        if best and 0.7 * blo <= len(best[0].split()) <= 1.3 * bhi:
            warn("using the closest script even though it is outside the word range")
            return self._pack(topic, best[0], best[1], cta)
        raise RuntimeError("the cloud LLM produced no usable script")

    def _pack(self, topic, body, data, cta):
        title = re.sub(r"\s+", " ", str(data.get("title") or topic.title)).strip()
        if len(title) > 60:
            title = title[:57].rstrip() + "..."
        hook = " ".join(re.sub(r"[^A-Za-z0-9$%+'!?-]", " ", str(data.get("hook") or "")).split()[:4]).upper()
        if len(hook.split()) < 3:
            hook = " ".join(keywords(topic.title, 4)).upper() or "WATCH THIS"
        scenes = [re.sub(r"\s+", " ", str(s)).strip() for s in (data.get("scenes") or []) if str(s).strip()]
        desc = clean_for_speech(data.get("description")) if data.get("description") else re.split(r"(?<=[.!?])\s+", body)[0]
        pinned = re.sub(r"\s+", " ", str(data.get("pinned_comment") or "")).strip()
        if not pinned or "bio" not in pinned.lower():
            pinned = "Every tool I actually use is at the first link in my bio."
        return Script(f"{body} {cta}", body, title, normalize_tags(data.get("tags")), scenes, desc, pinned, hook)


def write_post_package(cfg, s, topic, path):
    txt = (f"TITLE:\n{s.title}\n\n"
           f"DESCRIPTION:\n{s.description}\nThe exact AI and security tools I use: first link in my bio.\n\n{cfg.disclosure}\n\n{s.tags}\n\n"
           f"PINNED COMMENT:\n{s.pinned}\n\nFTC / AI DISCLOSURE:\n{cfg.disclosure}\n\nTAGS:\n{s.tags}\n\n"
           f"SOURCE: {topic.title} ({topic.source})\n\nSPOKEN SCRIPT:\n{s.text}\n")
    Path(path).write_text(txt, encoding="utf-8")


# =====================================================================
# 4. VOICE: Kokoro-82M (local neural TTS) + faster-whisper (millisecond word alignment)
# =====================================================================
@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Voiceover:
    pcm: np.ndarray
    words: list
    duration: float
    voice: str
    wav: str = None
    timing: str = "estimated"


def spread(tokens, start, end):
    """Fallback timing: share [start, end] between tokens by length, with a pause after punctuation."""
    if not tokens:
        return []
    weights = [len(t) + 2 + (6 if t[-1] in ".,!?;:" else 0) for t in tokens]
    total, t, out = float(sum(weights)), start, []
    for tok, w in zip(tokens, weights):
        d = (end - start) * w / total
        out.append(Word(tok, t, t + d))
        t += d
    return out


def align_words(text, heard, dur):
    """Keep the script's own words/punctuation, take timestamps from Whisper (difflib match, gaps interpolated)."""
    toks = text.split()
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    times = [None] * len(toks)
    for b in difflib.SequenceMatcher(None, [norm(t) for t in toks], [norm(h[0]) for h in heard], autojunk=False).get_matching_blocks():
        for k in range(b.size):
            times[b.a + k] = (heard[b.b + k][1], heard[b.b + k][2])
    out, i = [], 0
    while i < len(toks):
        if times[i]:
            out.append(Word(toks[i], *times[i]))
            i += 1
            continue
        j = i
        while j < len(toks) and times[j] is None:
            j += 1
        s = out[-1].end if out else 0.05
        e = times[j][0] if j < len(toks) else max(dur - 0.05, s + 0.1 * (j - i))
        out += spread(toks[i:j], s, max(e, s + 0.05 * (j - i)))
        i = j
    prev = 0.0
    for w in out:
        w.start = max(w.start, prev)
        w.end = max(w.end, w.start + 0.03)
        prev = w.end
    return out


def write_wav(path, arr, sr=SR):
    arr = np.asarray(arr)
    arr = arr[:, None] if arr.ndim == 1 else arr
    with wave.open(str(path), "wb") as w:
        w.setnchannels(arr.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(arr, -1, 1) * 32767).astype("<i2").tobytes())


class VoiceEngine:
    def __init__(self, cfg):
        self.cfg, self.tts, self.stt, self.stt_bad = cfg, None, None, False

    def _speak(self, text, voice, tmp, tag):
        if self.tts is None:
            from kokoro import KPipeline
            info("loading Kokoro-82M...")
            self.tts = KPipeline(lang_code="a")
        parts = []
        for sent in re.split(r"(?<=[.!?])\s+", text):
            chunks = []
            for _, _, aud in self.tts(sent, voice=voice, speed=self.cfg.voice_speed):
                if aud is not None:
                    a = aud.detach().cpu().numpy() if hasattr(aud, "detach") else np.asarray(aud)
                    chunks.append(a.astype(np.float32).reshape(-1))
            if chunks:
                parts += [np.concatenate(chunks), np.zeros(int(0.14 * 24000), np.float32)]
        if not parts:
            raise RuntimeError("Kokoro returned no audio")
        wav = Path(tmp) / f"voice_{tag}.wav"
        write_wav(wav, np.concatenate(parts), 24000)
        pcm = decode_audio(wav)          # ffmpeg resamples 24 kHz -> 44.1 kHz
        ok(f"Kokoro voice {voice}: {len(pcm) / SR:.1f}s")
        return Voiceover(pcm, [], len(pcm) / SR, voice, str(wav))

    def speak(self, text, tmp, tag):
        last = None
        for voice in self.cfg.kokoro_voices:
            try:
                return self._speak(text, voice, tmp, tag)
            except Exception as e:
                last = e
                warn(f"Kokoro voice {voice} failed ({type(e).__name__}: {e})")
        raise RuntimeError(f"Kokoro failed: {last}")

    def release_tts(self):
        if self.tts is not None:
            del self.tts
        self.tts = None
        free_gpu()

    def align(self, text, wav, dur):
        try:
            if self.stt is None and not self.stt_bad:
                from faster_whisper import WhisperModel
                info("loading faster-whisper base.en...")
                self.stt = WhisperModel("base.en", device="cuda", compute_type="float16")
            if self.stt is None:
                raise RuntimeError("Whisper unavailable")
            segs, _ = self.stt.transcribe(str(wav), language="en", word_timestamps=True)
            heard = [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or [])]
            if len(heard) < 5:
                raise RuntimeError("too few words recognised")
            return align_words(text, heard, dur), "whisper word timestamps"
        except Exception as e:
            self.stt_bad = True
            warn(f"Whisper alignment failed ({type(e).__name__}: {e}) - using estimated timings")
            return spread(text.split(), 0.12, max(dur - 0.1, 0.5)), "estimated"

    def release_stt(self):
        if self.stt is not None:
            del self.stt
        self.stt = None
        free_gpu()


# =====================================================================
# 5. MUSIC + SFX + MIX
# =====================================================================
def _hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def _add(out, start, sig):
    i = int(start * SR)
    if i >= len(out):
        return
    j = min(len(out), i + len(sig))
    out[i:j] += sig[: j - i].astype(np.float32)


PROGS = [  # (semitones above the key root, chord quality) - minor 0,3,7 / major 0,4,7
    [(0, 0), (8, 1), (3, 1), (10, 1)], [(0, 0), (5, 0), (8, 1), (10, 1)],
    [(0, 1), (7, 1), (9, 0), (5, 1)], [(0, 0), (3, 1), (10, 1), (7, 0)],
]


def synth_bgm(duration, rng, np_rng):
    """Randomized tempo / key / chord progression: pad + bass + kick + hat + arp."""
    total = duration + 2.0
    out = np.zeros(int(total * SR), np.float32)
    beat = 60.0 / rng.randint(96, 128)
    bar, key = 4 * beat, rng.randint(0, 11)
    prog = [(45 + (key + off) % 12, (0, 4, 7) if q else (0, 3, 7)) for off, q in rng.choice(PROGS)]
    pad_len = bar + 0.6
    tt = np.arange(int(pad_len * SR)) / SR
    pad_env = np.minimum(1.0, tt / 0.45) * np.minimum(1.0, (pad_len - tt) / 0.6)
    for b in range(int(math.ceil(total / bar)) + 1):
        root, iv = prog[b % 4]
        sig = np.zeros_like(tt)
        for k in iv + (12,):
            f = _hz(root + 12 + k)
            sig += np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 1.004 * tt) + 0.2 * np.sin(2 * np.pi * 2 * f * tt)
        _add(out, b * bar, 0.16 * pad_env * sig / 4)
    eighth = beat / 2
    nt = np.arange(int(eighth * 0.95 * SR)) / SR
    bass_env = np.minimum(1.0, nt / 0.006) * np.exp(-nt * 4.0)
    at = np.arange(int(0.45 * SR)) / SR
    arp_env = np.minimum(1.0, at / 0.004) * np.exp(-at * 9.0)
    kt = np.arange(int(0.28 * SR)) / SR
    kick = np.sin(2 * np.pi * (42 * kt + 85 * (1 - np.exp(-30 * kt)) / 30)) * np.exp(-kt * 10)
    ht = np.arange(int(0.05 * SR)) / SR
    bass_pat = rng.choice([[1, 0, 1, 1, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0, 1]])
    arp_pat = rng.choice([[0, 2, 1, 2, 0, 2, 1, 3], [0, 1, 2, 3, 2, 1, 2, 3], [0, 2, 3, 2, 1, 2, 3, 2]])
    for e in range(int(total / eighth)):
        t, step = e * eighth, e % 8
        root, iv = prog[int(t // bar) % 4]
        if bass_pat[step]:
            f = _hz(root - 12)
            _add(out, t, 0.13 * (np.sin(2 * np.pi * f * nt) + 0.5 * np.sin(2 * np.pi * 2 * f * nt) + 0.25 * np.sin(2 * np.pi * 3 * f * nt)) * bass_env)
        f = _hz(root + 24 + (iv + (12,))[arp_pat[step] % 4])
        _add(out, t, 0.15 * (np.sin(2 * np.pi * f * at) + 0.3 * np.sin(2 * np.pi * 2 * f * at)) * arp_env)
        if step % 2 == 0:
            _add(out, t, 0.17 * kick)
        else:
            _add(out, t, 0.11 * np.diff(np_rng.standard_normal(len(ht) + 1)) * np.exp(-ht * 70))
    return out[: int(duration * SR)]


def _band(np_rng, n, fc, bw=0.7):
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = np.fft.rfft(np_rng.standard_normal(n)) * np.exp(-0.5 * ((np.log2(f + 1.0) - math.log2(fc)) / bw) ** 2)
    x = np.fft.irfft(spec, n)
    return x / (np.abs(x).max() + 1e-9)


def _norm(x):
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def sfx_whoosh(np_rng, dur=0.5):
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    x = sum(_band(np_rng, n, fc) * np.exp(-0.5 * ((t - c) / 0.17) ** 2) for fc, c in ((600, .3), (1800, .5), (5000, .7)))
    return _norm(x * np.sin(np.pi * t) ** 0.7)


def sfx_riser(np_rng, dur=1.4):
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    tone = np.sin(2 * np.pi * np.cumsum(180 + 2600 * t ** 2) / SR)
    return _norm((0.6 * tone + 0.6 * _band(np_rng, n, 3500, 1.0)) * t ** 2)


def sfx_drop(np_rng, dur=0.9):
    n = int(dur * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * np.cumsum(38 + 110 * np.exp(-9 * t)) / SR) * np.exp(-3.2 * t)
    return _norm(body + 0.5 * _band(np_rng, n, 900, 1.2) * np.exp(-30 * t))


def plan_sfx(shots, np_rng):
    """Impact on the hook, whoosh on every cut, riser + drop into the 'twist' cut (~60% in)."""
    ev = [(0.0, sfx_drop(np_rng), 0.30)]
    tw = min(len(shots) - 1, max(1, int(len(shots) * 0.6)))
    for i, s in enumerate(shots):
        if i == 0:
            continue
        if i == tw:
            d = min(1.4, s.start)
            if d > 0.3:
                ev.append((s.start - d, sfx_riser(np_rng, d), 0.13))
            ev.append((s.start, sfx_drop(np_rng), 0.30))
        else:
            ev.append((max(0.0, s.start - 0.15), sfx_whoosh(np_rng), 0.12))
    return ev


def fit_music(duration, cfg, rng, np_rng):
    n = int(duration * SR)
    if cfg.bgm_file and os.path.exists(cfg.bgm_file):
        pcm = decode_audio(cfg.bgm_file)
        if len(pcm) < n + SR:
            pcm = np.tile(pcm, int(math.ceil((n + SR) / max(len(pcm), 1))))
        start = rng.randint(0, len(pcm) - n)
        out = pcm[start: start + n].copy()
    else:
        out = synth_bgm(duration, rng, np_rng)
    out = out / (np.abs(out).max() or 1.0) * 0.9
    fi, fo = min(int(0.8 * SR), n // 4), min(int(1.6 * SR), n // 3)
    out[:fi] *= np.linspace(0, 1, fi)
    out[-fo:] *= np.linspace(1, 0, fo)
    return out.astype(np.float32)


def _phone_band(x, cutoff=250.0):
    spec = np.fft.rfft(x)
    spec[: int(cutoff * len(x) / SR)] = 0
    return np.fft.irfft(spec, len(x)).astype(np.float32)


def _rms(x):
    x = np.abs(x)
    x = x[x > 0.05 * (x.max() if x.size else 0)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 1e-6


def duck_curve(voice, floor_db):
    """Sidechain-style gain curve: music dips while the voice speaks (fast attack, slow release)."""
    block = 441
    nb = len(voice) // block
    if nb == 0:
        return np.ones(len(voice), np.float32)
    env = np.abs(voice[: nb * block]).reshape(nb, block).mean(1)
    target = (env > max(env.max() * 0.08, 1e-4)).astype(np.float64)
    a_att, a_rel = 1 - math.exp(-1 / 4.0), 1 - math.exp(-1 / 35.0)
    s, sm = 0.0, np.empty(nb)
    for i, tg in enumerate(target):
        s += (a_att if tg > s else a_rel) * (tg - s)
        sm[i] = s
    gain = 1 + (10 ** (floor_db / 20) - 1) * sm
    return np.interp(np.arange(len(voice)), (np.arange(nb) + 0.5) * block, gain).astype(np.float32)


def mix_audio(voice_pcm, music, cfg, total_frames, sfx):
    n = int(round(total_frames * SR / cfg.fps))
    v = np.zeros(n, np.float32)
    off = int(cfg.lead_in * SR)
    seg = voice_pcm[: max(0, n - off)]
    v[off: off + len(seg)] = seg
    v *= 0.16 / _rms(v)
    if np.abs(v).max() > 0.95:
        v *= 0.95 / np.abs(v).max()
    m = np.zeros(n, np.float32)
    m[: min(n, len(music))] = music[:n]
    m *= (_rms(v) * 10 ** (cfg.bgm_db / 20)) / _rms(_phone_band(m))
    m *= duck_curve(v, cfg.duck_db)
    fx = np.zeros(n, np.float32)
    for t, sig, g in sfx:
        _add(fx, t, g * sig)
    mix = v + m + fx
    if np.abs(mix).max() > 0.97:
        mix *= 0.97 / np.abs(mix).max()
    return np.column_stack([mix, mix])


# =====================================================================
# 6. VISUALS: Pixabay videos + Pixabay images (backfill) + local SDXL-Lightning
# =====================================================================
@dataclass
class Asset:
    kind: str                 # "ai" | "photo" | "clip"
    img: Image.Image = None
    path: str = None          # pre-cropped 1080x1920 mp4 for clips
    dur: float = 0.0


PALETTES = ["deep blue and cyan neon glow", "purple and magenta neon glow",
            "teal and warm orange cinematic color grade", "icy white and electric blue glow"]
STYLE = "cinematic tech photography, {palette}, dramatic rim lighting, shallow depth of field, ultra detailed, sharp focus, 8k, vertical composition"
GENERIC_SCENES = [
    "futuristic data center with glowing blue servers", "holographic interface floating above a dark desk",
    "close-up of a glowing microchip with golden circuits", "humanoid robot hand touching a network of light",
    "neon city skyline at night with flowing data streams", "wall of giant screens filled with streaming code",
]


def procedural_bg(rng, size):
    w, h = size
    c1 = np.array([rng.randint(8, 30), rng.randint(14, 40), rng.randint(40, 90)], np.float32)
    c2 = np.array([rng.randint(0, 20), rng.randint(0, 20), rng.randint(15, 45)], np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    arr = np.repeat(c1 * (1 - t) + c2 * t, w, axis=1)
    arr[::96, :, :] += 18
    arr[:, ::96, :] += 18
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class LocalImageGen:
    """SDXL base + ByteDance 4-step Lightning LoRA + fp16-fix VAE on the Colab GPU."""
    BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    VAE = "madebyollin/sdxl-vae-fp16-fix"
    LORA_REPO, LORA_FILE = "ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors"

    def __init__(self, cfg):
        self.cfg, self.pipe, self.failed = cfg, None, False

    @staticmethod
    def usable():
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "no GPU (Runtime > Change runtime type > T4 GPU)"
            import diffusers  # noqa: F401
            import peft  # noqa: F401
        except ImportError as e:
            return False, f"python package '{e.name}' is missing (run Cell 1)"
        return True, ""

    def load(self):
        import torch
        from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline
        from huggingface_hub import hf_hub_download
        info("loading SDXL-Lightning onto the GPU (first time downloads ~7 GB)...")
        extra = {}
        try:
            extra["vae"] = AutoencoderKL.from_pretrained(self.VAE, torch_dtype=torch.float16)
        except Exception as e:
            warn(f"fp16-fix VAE unavailable ({type(e).__name__}) - using the default one")
        pipe = StableDiffusionXLPipeline.from_pretrained(self.BASE, torch_dtype=torch.float16, variant="fp16",
                                                         use_safetensors=True, **extra)
        pipe.load_lora_weights(hf_hub_download(self.LORA_REPO, self.LORA_FILE))
        pipe.fuse_lora()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to("cuda")

    def generate(self, prompt, seed):
        import torch
        w, h = self.cfg.local_image_size
        kw = dict(num_inference_steps=4, guidance_scale=0.0, width=w, height=h)
        try:
            return self.pipe(prompt, generator=torch.Generator("cuda").manual_seed(seed), **kw).images[0]
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            free_gpu()
            self.pipe.enable_vae_tiling()
            return self.pipe(prompt, generator=torch.Generator("cuda").manual_seed(seed), **kw).images[0]

    def release(self):
        if self.pipe is not None:
            del self.pipe
        self.pipe = None
        free_gpu()


class Seen:
    """Cross-run Pixabay de-duplication (ids persisted as JSON next to the outputs)."""
    def __init__(self, path):
        self.path = Path(path)
        try:
            self.ids = set(json.loads(self.path.read_text()))
        except Exception:
            self.ids = set()

    def __contains__(self, k): return k in self.ids
    def add(self, k): self.ids.add(k)

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(sorted(self.ids)[-6000:]))
        except Exception:
            pass


class VisualEngine:
    def __init__(self, cfg, rng, pixabay_key, tmp):
        self.cfg, self.rng, self.key, self.tmp = cfg, rng, pixabay_key, tmp
        self.local = LocalImageGen(cfg)
        self.seen = Seen(Path(cfg.out_dir) / ".pixabay_seen.json")
        self.dead = False

    def plan(self):
        good, why = self.local.usable()
        (ok if good else warn)("image brain: local SDXL-Lightning on the GPU" if good else
                               f"local SDXL unavailable ({why}) - Pixabay images will backfill")

    def release(self):
        self.local.release()

    def _hits(self, url, params):
        if self.dead or not self.key:
            return []
        try:
            r = http_get(url, params={"key": self.key, **params}, timeout=(8, 15), retries=1)
            if r.status_code != 200:
                warn(f"Pixabay {r.status_code}: {r.text[:100]!r} - skipping Pixabay for this short")
                self.dead = True
                return []
            hits = r.json().get("hits", [])
        except Exception as e:
            warn(f"Pixabay unreachable ({type(e).__name__}) - skipping Pixabay for this short")
            self.dead = True
            return []
        self.rng.shuffle(hits)
        return hits

    def _fetch_clip(self, url, dur, tag):
        cfg = self.cfg
        src, dst = Path(self.tmp) / f"src_{tag}.mp4", Path(self.tmp) / f"clip_{tag}.mp4"
        try:
            with SESSION.get(url, stream=True, timeout=(8, 40)) as r:
                if r.status_code != 200:
                    return None
                size = 0
                with open(src, "wb") as f:
                    for ch in r.iter_content(1 << 20):
                        size += len(ch)
                        if size > 60e6:
                            return None
                        f.write(ch)
            vf = (f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=increase,"
                  f"crop={cfg.width}:{cfg.height},fps={cfg.fps},setsar=1")       # center-crop, never stretch
            run_ff(["-i", str(src), "-t", f"{cfg.clip_max_s}", "-vf", vf, "-an", "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "19", "-pix_fmt", "yuv420p", str(dst)])
            return Asset("clip", path=str(dst), dur=min(float(dur or cfg.clip_max_s), cfg.clip_max_s))
        except Exception as e:
            warn(f"clip download/crop failed ({type(e).__name__}: {str(e)[:120]})")
            return None
        finally:
            src.unlink(missing_ok=True)

    def _pixabay_videos(self, queries, n, tag):
        got = []
        for q, cat in queries:
            if len(got) >= n or self.dead or not self.key:
                break
            cat, taken = (cat if cat in ("computer", "science") else "computer"), 0
            for h in self._hits("https://pixabay.com/api/videos/", {"q": q[:100], "video_type": "film", "category": cat,
                                                                    "safesearch": "true", "per_page": 30}):
                key = f"v{h['id']}"
                if key in self.seen:
                    continue
                v, url = h.get("videos") or {}, None
                for size in ("large", "medium"):
                    d = v.get(size) or {}
                    if d.get("url") and (d.get("size") or 0) < 40e6:
                        url = d["url"]
                        break
                clip = self._fetch_clip(url, h.get("duration"), f"{tag}_{len(got)}") if url else None
                if clip:
                    self.seen.add(key)
                    got.append(clip)
                    taken += 1
                if len(got) >= n or taken >= 2:
                    break
        return got

    def _download(self, url):
        try:
            r = http_get(url, timeout=(8, 25), retries=1)
            if r.status_code != 200 or len(r.content) < 5000:
                return None
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return img if min(img.size) >= 480 else None
        except Exception:
            return None

    def _pixabay_images(self, queries, n):
        got = []
        for q, cat in queries:
            if len(got) >= n or self.dead or not self.key:
                break
            cat, taken = (cat if cat in ("computer", "science") else "computer"), 0   # strictly tech / science
            for h in self._hits("https://pixabay.com/api/", {"q": q[:100], "image_type": "photo", "orientation": "vertical",
                                                             "category": cat, "safesearch": "true", "order": "popular",
                                                             "per_page": 50}):
                key = f"i{h['id']}"
                if key in self.seen:
                    continue
                img = self._download(h.get("largeImageURL"))
                if img:
                    self.seen.add(key)
                    got.append(Asset("photo", img=img))
                    taken += 1
                if len(got) >= n or taken >= 2:
                    break
        return got

    def _prompts(self, topic, scenes, n):
        hint = topic.visual_hint()
        pool = [s for s in (clean_scene(x) for x in scenes) if s][:n]
        for extra in topic.theme_scenes() + GENERIC_SCENES:
            if len(pool) >= n:
                break
            if extra not in pool:
                pool.append(extra)
        while len(pool) < n:
            pool.append(pool[len(pool) % max(1, len(pool))])
        style = STYLE.format(palette=self.rng.choice(PALETTES))
        return [(f"{p}, {style}" if hint.lower() in p.lower() else f"{p}, {hint}, {style}") for p in pool[:n]]

    def _local_batch(self, prompts):
        if self.local.failed:
            return []
        good, why = self.local.usable()
        if not good:
            warn(f"local SDXL unavailable: {why}")
            self.local.failed = True
            return []
        if self.local.pipe is None:
            try:
                self.local.load()
            except Exception as e:
                warn(f"could not start SDXL-Lightning ({type(e).__name__}: {e})")
                self.local.failed = True
                self.local.release()
                return []
        art, fails, t0 = [], 0, time.time()
        for i, p in enumerate(prompts, 1):
            try:
                art.append(Asset("ai", img=self.local.generate(p, self.rng.randint(1, 999999))))
                fails = 0
            except Exception as e:
                fails += 1
                warn(f"SDXL image {i}/{len(prompts)} failed ({type(e).__name__}: {e})")
                if fails >= 2:
                    break
        info(f"SDXL-Lightning: {len(art)}/{len(prompts)} images in {time.time() - t0:.0f}s")
        return art

    def collect(self, topic, scenes, tag):
        """Returns (ai_assets, real_assets): 3-5 real clips + 4-6 SDXL images (photos backfill any shortage)."""
        cfg, self.dead = self.cfg, False
        n_v, n_a = self.rng.randint(*cfg.n_clips), self.rng.randint(*cfg.n_ai)
        info(f"visual theme: {', '.join(t[0] for t in topic.themes()) or 'generic tech'} | target {n_v} clips + {n_a} AI images")
        queries = topic.photo_queries()
        real = self._pixabay_videos(queries, n_v, tag)
        ai = self._local_batch(self._prompts(topic, scenes, n_a))
        need = max(0, n_v - len(real)) + max(0, n_a - len(ai))
        if need:
            real += self._pixabay_images(queries, need)
        if len(real) + len(ai) < 3:
            warn("not enough images - adding generated backgrounds (check keys / network / GPU)")
            real += [Asset("photo", img=procedural_bg(self.rng, (cfg.width, cfg.height))) for _ in range(3 - len(real) - len(ai))]
        self.seen.save()
        info(f"visuals: {sum(a.kind == 'clip' for a in real)} clips + {sum(a.kind == 'photo' for a in real)} photos + {len(ai)} AI images")
        return ai, real


# =====================================================================
# 7. EDIT PLAN: sentence-synced cuts + interleaving (never >2 AI images in a row)
# =====================================================================
@dataclass
class Shot:
    start: float
    end: float
    asset: Asset
    reuse: int = 0


def plan_cuts(words, cfg, total, rng):
    lo, hi = cfg.cut_range
    ends = [(w.end + cfg.lead_in, w.text[-1:] in ".!?") for w in words]
    cuts, start = [], 0.0
    while total - start > hi:
        cands = [(b, s) for b, s in ends if start + lo <= b <= start + hi and total - b >= lo]
        if cands:
            sent = [c for c in cands if c[1]]
            cut = sent[0][0] if sent else min(cands, key=lambda c: abs(c[0] - start - rng.uniform(2.0, 2.8)))[0]
        else:
            cut = start + (hi if total - start >= hi + lo else (total - start) / 2)
        cuts.append((start, cut))
        start = cut
    cuts.append((start, total))
    return cuts


def plan_shots(cuts, ai, real, rng):
    kinds, p_ai = [], len(ai) / max(1, len(ai) + len(real))
    for i in range(len(cuts)):
        if not real:
            k = "ai"
        elif not ai:
            k = "real"
        elif i == 0:
            k = "ai"                      # strongest visual on the hook
        elif kinds[-2:] == ["ai", "ai"]:
            k = "real"
        else:
            k = "ai" if rng.random() < p_ai else "real"
        kinds.append(k)
    used, shots = {"ai": 0, "real": 0}, []
    for (s, e), k in zip(cuts, kinds):
        pool = ai if k == "ai" else real
        n = used[k]
        used[k] += 1
        shots.append(Shot(s, e, pool[n % len(pool)], n // len(pool)))
    return shots


# =====================================================================
# 8. COMPOSITOR: Ken Burns / clip frames -> FFmpeg (.ass karaoke, vignette, grain, progress bar, watermark, loudnorm)
# =====================================================================
def ass_time(t):
    cs = int(round(max(0.0, t) * 100))
    h, r = divmod(cs, 360000)
    m, r = divmod(r, 6000)
    return f"{h}:{m:02d}:{r // 100:02d}.{r % 100:02d}"


def write_ass(cfg, words, total_s, path):
    """3 words per line, bold white + black outline, the spoken word flashes neon yellow (\\k karaoke tags)."""
    W, H = cfg.width, cfg.height
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
            "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            # Secondary = text before it is spoken (white); Primary = highlight (neon yellow #FFF200, ASS BGR order)
            f"Style: Cap,DejaVu Sans,{cfg.caption_size},&H0000F2FF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,5,70,70,0,1\n"
            "Style: WM,DejaVu Sans,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,7,34,0,30,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    ev = ["Dialogue: 1,%s,%s,WM,,0,0,0,,{\\an7\\pos(34,30)\\alpha&H80&}@%s" % (ass_time(0), ass_time(total_s + 1), cfg.brand)]
    cy, n, lead = int(cfg.caption_center_y * H), cfg.caption_words, cfg.lead_in
    groups = [words[i:i + n] for i in range(0, len(words), n)]
    for gi, g in enumerate(groups):
        t0, t1 = g[0].start + lead, g[-1].end + lead + 0.25
        if gi + 1 < len(groups):
            t1 = min(t1, groups[gi + 1][0].start + lead)
        t1 = max(t1, t0 + 0.2)
        parts = []
        for i, w in enumerate(g):
            a = w.start + lead
            b = (g[i + 1].start if i + 1 < len(g) else w.end) + lead
            b = max(b, a + 0.05)
            off = round((b - t0) * 1000)
            txt = re.sub(r"[{}\\]", "", w.text)
            txt = txt.upper() if cfg.caption_uppercase else txt
            # \k = karaoke highlight; \t returns the word to white once it has been spoken (only the active word is yellow)
            parts.append("{\\k%d\\t(%d,%d,\\1c&H00FFFFFF&)}%s " % (max(1, round((b - a) * 100)), off, off + 1, txt))
        ev.append("Dialogue: 0,%s,%s,Cap,,0,0,0,,{\\an5\\pos(%d,%d)}%s" % (ass_time(t0), ass_time(t1), W // 2, cy, "".join(parts).rstrip()))
    Path(path).write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


class Compositor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = (round(cfg.width * 1.2), round(cfg.height * 1.2))   # 20% headroom so zooming stays sharp

    def _prepare(self, img):
        small = max(img.size) < self.base[1] * 0.8
        img = cover(img.convert("RGB"), self.base)
        return img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=70, threshold=2)) if small else img

    def _box(self, size, cut_no, p):
        bw, bh = size
        e = p * p * (3 - 2 * p)
        z = 1.0 + self.cfg.zoom * (e if cut_no % 2 == 0 else 1 - e)
        vw, vh = bw / z, bh / z
        sx = 1 if (cut_no // 2) % 2 == 0 else -1
        sy = 1 if (cut_no // 3) % 2 == 0 else -1
        cx = bw / 2 + sx * (2 * e - 1) * 0.7 * (bw - vw) / 2
        cy = bh / 2 + sy * (2 * e - 1) * 0.5 * (bh - vh) / 2
        return (cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2)

    def _ken_frames(self, base, cut_no, n):
        W, H = self.cfg.width, self.cfg.height
        for k in range(n):
            yield base.resize((W, H), RESAMPLE.BILINEAR, box=self._box(base.size, cut_no, k / max(1, n - 1))).tobytes()

    def _clip_frames(self, shot, n):
        cfg, a = self.cfg, shot.asset
        size, span = cfg.width * cfg.height * 3, n / cfg.fps
        off = (shot.reuse * 3.3) % max(0.1, a.dur - span - 0.1) if a.dur > span + 0.2 else 0.0
        p = subprocess.Popen([FFMPEG, "-loglevel", "error", "-ss", f"{off:.2f}", "-i", a.path, "-frames:v", str(n), "-an",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=size)
        last = None
        try:
            for _ in range(n):
                b = p.stdout.read(size)
                if len(b) == size:
                    last = b
                yield last or bytes(size)         # clip ended early -> hold the last frame
        finally:
            p.kill()
            p.wait()

    def render(self, shots, words, audio_path, total_frames, out_path, tmp):
        cfg = self.cfg
        W, H, fps = cfg.width, cfg.height, cfg.fps
        total_s, out = total_frames / fps, Path(out_path).resolve()
        wd = Path(tempfile.mkdtemp(dir=tmp))
        write_ass(cfg, words, total_s, wd / "caps.ass")
        fdir = next((os.path.dirname(p) for p in FONT_PATHS if os.path.exists(p)), None)
        steps = max(1, math.ceil(total_s / 0.5))     # thin top retention bar: cumulative drawbox steps every 0.5 s
        bars = ",".join(f"drawbox=x=0:y=0:w={round(W * (k + 1) / steps)}:h=8:color=0xFFF200@0.9:t=fill:enable='gte(t,{k * 0.5:.2f})'"
                        for k in range(steps))
        vf = ("format=yuv420p,vignette=PI/5,noise=alls=6:allf=t+u,ass=caps.ass" + (f":fontsdir={fdir}" if fdir else "") + "," + bars)
        maxrate = int(min(6_000_000, cfg.max_file_mb * 8e6 / total_s - 250_000))
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "-",
               "-i", str(audio_path), "-map", "0:v", "-map", "1:a", "-vf", vf,
               "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=44100",
               "-c:v", "libx264", "-preset", cfg.encode_preset, "-crf", str(cfg.video_crf), "-maxrate", str(maxrate),
               "-bufsize", str(2 * maxrate), "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart",
               "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]
        bases = {}
        for s in shots:
            if s.asset.img is not None and id(s.asset) not in bases:
                bases[id(s.asset)] = self._prepare(s.asset.img)
        info(f"rendering {total_frames} frames ({total_s:.1f}s, {len(shots)} cuts)...")
        log_path = Path(str(out) + ".log")
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log, cwd=wd)
            try:
                fi, nxt = 0, 0.25
                for si, s in enumerate(shots):
                    end = total_frames if si == len(shots) - 1 else min(total_frames, round(s.end * fps))
                    n = end - fi
                    if n <= 0:
                        continue
                    frames = self._clip_frames(s, n) if s.asset.img is None else self._ken_frames(bases[id(s.asset)], si + s.reuse, n)
                    for fr in frames:
                        proc.stdin.write(fr)
                    fi = end
                    if fi / total_frames >= nxt:
                        info(f"  render {int(nxt * 100)}%")
                        nxt += 0.25
                proc.stdin.close()
            except BrokenPipeError:
                pass
            except BaseException:
                proc.kill()
                raise
            rc = proc.wait()
        shutil.rmtree(wd, ignore_errors=True)
        if rc != 0:
            raise RuntimeError("ffmpeg failed: " + log_path.read_text(errors="ignore")[-600:])
        log_path.unlink(missing_ok=True)


def draw_center(d, cx, cy, text, font, fill, sw=0):
    b = d.textbbox((0, 0), text, font=font, stroke_width=sw)
    d.text((cx - (b[2] + b[0]) / 2, cy - (b[3] + b[1]) / 2), text, font=font, fill=fill, stroke_width=sw, stroke_fill=(0, 0, 0, 255))


def make_thumbnail(cfg, img, hook, out):
    """1080x1920 {stem}_thumb.jpg: hook frame + bold 3-4 word banner + @brand badge."""
    W, H = cfg.width, cfg.height
    arr = np.asarray(cover(img.convert("RGB"), (W, H)), np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    arr = arr * (1 - 0.55 * np.clip((y - 0.5) / 0.5, 0, 1) - 0.3 * np.clip((0.2 - y) / 0.2, 0, 1))
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im)
    words = hook.upper().split()[:4] or ["WATCH", "THIS"]
    lines = [" ".join(words[:2]), " ".join(words[2:])] if len(words) > 2 else [" ".join(words)]
    size = 170
    while size > 60 and max(d.textlength(l, font=get_font(cfg, size)) for l in lines) > W - 200:
        size -= 8
    font, lh = get_font(cfg, size), int(size * 1.35)
    y0 = int(H * 0.46 - (lh + 14) * len(lines) / 2)
    for i, l in enumerate(lines):
        w, top = d.textlength(l, font=font), y0 + i * (lh + 14)
        d.rounded_rectangle([(W - w) / 2 - 40, top, (W + w) / 2 + 40, top + lh], radius=30, fill=(0, 0, 0, 215))
        draw_center(d, W / 2, top + lh / 2, l, font, (255, 242, 0, 255), 6)
    badge, bf = "@" + cfg.brand, get_font(cfg, 56)
    bw = d.textlength(badge, font=bf)
    d.rounded_rectangle([(W - bw) / 2 - 38, H - 250, (W + bw) / 2 + 38, H - 150], radius=50, fill=(255, 242, 0, 255))
    draw_center(d, W / 2, H - 200, badge, bf, (0, 0, 0, 255))
    im.convert("RGB").save(out, quality=92)


# =====================================================================
# 9. TELEGRAM (sendVideo + sendPhoto + sendDocument over plain HTTPS)
# =====================================================================
class Telegram:
    def __init__(self, token, chat_id, base="https://api.telegram.org"):
        self.token, self.chat, self.base = token, chat_id, base

    def _redact(self, s):
        return str(s).replace(self.token, "***")

    def check(self):
        try:
            j = http_get(f"{self.base}/bot{self.token}/getMe", timeout=(8, 15), retries=1).json()
            return bool(j.get("ok")), (j.get("result") or {}).get("username", "")
        except Exception as e:
            return False, self._redact(e)

    def _send(self, method, field, path, data, mime):
        url, err = f"{self.base}/bot{self.token}/{method}", None
        data = {"chat_id": self.chat, "parse_mode": "HTML", **data}
        for attempt in range(3):
            try:
                with open(path, "rb") as f:
                    r = SESSION.post(url, data=data, files={field: (Path(path).name, f, mime)}, timeout=(15, 900))
                try:
                    j = r.json()
                except ValueError:
                    j = {}
                if j.get("ok"):
                    return j
                err = f"HTTP {r.status_code}: {j.get('description', r.text[:120])}"
                if r.status_code == 429:
                    time.sleep(int((j.get("parameters") or {}).get("retry_after", 5)) + 1)
                    continue
                break
            except requests.RequestException as e:
                err = self._redact(e)
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"Telegram {method} failed - {err}")

    def send_video(self, path, caption, duration, w, h):
        return self._send("sendVideo", "video", path, {"caption": caption, "supports_streaming": "true",
                                                       "duration": int(round(duration)), "width": w, "height": h}, "video/mp4")

    def send_photo(self, path, caption):
        return self._send("sendPhoto", "photo", path, {"caption": caption}, "image/jpeg")

    def send_document(self, path, caption):
        return self._send("sendDocument", "document", path, {"caption": caption}, "text/plain")


# =====================================================================
# 10. PIPELINE - staged so only one model is ever on the GPU
# =====================================================================
@dataclass
class Job:
    idx: int
    topic: Topic
    script: Script = None
    voice: Voiceover = None
    ai: list = None
    real: list = None
    total_frames: int = 0
    path: str = None
    thumb: str = None
    pkg: str = None
    sent: bool = False
    error: str = ""


def run_stage(jobs, label, fn, workers=1):
    print(f"\n==================== {label} ====================")

    def one(j):
        print(f"--- short {j.idx}/{len(jobs)}: {j.topic.title[:80]}")
        try:
            fn(j)
        except Exception as e:
            j.error = f"{label.split(': ')[-1]} failed: {type(e).__name__}: {e}"
            warn(f"short {j.idx} dropped - {j.error}")
            traceback.print_exc()

    live = [j for j in jobs if not j.error]
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, live))
    else:
        for j in live:
            one(j)


def hook_image(j, tmp):
    """Strongest hook frame = the first SDXL image (the hook visual); else a photo; else a frame from the first clip."""
    for a in (j.ai or []) + (j.real or []):
        if a.img is not None:
            return a.img
    jpg = Path(tmp) / f"hook_{j.idx}.jpg"
    run_ff(["-ss", "0.8", "-i", j.real[0].path, "-frames:v", "1", str(jpg)])
    return Image.open(jpg).convert("RGB")


def finish_short(cfg, comp, j, tmp, rng, np_rng):
    """Cuts -> audio mastering -> FFmpeg render -> thumbnail -> post package."""
    total_s = j.total_frames / cfg.fps
    shots = plan_shots(plan_cuts(j.voice.words, cfg, total_s, rng), j.ai, j.real, rng)
    if not j.real and j.ai:
        warn("no real footage available - AI images may run back to back")
    music = fit_music(total_s, cfg, rng, np_rng)
    wav = Path(tmp) / f"mix_{j.idx}.wav"
    write_wav(wav, mix_audio(j.voice.pcm, music, cfg, j.total_frames, plan_sfx(shots, np_rng)))
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"short_{datetime.now():%Y%m%d_%H%M%S}_{j.idx}"
    comp.render(shots, j.voice.words, wav, j.total_frames, out_dir / f"{stem}.mp4", tmp)
    j.path = str(out_dir / f"{stem}.mp4")
    j.thumb, j.pkg = str(out_dir / f"{stem}_thumb.jpg"), str(out_dir / f"{stem}_post_package.txt")
    make_thumbnail(cfg, hook_image(j, tmp), j.script.hook, j.thumb)
    write_post_package(cfg, j.script, j.topic, j.pkg)
    j.ai = j.real = None


def preflight(cfg, eng, tg, pix_key):
    if not FFMPEG:
        raise RuntimeError("ffmpeg not found - run Cell 1")
    for mod, pkg in (("kokoro", "kokoro"), ("faster_whisper", "faster-whisper")):
        if not importlib.util.find_spec(mod):
            raise RuntimeError(f"{pkg} missing - run Cell 1")
    eng["script"].setup()
    eng["visual"].plan()
    if tg:
        good, who = tg.check()
        (ok if good else warn)(f"Telegram bot check: {'@' + who if good else who}")
    elif cfg.send_to_telegram:
        warn("TELEGRAM_BOT_TOKEN / TELEGRAM_CHANNEL missing - files will only be saved to " + cfg.out_dir)
    if not pix_key:
        warn("no PIXABAY_API_KEY - no real B-roll; visuals will be AI-only / generated backgrounds")


def run_pipeline(cfg=None):
    cfg = cfg or CFG
    rng, np_rng = random.Random(cfg.seed), np.random.default_rng(cfg.seed)
    pix_key = get_secret("PIXABAY_API_KEY")
    token, chat = get_secret("TELEGRAM_BOT_TOKEN"), get_secret("TELEGRAM_CHANNEL")
    tg = Telegram(token, chat) if (cfg.send_to_telegram and token and chat) else None

    with tempfile.TemporaryDirectory(prefix="shorts_") as tmp:
        eng = {"script": ScriptEngine(cfg, rng), "voice": VoiceEngine(cfg), "comp": Compositor(cfg),
               "visual": VisualEngine(cfg, rng, pix_key, tmp)}
        preflight(cfg, eng, tg, pix_key)
        info(f"collecting {cfg.num_shorts} topics ({cfg.topic_source})...")
        jobs = [Job(i, t) for i, t in enumerate(TopicEngine(cfg).get_topics(cfg.num_shorts), 1)]

        def do_script(j):
            j.script = eng["script"].generate(j.topic, HOOK_STYLES[(j.idx - 1) % len(HOOK_STYLES)])

        def do_voice(j):
            j.voice = eng["voice"].speak(j.script.text, tmp, j.idx)
            j.total_frames = math.ceil((cfg.lead_in + j.voice.duration + cfg.tail) * cfg.fps)

        def do_align(j):
            j.voice.words, j.voice.timing = eng["voice"].align(j.script.text, j.voice.wav, j.voice.duration)
            ok(f"{len(j.voice.words)} words aligned ({j.voice.timing})")

        def do_visuals(j):
            j.ai, j.real = eng["visual"].collect(j.topic, j.script.scenes, j.idx)

        def do_render(j):
            t0 = time.time()
            finish_short(cfg, eng["comp"], j, tmp, rng, np_rng)
            ok(f"rendered {Path(j.path).name} ({Path(j.path).stat().st_size / 1048576:.1f} MB) + thumbnail + post package in {time.time() - t0:.0f}s")
            if not tg:
                return
            cap = f"<b>{html.escape(j.script.title)}</b>\n\n{html.escape(j.script.tags)}\n@{cfg.brand}"
            total_s = j.total_frames / cfg.fps
            for name, fn in (("thumbnail", lambda: tg.send_photo(j.thumb, cap)),
                             ("video", lambda: tg.send_video(j.path, cap, total_s, cfg.width, cfg.height)),
                             ("post package", lambda: tg.send_document(j.pkg, "Post package (title, description, pinned comment, disclosure, tags)"))):
                try:
                    fn()
                    ok(f"sent {name} to Telegram")
                    j.sent = j.sent or name == "video"
                except Exception as e:
                    warn(f"{e}")
            if j.sent and cfg.delete_after_send:
                Path(j.path).unlink(missing_ok=True)

        run_stage(jobs, "STAGE 1/5: researching + writing scripts (cloud LLM)", do_script)
        run_stage(jobs, "STAGE 2/5: Kokoro-82M voice-overs", do_voice)
        eng["voice"].release_tts()
        run_stage(jobs, "STAGE 3/5: Whisper word alignment", do_align)
        eng["voice"].release_stt()
        run_stage(jobs, "STAGE 4/5: Pixabay B-roll + SDXL-Lightning art", do_visuals)
        eng["visual"].release()
        run_stage(jobs, "STAGE 5/5: FFmpeg render + thumbnail + posting", do_render, workers=cfg.render_workers)

    print("\n==================== SUMMARY ====================")
    results = []
    for j in jobs:
        state = ("sent" if j.sent else "saved " + str(j.path)) if j.path else "FAILED - " + j.error
        print(f"{j.idx}. {j.topic.title[:60]}  ->  {state}")
        results.append({"topic": j.topic.title, "title": j.script.title if j.script else None, "path": j.path,
                        "thumb": j.thumb, "post_package": j.pkg, "sent": j.sent, "error": j.error})
    return results


# =====================================================================
# 11. SELFTEST - 5 s video + thumbnail + post package, no GPU / network / keys
# =====================================================================
def selftest(cfg=None):
    cfg = replace(cfg or CFG, out_dir=str(Path(tempfile.gettempdir()) / "shorts_selftest"), tail=0.6, seed=7, send_to_telegram=False)
    if not FFMPEG:
        raise RuntimeError("ffmpeg not found")
    rng, np_rng = random.Random(7), np.random.default_rng(7)
    W, H = cfg.width, cfg.height
    text = "OPsblueprint self test. Captions, grain, cuts, sound and thumbnail all render. Everything works fine."
    words = spread(text.split(), 0.1, 3.9)
    pcm = np.zeros(int(4.0 * SR), np.float32)           # synthetic "voice": a tone burst per word
    tt = np.arange(len(pcm)) / SR
    for w in words:
        i, k = int(w.start * SR), int(w.end * SR)
        pcm[i:k] += (0.3 * np.sin(2 * np.pi * 170 * tt[i:k]) * np.hanning(k - i)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="selftest_") as tmp:
        clip = Path(tmp) / "clip.mp4"
        run_ff(["-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={cfg.fps}", "-t", "3", "-pix_fmt", "yuv420p", str(clip)])
        script = Script(text, text, "Selftest: the renderer works", "#Tech #Shorts #AI", [], "Renderer self test.",
                        "Every tool I use is at the first link in my bio.", "SELF TEST PASSED")
        j = Job(1, Topic("Selftest", "", "selftest"), script, Voiceover(pcm, words, 4.0, "synthetic"))
        j.total_frames = math.ceil((cfg.lead_in + 4.0 + cfg.tail) * cfg.fps)
        j.ai = [Asset("ai", img=procedural_bg(rng, (W, H))) for _ in range(3)]
        j.real = [Asset("clip", path=str(clip), dur=3.0)]
        finish_short(cfg, Compositor(cfg), j, tmp, rng, np_rng)
    for p in (j.path, j.thumb, j.pkg):
        if not Path(p).exists() or Path(p).stat().st_size < 500:
            raise RuntimeError(f"selftest failed: {p} missing or empty")
        ok(f"selftest output: {p} ({Path(p).stat().st_size / 1024:.0f} KB)")
    ok("SELFTEST PASSED")
    return j.path, j.thumb, j.pkg


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(CFG)
    else:
        run_pipeline()
