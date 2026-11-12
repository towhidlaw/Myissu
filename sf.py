# TECH SHORTS FACTORY v5 — Final Monetization & Affiliate Edition (`OPsblueprint`)

```python
# =====================================================================
# TECH SHORTS FACTORY v5  —  FINAL MONETIZATION & AFFILIATE EDITION
# Brand: @OPsblueprint  |  3 Shorts per run  |  Colab-safe (T4, free tier)
#
# PIPELINE
#   Google Tech News RSS + Hacker News + Wikipedia facts
#     -> CloudLLM script brain (Gemini API, GitHub Models fallback)  [NO local LLM, NO Ollama]
#     -> Kokoro-82M local neural TTS  -> faster-whisper word timestamps (then both unloaded)
#     -> Hybrid visuals: real Pixabay MP4 b-roll + local SDXL-Lightning AI art (Ken Burns)
#     -> FFmpeg compositor: .ass karaoke captions, progress bar, film grain, @OPsblueprint watermark
#     -> Studio audio master (synth BGM + whoosh/riser/drop SFX + sidechain duck + loudnorm)
#     -> Auto thumbnail + copy-paste post package -> Telegram (video + photo + document)
#
# ---------------------------------------------------------------------
# COLAB CELL 1  (exact, conflict-free — does NOT touch torch / numpy / CUDA)
# ---------------------------------------------------------------------
# !apt-get -qq update && apt-get -qq install -y ffmpeg espeak-ng fonts-dejavu-core
# !pip -q install "kokoro>=0.9.4" "misaki[en]" soundfile
# !pip -q install faster-whisper
# !pip -q install -U diffusers peft accelerate safetensors huggingface_hub
# !pip -q install requests pillow
# # (torch, torchvision, numpy stay exactly as Colab ships them)
#
# COLAB SECRETS (Key icon, left sidebar):
#   GEMINI_API_KEY        primary script brain   (https://aistudio.google.com/apikey)
#   GITHUB_TOKEN          fallback script brain  (GitHub Models, gpt-4o-mini)
#   PIXABAY_API_KEY       real b-roll video + photo backfill
#   TELEGRAM_BOT_TOKEN / TELEGRAM_CHANNEL   delivery
#
# RUN:   python factory_v5.py            |    python factory_v5.py --selftest
#        (in Colab:  run_pipeline()      |    selftest(CFG) )
# =====================================================================
import argparse
import gc
import html
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
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RESAMPLE = getattr(Image, "Resampling", Image)
SR = 44100  # master audio sample rate


# =====================================================================
# 0. CONFIG
# =====================================================================
@dataclass
class Config:
    # ---- run ----
    brand: str = "OPsblueprint"
    num_shorts: int = 3
    render_workers: int = 1                 # 1 = no parallel renders (prevents Colab RAM crashes)
    out_dir: str = "/content/shorts" if os.path.isdir("/content") else "./shorts_out"
    send_to_telegram: bool = True
    delete_after_send: bool = False         # keep thumb + package locally by default
    seed: int = None

    # ---- topics ----
    topic_source: str = "tech"              # "tech" | "evergreen"

    # ---- cloud script brain ----
    gemini_models: tuple = ("gemini-2.5-flash", "gemini-2.5-flash-lite")  # 2nd used on HTTP 429
    github_model: str = "gpt-4o-mini"
    script_words: tuple = (125, 145)
    script_attempts: int = 3
    scenes_per_script: int = 8

    # ---- affiliate / bio funnel ----
    bio_cta: str = ("If you want the exact AI and security tools I actually run, "
                    "they are at the first link in my bio.")
    affiliate_note: str = "Some links in my bio are affiliate links."

    # ---- voice (Kokoro-82M local) ----
    kokoro_voice: str = "am_michael"        # am_michael / am_adam / af_heart ...
    kokoro_speed: float = 1.04
    whisper_model: str = "base.en"
    lead_in: float = 0.35
    tail: float = 0.9

    # ---- music / mix ----
    bgm_db: float = -11.0
    duck_db: float = -6.0
    sfx_db: float = -9.0

    # ---- visuals ----
    real_clips: tuple = (3, 5)              # Pixabay MP4 b-roll per short
    ai_images: tuple = (4, 6)               # local SDXL-Lightning images per short
    local_image_size: tuple = (768, 1344)
    pixabay_categories: tuple = ("computer", "science")
    cut_range: tuple = (1.5, 3.2)           # variable sentence-synced cut length
    zoom: float = 0.16

    # ---- video ----
    width: int = 1080
    height: int = 1920
    fps: int = 30
    caption_words: int = 3
    caption_uppercase: bool = True
    caption_size: int = 92
    video_crf: int = 20
    encode_preset: str = "veryfast"
    max_file_mb: int = 45


CFG = Config()


# =====================================================================
# 1. UTILITIES
# =====================================================================
def info(m): print(f"[*] {m}", flush=True)
def ok(m): print(f"[+] {m}", flush=True)
def warn(m):
    m = str(m)
    print(f"[-] {m if len(m) <= 330 else m[:327] + '...'}", flush=True)


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) OPsblueprintFactory/5.0"})


def http_get(url, *, params=None, headers=None, timeout=(8, 25), retries=2):
    last = None
    for a in range(retries + 1):
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and a < retries:
                time.sleep(2 * (a + 1)); continue
            return r
        except requests.RequestException as e:
            last = e
            if a < retries:
                time.sleep(1.5 * (a + 1))
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
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def find_font():
    for p in FONTS:
        if os.path.exists(p):
            return p
    try:
        import matplotlib
        p = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    return None


FONT_PATH = find_font()


def ff(args, cwd=None, timeout=1800):
    """Run ffmpeg, raise with tail of stderr on failure."""
    p = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error"] + [str(a) for a in args],
                       capture_output=True, cwd=cwd, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg: " + p.stderr.decode("utf-8", "ignore")[-500:])
    return p


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


_EMOJI = re.compile("[\U00010000-\U0010ffff\u2600-\u27bf\u2b00-\u2bff\ufe0f\u200d]")


def clean_for_speech(text):
    t = _EMOJI.sub("", str(text or ""))
    t = t.translate({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"',
                     0x2026: "...", 0x2013: "-", 0x2014: ", "})
    t = re.sub(r"(?im)(^\s*|[.!?]\s+)(?:hook|script|cta|outro|intro|narrator|voice[- ]?over)\s*[:\-]\s*", r"\1", t)
    t = re.sub(r"\((?:pause|beat|music|sfx)[^)]*\)", " ", t, flags=re.I)
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"[*`~>|\"]+", "", t)
    t = t.replace("&", " and ")
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t


def parse_json_loose(s):
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    try:
        return json.loads(s, strict=False)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return json.loads(m.group(0), strict=False)
        except Exception:
            pass
    m = re.search(r'"script"\s*:\s*"((?:[^"\\]|\\.)*)"', s, re.S)
    return {"script": m.group(1)} if m else None


_SCENE_BAN = re.compile(r"\b(text|logos?|captions?|titles?|words?|letters?|watermark|slogan|writing)\b", re.I)


def clean_scene(s):
    s = re.sub(r'"[^"]*"', " ", str(s))
    s = _SCENE_BAN.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.;:")
    return " ".join(s.split()[:22])


def normalize_tags(raw, extra=("#Tech", "#AI", "#Shorts")):
    items = re.findall(r"#?\w+", raw) if isinstance(raw, str) else [str(x) for x in (raw or [])]
    tags = []
    for x in items:
        x = re.sub(r"\W", "", x)[:28]
        if x and ("#" + x).lower() not in [t.lower() for t in tags]:
            tags.append("#" + x)
    for e in extra:
        if len(tags) < 5 and e.lower() not in [t.lower() for t in tags]:
            tags.append(e)
    return " ".join(tags[:8])


STOP = set("""a an and are as at be but by for from has have how i in into is it its of on or our that the
their this to was we what when where which who why will with you your new says said after over about more
than up out us not can just now amid vs""".split())


def keywords(text, k=4):
    toks = [t.strip(".-'") for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9+.\-']*", text or "")]
    toks = [t for t in toks if t and t.lower() not in STOP]
    strong = [t for t in toks if t[0].isupper() or any(c.isdigit() for c in t)]
    out, seen = [], set()
    for t in strong + [t for t in toks if t not in strong]:
        if t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return out[:k]


def cover(img, size):
    tw, th = size
    sw, sh = img.size
    s = max(tw / sw, th / sh)
    nw, nh = max(tw, math.ceil(sw * s)), max(th, math.ceil(sh * s))
    img = img.resize((nw, nh), RESAMPLE.LANCZOS)
    l, t = (nw - tw) // 2, (nh - th) // 2
    return img.crop((l, t, l + tw, t + th))


def decode_audio(path):
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "f32le",
                          "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").copy()


def write_wav(path, stereo):
    pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(stereo.shape[1]); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# =====================================================================
# 2. TOPICS + GROUNDING FACTS  (Google Tech News RSS + Hacker News + Wikipedia)
# =====================================================================
TECH_THEMES = [
    ("ai", r"\b(ai|artificial intelligence|openai|chatgpt|gpt-?\d*|llms?|gemini|claude|anthropic|copilot|deepseek|grok|agents?|neural)\b",
     ["artificial intelligence", "neural network", "machine learning", "robot"], "computer",
     ["glowing neural network of light filaments in a dark void",
      "humanoid robot face made of circuitry with cyan eyes",
      "holographic brain of data streams above a dark desk"]),
    ("chips", r"\b(chips?|nvidia|amd|intel|cpus?|gpus?|semiconductors?|tsmc|silicon|blackwell|rtx)\b",
     ["microchip", "circuit board", "computer processor"], "computer",
     ["macro shot of a glowing silicon chip on a dark circuit board",
      "silicon wafer reflecting rainbow light in a clean room",
      "circuit board city skyline from above with glowing traces"]),
    ("security", r"\b(hack(?:ed|ers?|ing)?|breach(?:es)?|cyber\w*|ransomware|malware|security|vulnerabilit\w+|privacy|encryption|phishing|deepfakes?)\b",
     ["cyber security", "hacker", "data security"], "computer",
     ["glowing padlock of light over a dark digital grid",
      "hooded silhouette in front of cascading green code",
      "broken digital padlock with sparks in a dark server room"]),
    ("phone", r"\b(iphone|android|samsung|galaxy|pixel|smartphones?|phones?|ios|foldables?)\b",
     ["smartphone", "mobile technology", "smartphone camera"], "computer",
     ["flagship smartphone floating above a dark reflective surface",
      "macro shot of a smartphone camera lens array glowing softly",
      "exploded view of phone components hovering in mid air"]),
    ("cloud", r"\b(cloud|aws|azure|data ?centers?|servers?|networks?|5g|internet|starlink|outage)\b",
     ["data center", "server room", "cloud computing"], "computer",
     ["endless rows of server racks with glowing blue lights",
      "fiber optic cables with bright pulses of light in the dark",
      "aerial view of a giant data center campus at night"]),
    ("robots", r"\b(robots?|robotics|humanoids?|drones?|automation|optimus)\b",
     ["robot", "humanoid robot", "drone"], "computer",
     ["humanoid robot in a dark futuristic lab with rim light",
      "close-up of a robotic hand holding a glowing orb",
      "swarm of drones flying over a neon city at night"]),
    ("auto", r"\b(tesla|evs?|electric vehicles?|self-driving|autonomous|waymo|batter(?:y|ies))\b",
     ["electric car technology", "autonomous vehicle", "battery technology"], "computer",
     ["sleek electric concept car in a dark futuristic garage",
      "electric car charging at night with a glowing blue cable",
      "self-driving car scanning a neon street with light beams"]),
    ("space", r"\b(spacex|nasa|rockets?|satellites?|starship|space|moon|mars)\b",
     ["space technology", "rocket launch", "satellite"], "science",
     ["rocket launching at night with a bright exhaust plume",
      "satellite orbiting Earth with the sun rising behind it",
      "futuristic space station orbiting a blue planet"]),
    ("quantum", r"\b(quantum)\b", ["quantum computer", "futuristic technology"], "science",
     ["quantum computer chandelier of golden cables in a cryogenic lab",
      "glowing qubits floating as spheres of light on dark background"]),
    ("software", r"\b(software|code|coding|developers?|github|programming|linux|open[- ]source|api|browsers?)\b",
     ["programming code", "software development"], "computer",
     ["glowing lines of code floating in dark space",
      "developer silhouette at a desk with glowing monitors of code"]),
]
GENERIC_TECH = ["technology", "artificial intelligence", "futuristic technology", "data center", "circuit board"]
GENERIC_SCENES = ["futuristic data center with glowing blue servers",
                  "holographic interface floating above a dark desk",
                  "close-up of a glowing microchip with golden circuits",
                  "neon city skyline at night with flowing data streams",
                  "abstract network of light nodes on a dark background"]
_THEME_RX = [(n, re.compile(rx, re.I), terms, cat, sc) for n, rx, terms, cat, sc in TECH_THEMES]


@dataclass
class Topic:
    title: str
    context: str
    source: str = "news"

    def themes(self):
        text = self.title + " " + self.context.split("\n")[0]
        scored = []
        for i, (name, rx, terms, cat, sc) in enumerate(_THEME_RX):
            n = len(rx.findall(text))
            if n:
                scored.append((-n, i, name, terms, cat, sc))
        scored.sort()
        out = [s[2:] for s in scored[:1]]
        if len(scored) > 1:
            out.append(scored[1][2:])
        return out

    def queries(self):
        qs, seen = [], set()
        for th in self.themes():
            cat = th[2] if th[2] in CFG.pixabay_categories else "computer"
            for t in th[1]:
                if t.lower() not in seen:
                    seen.add(t.lower()); qs.append((t, cat))
        for g in GENERIC_TECH:
            if g.lower() not in seen:
                seen.add(g.lower()); qs.append((g, "computer"))
        return qs

    def theme_scenes(self):
        return [s for th in self.themes() for s in th[3]]

    def hint(self):
        th = self.themes()
        return th[0][1][0] if th else "technology"


class TopicEngine:
    NEWS = "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en"
    HN_TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
    HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{}.json"
    EVERGREEN = ["AI Agents Quietly Rewriting Everyday Workflows",
                 "Humanoid Robots Entering Global Manufacturing",
                 "Why Data Centers Became the New Power Plants",
                 "Deepfake Scams and the Collapse of Digital Trust",
                 "Quantum Computing's Race Against Modern Encryption"]

    def __init__(self, cfg):
        self.cfg = cfg

    def _news(self, limit):
        r = http_get(self.NEWS); r.raise_for_status()
        out = []
        for item in ET.fromstring(r.content).findall(".//item"):
            raw = (item.findtext("title") or "").strip()
            if not raw:
                continue
            headline, _, source = raw.rpartition(" - ")
            headline = headline or raw
            desc = item.findtext("description") or ""
            rel = [html.unescape(re.sub(r"<[^>]+>", "", m)).strip()
                   for m in re.findall(r"<a [^>]*>(.*?)</a>", desc, re.S)]
            rel = [x for x in rel if x and x.lower() != headline.lower()][:3]
            ctx = f"News headline: {headline}" + (f" (via {source})" if source else "")
            if rel:
                ctx += ". Other coverage of the same story: " + " | ".join(rel)
            out.append(Topic(headline, ctx, "tech-news"))
            if len(out) >= limit:
                break
        return out

    def _hn(self, limit):
        ids = http_get(self.HN_TOP).json()[:40]
        out = []
        for i in ids:
            if len(out) >= limit:
                break
            try:
                it = http_get(self.HN_ITEM.format(i), timeout=(5, 8), retries=1).json()
            except Exception:
                continue
            if not it or it.get("type") != "story":
                continue
            title = re.sub(r"(?i)^show hn:\s*", "", (it.get("title") or "").strip())
            if not title or re.match(r"(?i)^(ask|tell) hn", title):
                continue
            dom = re.sub(r"^www\.", "", re.sub(r"https?://([^/]+).*", r"\1", it.get("url") or ""))
            ctx = (f"Hacker News front page: {title}"
                   + (f" (source {dom}, {it.get('score', 0)} points, {it.get('descendants', 0)} comments)" if dom else ""))
            out.append(Topic(title, ctx, "hacker-news"))
        return out

    @staticmethod
    def wikipedia(title):
        kw = keywords(title, 3)
        if not kw:
            return ""
        try:
            hits = http_get("https://en.wikipedia.org/w/api.php", timeout=(5, 9), retries=1,
                            params={"action": "query", "list": "search", "srsearch": " ".join(kw),
                                    "srlimit": 3, "format": "json"}).json().get("query", {}).get("search", [])
        except Exception:
            return ""
        tw = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", title)}
        for h in hits:
            if not any(w.lower() in tw for w in re.findall(r"[A-Za-z0-9]+", h["title"])):
                continue
            try:
                s = http_get("https://en.wikipedia.org/api/rest_v1/page/summary/"
                             + quote(h["title"].replace(" ", "_"), safe=""), timeout=(5, 9), retries=1).json()
            except Exception:
                continue
            ext = (s.get("extract") or "").strip()
            if s.get("type") == "standard" and ext:
                if len(ext) > 360:
                    ext = ext[:360].rsplit(". ", 1)[0].rstrip(".") + "."
                return f"Background (Wikipedia, may predate the news): {ext}"
        return ""

    def get_topics(self, n):
        topics = []

        def dup(t):
            a = set(re.findall(r"\w+", t.title.lower()))
            for o in topics:
                b = set(re.findall(r"\w+", o.title.lower()))
                if a and b and len(a & b) / len(a | b) >= 0.5:
                    return True
            return False

        if self.cfg.topic_source == "tech":
            for fn in (self._news, self._hn):
                if len(topics) >= n:
                    break
                try:
                    found = fn(n * 4)
                    ok(f"{fn.__name__.strip('_')}: {len(found)} candidates")
                except Exception as e:
                    warn(f"{fn.__name__.strip('_')} failed ({type(e).__name__}: {e})")
                    continue
                for t in found:
                    if len(topics) < n and not dup(t):
                        topics.append(t)
        if len(topics) < n:
            pool = [e for e in self.EVERGREEN if not dup(Topic(e, ""))]
            random.shuffle(pool)
            for e in pool[: n - len(topics)]:
                topics.append(Topic(e, f"Evergreen topic: {e}. No breaking news - stay general, no invented stats.",
                                    "evergreen"))
        for t in topics:
            if t.source != "evergreen":
                bg = self.wikipedia(t.title)
                if bg:
                    t.context += "\n" + bg
        return topics[:n]


# =====================================================================
# 3. CLOUD SCRIPT BRAIN  (Gemini primary -> GitHub Models fallback)
# =====================================================================
@dataclass
class Script:
    text: str
    body: str
    title: str
    tags: str
    scenes: list
    hook: str = ""


SYSTEM_PROMPT = (
    "You are the lead writer and on-camera voice of a sharp, first-person investigative tech channel. "
    "You write 45-55 second vertical voice-over scripts that hold attention to the last word. "
    "You are strictly accurate: you only state facts present in the FACTS block, and when facts are thin you "
    "explain the stakes instead of inventing numbers, dates, names or quotes. "
    "Reply with exactly one valid JSON object and nothing else.")

HOOK_STYLES = [
    "a bold claim that sounds too big to be true but is backed by the facts",
    "a sharp question aimed straight at the viewer",
    "a 'nobody is talking about this' reveal",
    "a warning about what the viewer is about to get wrong",
    "a fast 'here is what just happened' tease that promises a twist",
]

CTA_RX = re.compile(r"\b(follow|subscribe|comment|like and|share this|link in|bio|notifications?)\b", re.I)


def strip_trailing_cta(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    while len(sents) > 3 and CTA_RX.search(sents[-1]):
        sents.pop()
    return " ".join(sents)


class CloudLLM:
    """Google AI Studio (Gemini) primary, GitHub Models (gpt-4o-mini) fallback. requests.post only."""
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    GITHUB_URL = "https://models.inference.ai.azure.com/chat/completions"

    def __init__(self, cfg):
        self.cfg = cfg
        self.gemini_key = get_secret("GEMINI_API_KEY")
        self.github_token = get_secret("GITHUB_TOKEN")
        self.last = ""

    def available(self):
        return bool(self.gemini_key or self.github_token)

    def label(self):
        if self.gemini_key:
            return f"Gemini ({self.cfg.gemini_models[0]}) + GitHub Models fallback" if self.github_token \
                else f"Gemini ({self.cfg.gemini_models[0]})"
        return f"GitHub Models ({self.cfg.github_model})"

    def _gemini(self, model, system, user):
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0,
                                 "maxOutputTokens": 2048},
        }
        r = SESSION.post(self.GEMINI_URL.format(model=model), params={"key": self.gemini_key},
                         json=payload, timeout=(10, 180))
        if r.status_code == 429:
            raise RuntimeError("429 rate limited")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:180]}")
        j = r.json()
        parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    def _github(self, system, user):
        payload = {"model": self.cfg.github_model, "temperature": 1.0,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        r = SESSION.post(self.GITHUB_URL, headers={"Authorization": f"Bearer {self.github_token}",
                                                   "Content-Type": "application/json"},
                         json=payload, timeout=(10, 180))
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:180]}")
        return r.json()["choices"][0]["message"]["content"]

    def chat_json(self, system, user):
        errs = []
        if self.gemini_key:
            for m in self.cfg.gemini_models:
                try:
                    out = self._gemini(m, system, user)
                    self.last = f"gemini:{m}"
                    return parse_json_loose(out)
                except Exception as e:
                    errs.append(f"gemini/{m}: {e}")
                    warn(f"Gemini {m} failed ({e}) - trying next")
        if self.github_token:
            try:
                out = self._github(system, user)
                self.last = f"github:{self.cfg.github_model}"
                return parse_json_loose(out)
            except Exception as e:
                errs.append(f"github: {e}")
                warn(f"GitHub Models failed ({e})")
        raise RuntimeError("cloud LLM unavailable -> " + " | ".join(errs[-2:]))


class ScriptEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.llm = CloudLLM(cfg)

    def setup(self):
        if not self.llm.available():
            warn("no GEMINI_API_KEY and no GITHUB_TOKEN - scripts cannot be written (add a Colab secret)")
            return False
        ok(f"script brain: {self.llm.label()}")
        return True

    def _prompt(self, topic, hook, note=""):
        lo, hi = self.cfg.script_words
        n = self.cfg.scenes_per_script
        return f"""TOPIC: {topic.title}

FACTS (the only facts you may state - never invent numbers, dates, names or quotes):
{topic.context}

Write the spoken BODY of a vertical short: {lo}-{hi} words, first person, investigative creator voice.
Structure:
1. HOOK (1-2 sentences): {hook}. Must stop the scroll in two seconds.
2. WHAT HAPPENED (2-3 sentences): the facts, plainly.
3. WHY IT MATTERS (3-4 sentences): what changes for a normal person, one concrete example.
4. THE TWIST (2 sentences): the catch, risk or detail nobody mentions.
5. CLOSING LINE (1 sentence): a sharp, memorable last thought.
Style: spoken English, contractions, "I" and "you", 6-14 word sentences with a few very short ones.
No filler, no emojis, no hashtags, no markdown, no stage directions, no labels.
Do NOT write any call to action - one is appended separately.

Return one JSON object:
"script": the body as one plain-text paragraph (no line breaks, no double quotes inside),
"title": curiosity-driven, under 60 characters, not clickbait-false,
"hook_banner": 3-4 punchy UPPERCASE words for a thumbnail banner,
"description": 2 sentences for the video description,
"pinned_comment": 1-2 sentences that make people want the tools in the bio link,
"tags": 6 relevant hashtags,
"scenes": exactly {n} image prompts for an AI image model, story order, 10-16 words each:
          subject + setting + lighting, technology only (devices, chips, robots, code, data centers),
          no faces, no brands, no logos, no readable text.
{note}"""

    def generate(self, topic, hook):
        lo, hi = self.cfg.script_words
        mid, slack = (lo + hi) / 2, 15
        best, note = None, ""
        for attempt in range(1, self.cfg.script_attempts + 1):
            data = self.llm.chat_json(SYSTEM_PROMPT, self._prompt(topic, hook, note))
            if not isinstance(data, dict):
                note = "\nYour last reply was not valid JSON. Reply with the JSON object only."
                continue
            body = strip_trailing_cta(clean_for_speech(data.get("script")))
            n = len(body.split())
            if best is None or abs(n - mid) < abs(len(best[0].split()) - mid):
                best = (body, data)
            if lo - slack <= n <= hi + slack:
                info(f"script: {n} words + bio CTA (attempt {attempt}, {self.llm.last})")
                return self._pack(topic, body, data, hook)
            warn(f"script body had {n} words, want {lo}-{hi} (attempt {attempt})")
            note = f"\nYour previous script had {n} words. It MUST be {lo}-{hi} words."
        if best and 0.5 * lo <= len(best[0].split()) <= 1.5 * hi:
            warn("using closest script outside the target word range")
            return self._pack(topic, best[0], best[1], hook)
        raise RuntimeError("cloud LLM produced no usable script")

    def _pack(self, topic, body, data, hook):
        title = re.sub(r"\s+", " ", str(data.get("title") or topic.title)).strip()
        if len(title) > 60:
            title = title[:57].rstrip() + "..."
        scenes = [clean_scene(s) for s in (data.get("scenes") or []) if str(s).strip()]
        scenes = [s for s in scenes if s]
        banner = re.sub(r"[^A-Za-z0-9 ]", "", str(data.get("hook_banner") or title)).upper().strip()
        banner = " ".join(banner.split()[:4]) or " ".join(title.upper().split()[:4])
        spoken = f"{body} {clean_for_speech(self.cfg.bio_cta)}".strip()
        s = Script(spoken, body, title, normalize_tags(data.get("tags")), scenes, banner)
        s.description = str(data.get("description") or body.split(".")[0] + ".").strip()
        s.pinned = str(data.get("pinned_comment")
                       or "Everything I use to test this stuff is in the first link in my bio.").strip()
        return s


# =====================================================================
# 4. VOICE: Kokoro-82M (local neural TTS) -> faster-whisper word timestamps
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


def spread(tokens, start, end):
    if not tokens:
        return []
    w = [len(t) + 2 + (6 if t[-1] in ".,!?;:" else 0) for t in tokens]
    tot, t, out = float(sum(w)), start, []
    for tok, wi in zip(tokens, w):
        d = (end - start) * wi / tot
        out.append(Word(tok, t, t + d)); t += d
    return out


class VoiceEngine:
    """Kokoro -> wav, then faster-whisper word alignment. Both unloaded from VRAM afterwards."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _kokoro(self, text, wav_path):
        from kokoro import KPipeline
        import soundfile as sf
        pipe = KPipeline(lang_code="a")
        chunks = []
        for item in pipe(text, voice=self.cfg.kokoro_voice, speed=self.cfg.kokoro_speed):
            a = item[-1] if isinstance(item, (tuple, list)) else getattr(item, "audio", None)
            if a is None:
                continue
            try:
                a = a.detach().cpu().numpy()
            except Exception:
                a = np.asarray(a)
            chunks.append(np.asarray(a, dtype=np.float32).reshape(-1))
        del pipe
        gc.collect(); free_gpu()
        if not chunks:
            raise RuntimeError("Kokoro returned no audio")
        audio = np.concatenate(chunks)
        audio = audio / (np.abs(audio).max() or 1.0) * 0.92
        sf.write(str(wav_path), audio, 24000)
        return audio, 24000

    def _align(self, wav_path, text, duration):
        try:
            from faster_whisper import WhisperModel
            dev = "cuda" if has_cuda() else "cpu"
            ct = "float16" if dev == "cuda" else "int8"
            model = WhisperModel(self.cfg.whisper_model, device=dev, compute_type=ct)
            segs, _ = model.transcribe(str(wav_path), word_timestamps=True, vad_filter=False,
                                       beam_size=1, language="en")
            words = []
            for s in segs:
                for w in (s.words or []):
                    t = w.word.strip()
                    if t:
                        words.append(Word(t, float(w.start), float(w.end)))
            del model
            gc.collect(); free_gpu()
            if words:
                ok(f"word alignment: {len(words)} words (faster-whisper {self.cfg.whisper_model}, {dev})")
                return words
        except Exception as e:
            warn(f"faster-whisper alignment failed ({type(e).__name__}: {e}) - estimating timings")
        gc.collect(); free_gpu()
        return spread(text.split(), 0.1, max(duration - 0.1, 0.6))

    def synthesize(self, text, tmp):
        wav = Path(tmp) / "voice.wav"
        audio, sr = self._kokoro(text, wav)
        words = self._align(wav, text, len(audio) / sr)
        pcm = decode_audio(wav)
        dur = len(pcm) / SR
        ok(f"voice: Kokoro {self.cfg.kokoro_voice}, {dur:.1f}s")
        return Voiceover(pcm, words, dur, self.cfg.kokoro_voice)


# =====================================================================
# 5. STUDIO AUDIO: synth BGM + SFX + sidechain duck  (loudnorm applied by ffmpeg)
# =====================================================================
def _hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def _add(out, start, sig, gain=1.0):
    i = int(start * SR)
    if i >= len(out) or i < 0:
        return
    j = min(len(out), i + len(sig))
    out[i:j] += gain * sig[: j - i].astype(np.float32)


def sfx_whoosh(np_rng, dur=0.55):
    t = np.arange(int(dur * SR)) / SR
    n = np_rng.standard_normal(len(t)).astype(np.float32)
    k = np.hanning(600).astype(np.float32); k /= k.sum()
    n = np.convolve(n, k, "same")
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    return (n * env * 0.8).astype(np.float32)


def sfx_riser(np_rng, dur=1.2):
    t = np.arange(int(dur * SR)) / SR
    f = 220 * np.exp(2.3 * t / dur)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.5
    s += 0.4 * np.convolve(np_rng.standard_normal(len(t)), np.hanning(300) / 150, "same")
    return (s * (t / dur) ** 2).astype(np.float32)


def sfx_drop(dur=0.7):
    t = np.arange(int(dur * SR)) / SR
    f = 110 * np.exp(-3.5 * t) + 38
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.5)).astype(np.float32)


def synth_bgm(duration, rng, np_rng):
    """Randomized tempo / key / chord progression: pad + bass + kick + hat + arp."""
    total = duration + 2.0
    out = np.zeros(int(total * SR), np.float32)
    beat = 60.0 / rng.choice([96, 104, 112, 120, 128])
    bar = 4 * beat
    minor, major = (0, 3, 7), (0, 4, 7)
    key = rng.choice([0, 2, 3, 5, 7, 8, 10])
    prog = rng.choice([
        [(57, minor), (53, major), (48, major), (55, major)],
        [(52, minor), (48, major), (55, major), (50, major)],
        [(50, minor), (46, major), (53, major), (48, major)],
        [(57, minor), (55, major), (53, major), (52, minor)],
    ])
    prog = [(r + key, iv) for r, iv in prog]
    pad_len = bar + 0.6
    tt = np.arange(int(pad_len * SR)) / SR
    pad_env = np.minimum(1.0, tt / 0.45) * np.minimum(1.0, (pad_len - tt) / 0.6)
    for b in range(int(math.ceil(total / bar)) + 1):
        root, iv = prog[b % 4]
        sig = np.zeros_like(tt)
        for k in iv + (12,):
            f = _hz(root + 12 + k)
            sig += np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 1.004 * tt)
        _add(out, b * bar, 0.16 * pad_env * sig / 4)
    eighth = beat / 2
    nt = np.arange(int(eighth * 0.95 * SR)) / SR
    bass_env = np.minimum(1.0, nt / 0.006) * np.exp(-nt * 4.0)
    at = np.arange(int(0.45 * SR)) / SR
    arp_env = np.minimum(1.0, at / 0.004) * np.exp(-at * 9.0)
    kt = np.arange(int(0.28 * SR)) / SR
    kick = np.sin(2 * np.pi * (42 * kt + 85 * (1 - np.exp(-30 * kt)) / 30)) * np.exp(-kt * 10)
    ht = np.arange(int(0.05 * SR)) / SR
    bass_pat, arp_pat = [1, 0, 1, 1, 0, 1, 1, 0], [0, 2, 1, 2, 0, 2, 1, 3]
    for e in range(int(total / eighth)):
        t, step = e * eighth, e % 8
        root, iv = prog[int(t // bar) % 4]
        if bass_pat[step]:
            f = _hz(root - 12)
            _add(out, t, 0.13 * (np.sin(2 * np.pi * f * nt) + 0.5 * np.sin(2 * np.pi * 2 * f * nt)) * bass_env)
        f = _hz(root + 24 + (iv + (12,))[arp_pat[step] % 4])
        _add(out, t, 0.16 * (np.sin(2 * np.pi * f * at) + 0.3 * np.sin(2 * np.pi * 2 * f * at)) * arp_env)
        if step % 2 == 0:
            _add(out, t, 0.17 * kick)
        else:
            _add(out, t, 0.10 * np.diff(np_rng.standard_normal(len(ht) + 1)).astype(np.float32) * np.exp(-ht * 70))
    return out[: int(duration * SR)]


def _phone_band(x, cutoff=250.0):
    spec = np.fft.rfft(x)
    spec[: int(cutoff * len(x) / SR)] = 0
    return np.fft.irfft(spec, len(x)).astype(np.float32)


def _rms(x):
    x = np.abs(x)
    x = x[x > 0.05 * (x.max() if x.size else 0)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 1e-6


def duck_curve(voice, floor_db):
    block = 441
    nb = len(voice) // block
    if nb == 0:
        return np.ones(len(voice), np.float32)
    env = np.abs(voice[: nb * block]).reshape(nb, block).mean(1)
    target = (env > max(env.max() * 0.08, 1e-4)).astype(np.float64)
    a_att, a_rel = 1 - math.exp(-1 / (0.04 * 100)), 1 - math.exp(-1 / (0.35 * 100))
    s, sm = 0.0, np.empty(nb)
    for i, tg in enumerate(target):
        s += (a_att if tg > s else a_rel) * (tg - s)
        sm[i] = s
    gain = 1 + (10 ** (floor_db / 20) - 1) * sm
    xs = (np.arange(nb) + 0.5) * block
    return np.interp(np.arange(len(voice)), xs, gain).astype(np.float32)


def build_audio(voice_pcm, total_s, cut_times, cfg, rng, np_rng, path):
    """Voice + randomized BGM + transition SFX, sidechain-ducked. loudnorm happens in ffmpeg."""
    n = int(total_s * SR)
    v = np.zeros(n, np.float32)
    off = int(cfg.lead_in * SR)
    seg = voice_pcm[: max(0, n - off)]
    v[off: off + len(seg)] = seg
    v *= 0.16 / _rms(v)
    if np.abs(v).max() > 0.95:
        v *= 0.95 / np.abs(v).max()

    m = np.zeros(n, np.float32)
    bgm = synth_bgm(total_s, rng, np_rng)
    m[: min(n, len(bgm))] = bgm[:n]
    m = m / (np.abs(m).max() or 1.0) * 0.9
    fi, fo = min(int(0.8 * SR), n // 4), min(int(1.4 * SR), n // 3)
    m[:fi] *= np.linspace(0, 1, fi)
    m[-fo:] *= np.linspace(1, 0, fo)
    m *= (_rms(v) * 10 ** (cfg.bgm_db / 20)) / max(_rms(_phone_band(m)), 1e-6)
    m *= duck_curve(v, cfg.duck_db)

    s = np.zeros(n, np.float32)
    _add(s, 0.0, sfx_drop(), 0.9)
    for i, t in enumerate(cut_times[1:], 1):
        if rng.random() < 0.55:
            _add(s, max(0.0, t - 0.18), sfx_whoosh(np_rng), 0.7)
        if i == max(1, len(cut_times) // 2) and rng.random() < 0.7:
            _add(s, max(0.0, t - 1.2), sfx_riser(np_rng), 0.5)
            _add(s, t, sfx_drop(), 0.8)
    if _rms(s) > 1e-5:
        s *= (_rms(v) * 10 ** (cfg.sfx_db / 20)) / _rms(s)

    mix = v + m + s
    peak = np.abs(mix).max()
    if peak > 0.97:
        mix *= 0.97 / peak
    write_wav(path, np.column_stack([mix, mix]))
    return path


# =====================================================================
# 6. VISUALS: Pixabay VIDEO b-roll + Pixabay IMAGE backfill + local SDXL-Lightning
# =====================================================================
PALETTES = ["deep blue and cyan neon glow", "purple and magenta neon glow",
            "teal and warm orange cinematic grade", "icy white and electric blue glow"]
STYLE = ("cinematic tech photography, {palette}, dramatic rim lighting, shallow depth of field, "
         "ultra detailed, sharp focus, 8k, vertical composition")


def procedural_bg(rng, size):
    w, h = size
    c1 = np.array([rng.randint(8, 30), rng.randint(14, 42), rng.randint(40, 95)], np.float32)
    c2 = np.array([rng.randint(0, 18), rng.randint(0, 20), rng.randint(12, 45)], np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    arr = np.repeat(c1 * (1 - t) + c2 * t, w, axis=1)
    arr[::96, :, :] += 18
    arr[:, ::96, :] += 18
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class LocalImageGen:
    """SDXL base + ByteDance SDXL-Lightning 4-step LoRA + fp16-fix VAE on the Colab GPU."""
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
            import peft       # noqa: F401
        except ImportError as e:
            return False, f"python package '{getattr(e, 'name', e)}' missing (run Cell 1)"
        return True, ""

    def load(self):
        import torch
        from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline
        from huggingface_hub import hf_hub_download
        info("loading SDXL-Lightning onto the GPU (first run downloads ~7 GB)...")
        extra = {}
        try:
            extra["vae"] = AutoencoderKL.from_pretrained(self.VAE, torch_dtype=torch.float16)
        except Exception as e:
            warn(f"fp16 VAE unavailable ({type(e).__name__}) - using default")
        pipe = StableDiffusionXLPipeline.from_pretrained(self.BASE, torch_dtype=torch.float16,
                                                         variant="fp16", use_safetensors=True, **extra)
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
            try:
                self.pipe.vae.enable_tiling()
            except Exception:
                self.pipe.enable_vae_tiling()
            return self.pipe(prompt, generator=torch.Generator("cuda").manual_seed(seed), **kw).images[0]

    def release(self):
        self.pipe = None
        free_gpu()


class VisualEngine:
    PIX_VIDEO = "https://pixabay.com/api/videos/"
    PIX_IMAGE = "https://pixabay.com/api/"

    def __init__(self, cfg, rng, pixabay_key):
        self.cfg, self.rng, self.key = cfg, rng, pixabay_key
        self.local = LocalImageGen(cfg)
        self.state_path = Path(cfg.out_dir) / ".pixabay_seen.json"
        self._seen = {"videos": [], "images": []}
        try:
            self._seen.update(json.loads(self.state_path.read_text()))
        except Exception:
            pass
        self._dead_v = self._dead_i = False
        self._warned = False

    # ---- cross-run dedup ----
    def _save_state(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self._seen["videos"] = self._seen["videos"][-800:]
            self._seen["images"] = self._seen["images"][-800:]
            self.state_path.write_text(json.dumps(self._seen))
        except Exception:
            pass

    def plan(self):
        good, why = self.local.usable()
        ok("AI art: local SDXL-Lightning (GPU)") if good else warn(f"local AI art unavailable: {why}")
        if not self.key:
            warn("no PIXABAY_API_KEY - no real b-roll, AI art only")

    def release(self):
        self.local.release()
        self._save_state()

    # ---- Pixabay VIDEO b-roll ----
    def _pixabay_videos(self, queries, n, tmp):
        out = []
        if not self.key or n <= 0 or self._dead_v:
            return out
        for q, _cat in queries:
            if len(out) >= n:
                break
            params = {"key": self.key, "q": q[:100], "video_type": "film", "safesearch": "true",
                      "order": "popular", "per_page": 50}
            try:
                r = http_get(self.PIX_VIDEO, params=params, timeout=(8, 20), retries=1)
                if r.status_code != 200:
                    warn(f"Pixabay videos HTTP {r.status_code} - skipping b-roll")
                    self._dead_v = True
                    return out
                hits = r.json().get("hits", [])
            except Exception as e:
                warn(f"Pixabay videos unreachable ({type(e).__name__})")
                self._dead_v = True
                return out
            self.rng.shuffle(hits)
            taken = 0
            for h in hits:
                if h["id"] in self._seen["videos"]:
                    continue
                vids = h.get("videos") or {}
                url = (vids.get("large") or {}).get("url") or (vids.get("medium") or {}).get("url")
                if not url:
                    continue
                dst = Path(tmp) / f"broll_{h['id']}.mp4"
                try:
                    rr = http_get(url, timeout=(10, 60), retries=1)
                    if rr.status_code != 200 or len(rr.content) < 60000:
                        continue
                    dst.write_bytes(rr.content)
                except Exception:
                    continue
                self._seen["videos"].append(h["id"])
                out.append({"kind": "video", "path": str(dst)})
                taken += 1
                if len(out) >= n or taken >= 2:
                    break
        info(f"real b-roll clips: {len(out)}")
        return out

    # ---- Pixabay IMAGE backfill (locked to computer/science) ----
    def _pixabay_images(self, queries, n, tmp):
        out = []
        if not self.key or n <= 0 or self._dead_i:
            return out
        for q, cat in queries:
            if len(out) >= n:
                break
            cat = cat if cat in self.cfg.pixabay_categories else "computer"
            params = {"key": self.key, "q": q[:100], "image_type": "photo", "orientation": "vertical",
                      "safesearch": "true", "order": "popular", "category": cat, "per_page": 60}
            try:
                r = http_get(self.PIX_IMAGE, params=params, timeout=(8, 18), retries=1)
                if r.status_code != 200:
                    warn(f"Pixabay images HTTP {r.status_code} - skipping photos")
                    self._dead_i = True
                    return out
                hits = r.json().get("hits", [])
            except Exception as e:
                warn(f"Pixabay images unreachable ({type(e).__name__})")
                self._dead_i = True
                return out
            self.rng.shuffle(hits)
            taken = 0
            for h in hits:
                if h["id"] in self._seen["images"]:
                    continue
                try:
                    rr = http_get(h.get("largeImageURL"), timeout=(8, 30), retries=1)
                    if rr.status_code != 200 or len(rr.content) < 20000:
                        continue
                    img = Image.open(BytesIO(rr.content)).convert("RGB")
                except Exception:
                    continue
                if min(img.size) < 480:
                    continue
                dst = Path(tmp) / f"photo_{h['id']}.jpg"
                cover(img, (self.cfg.width, self.cfg.height)).save(dst, quality=92)
                self._seen["images"].append(h["id"])
                out.append({"kind": "image", "path": str(dst)})
                taken += 1
                if len(out) >= n or taken >= 2:
                    break
        return out

    # ---- AI art ----
    def _prompts(self, topic, scenes, n):
        hint = topic.hint()
        pool = [s for s in scenes if s][:n]
        for extra in topic.theme_scenes() + GENERIC_SCENES:
            if len(pool) >= n:
                break
            if extra not in pool:
                pool.append(extra)
        while len(pool) < n and pool:
            pool.append(pool[len(pool) % len(pool)])
        style = STYLE.format(palette=self.rng.choice(PALETTES))
        return [(p if hint.lower() in p.lower() else f"{p}, {hint}") + ", " + style for p in pool[:n]]

    def _ai_images(self, prompts, tmp):
        out = []
        if self.local.failed or not prompts:
            return out
        good, why = self.local.usable()
        if not good:
            if not self._warned:
                warn(f"local AI art unavailable: {why}")
                self._warned = True
            self.local.failed = True
            return out
        if self.local.pipe is None:
            try:
                self.local.load()
            except Exception as e:
                warn(f"SDXL load failed ({type(e).__name__}: {e})")
                self.local.failed = True
                self.local.release()
                return out
        t0, fails = time.time(), 0
        for i, p in enumerate(prompts, 1):
            try:
                img = self.local.generate(p, self.rng.randint(1, 999999))
                dst = Path(tmp) / f"ai_{i}_{self.rng.randint(1000,9999)}.jpg"
                cover(img, (self.cfg.width, self.cfg.height)).save(dst, quality=93)
                out.append({"kind": "image", "path": str(dst)})
                fails = 0
            except Exception as e:
                fails += 1
                warn(f"AI image {i} failed ({type(e).__name__}: {e})")
                if fails >= 2:
                    break
        info(f"AI art: {len(out)}/{len(prompts)} images in {time.time() - t0:.0f}s")
        return out

    @staticmethod
    def _interleave(videos, ais):
        """Mix b-roll and AI art; never more than 2 AI images back to back."""
        out, run = [], 0
        v, a = list(videos), list(ais)
        while v or a:
            if a and run < 2 and (not v or len(a) >= len(v)):
                out.append(a.pop(0)); run += 1
            elif v:
                out.append(v.pop(0)); run = 0
            elif a:
                out.append(a.pop(0)); run += 1
        return out

    def collect(self, topic, scenes, tmp):
        cfg = self.cfg
        themes = ", ".join(t[0] for t in topic.themes()) or "generic tech"
        info(f"visual theme: {themes}")
        queries = topic.queries()
        n_v = self.rng.randint(*cfg.real_clips)
        n_a = self.rng.randint(*cfg.ai_images)
        videos = self._pixabay_videos(queries, n_v, tmp)
        ais = self._ai_images(self._prompts(topic, scenes, n_a), tmp)
        need = max(0, (n_v + n_a) - len(videos) - len(ais))
        if need:
            ais += self._pixabay_images(queries, need, tmp)
        if len(videos) + len(ais) < 3:
            miss = 3 - len(videos) - len(ais)
            warn(f"not enough visuals - generating {miss} procedural background(s)")
            for i in range(miss):
                dst = Path(tmp) / f"bg_{i}.jpg"
                procedural_bg(self.rng, (cfg.width, cfg.height)).save(dst, quality=90)
                ais.append({"kind": "image", "path": str(dst)})
        seq = self._interleave(videos, ais)
        info(f"visual pool: {len(videos)} b-roll clips + {len(ais)} stills")
        self._save_state()
        return seq


# =====================================================================
# 7. COMPOSITOR: segments -> concat -> karaoke .ass + bar + grain + watermark + loudnorm
# =====================================================================
def _ts(t):
    t = max(0.0, float(t))
    h = int(t // 3600); m = int((t % 3600) // 60); s = t - 3600 * h - 60 * m
    return "%d:%02d:%05.2f" % (h, m, s)


def plan_cuts(words, total_s, rng, lo, hi):
    """Variable, sentence-synced cut boundaries covering the whole timeline."""
    cuts, start = [], 0.0
    for w in words:
        d = w.end - start
        end_punct = w.text.rstrip()[-1:] in ".!?,;:"
        if d >= hi or (d >= lo and end_punct) or (d >= lo + rng.random() * (hi - lo) and d >= lo):
            cuts.append((start, min(w.end, total_s)))
            start = min(w.end, total_s)
    if total_s - start > 0.45:
        cuts.append((start, total_s))
    elif cuts:
        cuts[-1] = (cuts[-1][0], total_s)
    cuts = [(a, b) for a, b in cuts if b - a >= 0.45]
    if not cuts:
        n = max(2, int(total_s / 2.4))
        step = total_s / n
        cuts = [(i * step, (i + 1) * step) for i in range(n)]
    return cuts


class Compositor:
    """ASS karaoke colours: PrimaryColour (sung) = neon yellow #FFF200 -> &H0000F2FF,
       SecondaryColour (not yet sung) = white &H00FFFFFF, black outline."""
    ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Pop,DejaVu Sans,{size},&H0000F2FF,&H00FFFFFF,&H00000000,&H78000000,-1,0,0,0,100,100,1,0,1,7,3,2,70,70,560,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def __init__(self, cfg):
        self.cfg = cfg

    # ---- captions ----
    def build_ass(self, words, path):
        cfg = self.cfg
        groups = [words[i:i + cfg.caption_words] for i in range(0, len(words), cfg.caption_words)]
        lines = [self.ASS_HEADER.format(size=cfg.caption_size)]
        for gi, g in enumerate(groups):
            st = max(0.0, g[0].start - 0.03)
            en = g[-1].end + 0.10
            if gi + 1 < len(groups):
                en = min(en, groups[gi + 1][0].start + 0.02)
            if en <= st:
                en = st + 0.18
            txt = r"{\fad(70,60)}"
            for w in g:
                cs = max(4, int(round((w.end - w.start) * 100)))
                t = w.text.strip().replace("{", "").replace("}", "").replace("\n", " ")
                txt += r"{\k%d}%s " % (cs, t.upper() if cfg.caption_uppercase else t)
            lines.append("Dialogue: 0,%s,%s,Pop,,0,0,0,,%s" % (_ts(st), _ts(en), txt.strip()))
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        return path

    # ---- per-segment normalisation ----
    def _video_clip(self, src, dur, dst):
        vf = (f"scale={self.cfg.width}:{self.cfg.height}:force_original_aspect_ratio=increase,"
              f"crop={self.cfg.width}:{self.cfg.height},fps={self.cfg.fps},setsar=1,format=yuv420p")
        ff(["-stream_loop", "3", "-i", src, "-t", f"{dur:.3f}", "-an", "-vf", vf,
            "-c:v", "libx264", "-preset", self.cfg.encode_preset, "-crf", self.cfg.video_crf,
            "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", dst])

    def _image_clip(self, src, dur, idx, dst, rng):
        cfg = self.cfg
        n = max(2, int(round(dur * cfg.fps)))
        zmax = 1.0 + cfg.zoom
        rate = cfg.zoom / n
        zin = (idx % 2 == 0)
        z = (f"min(1+{rate:.6f}*on,{zmax:.3f})" if zin else f"max({zmax:.3f}-{rate:.6f}*on,1)")
        dx = rng.choice([-1, 0, 1]) * 90
        dy = rng.choice([-1, 0, 1]) * 110
        x = f"(iw-iw/zoom)/2+({dx})*(on/{n})"
        y = f"(ih-ih/zoom)/2+({dy})*(on/{n})"
        vf = (f"scale={cfg.width*2}:{cfg.height*2}:force_original_aspect_ratio=increase,"
              f"crop={cfg.width*2}:{cfg.height*2},fps={cfg.fps},"
              f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={cfg.width}x{cfg.height}:fps={cfg.fps},"
              f"setsar=1,format=yuv420p")
        ff(["-loop", "1", "-i", src, "-t", f"{dur:.3f}", "-an", "-vf", vf,
            "-c:v", "libx264", "-preset", self.cfg.encode_preset, "-crf", self.cfg.video_crf,
            "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", dst])

    def render(self, segments, cuts, words, audio_wav, out_path, tmp, rng):
        cfg = self.cfg
        total_s = cuts[-1][1]
        work = Path(tmp) / f"seg_{rng.randint(1000,9999)}"
        work.mkdir(parents=True, exist_ok=True)
        info(f"rendering {len(cuts)} cuts / {total_s:.1f}s from {len(segments)} visuals...")
        parts = []
        for i, (a, b) in enumerate(cuts):
            d = max(0.5, round((b - a) * cfg.fps) / cfg.fps)
            seg = segments[i % len(segments)]
            dst = work / f"c{i:03d}.mp4"
            try:
                if seg["kind"] == "video":
                    self._video_clip(seg["path"], d, str(dst))
                else:
                    self._image_clip(seg["path"], d, i, str(dst), rng)
            except Exception as e:
                warn(f"segment {i} failed ({e}) - substituting still")
                alt = next((s for s in segments if s["kind"] == "image"), None)
                if not alt:
                    raise
                self._image_clip(alt["path"], d, i, str(dst), rng)
            parts.append(dst.name)
            if i and i % 5 == 0:
                info(f"  segments {i}/{len(cuts)}")

        (work / "list.txt").write_text("".join(f"file '{p}'\n" for p in parts))
        ff(["-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "concat.mp4"], cwd=str(work))

        ass = self.build_ass(words, work / "captions.ass")
        font = FONT_PATH or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        brand = "@" + cfg.brand
        vf = (f"noise=alls=6:allf=t+u,"
              f"vignette=PI/5,"
              f"subtitles=captions.ass,"
              f"drawbox=x=0:y=0:w='iw*t/{total_s:.3f}':h=8:color=0xFFF200@0.92:t=fill,"
              f"drawtext=fontfile='{font}':text='{brand}':x=34:y=30:fontsize=38:"
              f"fontcolor=white@0.42:shadowcolor=black@0.55:shadowx=2:shadowy=2")
        maxrate = int(max(1_200_000, min(6_500_000, cfg.max_file_mb * 8e6 / max(total_s, 1) - 250_000)))
        ff(["-i", "concat.mp4", "-i", str(Path(audio_wav).resolve()),
            "-map", "0:v", "-map", "1:a",
            "-vf", vf,
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-c:v", "libx264", "-preset", cfg.encode_preset, "-crf", cfg.video_crf,
            "-maxrate", maxrate, "-bufsize", 2 * maxrate,
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-shortest",
            str(Path(out_path).resolve())], cwd=str(work))
        shutil.rmtree(work, ignore_errors=True)
        return out_path


# =====================================================================
# 8. THUMBNAIL + POST PACKAGE
# =====================================================================
def make_thumbnail(video_path, out_jpg, banner_text, cfg, at=None):
    """1080x1920 thumbnail from the strongest hook frame + bold banner + @brand badge."""
    frame = Path(str(out_jpg) + ".src.png")
    at = 1.4 if at is None else at
    try:
        ff(["-ss", f"{at:.2f}", "-i", str(video_path), "-frames:v", "1", str(frame)])
        img = Image.open(frame).convert("RGB")
    except Exception:
        img = Image.new("RGB", (cfg.width, cfg.height), (14, 18, 34))
    finally:
        frame.unlink(missing_ok=True)
    img = cover(img, (cfg.width, cfg.height))
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=70, threshold=2))

    W, H = img.size
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rectangle([0, int(H * 0.60), W, H], fill=(0, 0, 0, 120))

    def font(sz):
        try:
            return ImageFont.truetype(FONT_PATH, sz) if FONT_PATH else ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    words = (banner_text or "BIG TECH SHIFT").upper().split()[:4]
    lines, cur = [], []
    f = font(128)
    for w in words:
        if cur and d.textlength(" ".join(cur + [w]), font=f) > W - 150:
            lines.append(" ".join(cur)); cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    lh = 150
    y = int(H * 0.66)
    for ln in lines:
        tw = d.textlength(ln, font=f)
        x = (W - tw) / 2
        d.rounded_rectangle([x - 34, y - 18, x + tw + 34, y + lh - 36], radius=22, fill=(0, 0, 0, 205))
        d.text((x, y - 10), ln, font=f, fill=(255, 242, 0, 255), stroke_width=6, stroke_fill=(0, 0, 0, 255))
        y += lh

    fb = font(46)
    badge = "@" + cfg.brand
    bw = d.textlength(badge, font=fb)
    d.rounded_rectangle([34, 30, 34 + bw + 44, 30 + 74], radius=16, fill=(0, 0, 0, 170))
    d.text((56, 44), badge, font=fb, fill=(255, 255, 255, 235))

    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    out.save(out_jpg, quality=92)
    return str(out_jpg)


def post_package(script, topic, cfg, path):
    tags = script.tags
    desc = getattr(script, "description", script.body.split(".")[0] + ".")
    pinned = getattr(script, "pinned", "Tools I use are in the first link in my bio.")
    txt = f"""=====================================================
 {cfg.brand.upper()} — COPY & PASTE POST PACKAGE
 {datetime.now():%Y-%m-%d %H:%M}
=====================================================

[ TITLE ]
{script.title}

[ DESCRIPTION ]
{desc}

Everything I use to test, build and secure this stuff is in the first link in my bio.

Source topic: {topic.title}
{tags} #OPsblueprint

[ PINNED COMMENT ]
{pinned}
First link in my bio has the exact stack.

[ FTC / AFFILIATE DISCLOSURE ]
Disclosure: {cfg.affiliate_note} If you buy through them I may earn a small commission at no extra
cost to you. This video is for information and education only and is not financial, legal or
security advice. Opinions are my own.

[ TAGS ]
{tags} #OPsblueprint #TechNews #Shorts

[ SPOKEN SCRIPT ]
{script.text}
=====================================================
"""
    Path(path).write_text(txt, encoding="utf-8")
    return str(path)


# =====================================================================
# 9. TELEGRAM (video + photo + document)
# =====================================================================
class Telegram:
    def __init__(self, token, chat, base="https://api.telegram.org"):
        self.token, self.chat, self.base = token, chat, base

    def _red(self, s):
        return str(s).replace(self.token, "***")

    def check(self):
        try:
            j = http_get(f"{self.base}/bot{self.token}/getMe", timeout=(8, 15), retries=1).json()
            return bool(j.get("ok")), (j.get("result") or {}).get("username", "")
        except Exception as e:
            return False, self._red(e)

    def _post(self, method, data, files, timeout=(15, 900)):
        err = None
        for a in range(3):
            try:
                opened = {k: (Path(v).name, open(v, "rb"), m) for k, (v, m) in files.items()}
                try:
                    r = SESSION.post(f"{self.base}/bot{self.token}/{method}", data=data,
                                     files=opened, timeout=timeout)
                finally:
                    for _, fh, _ in opened.values():
                        fh.close()
                j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                if j.get("ok"):
                    return j
                err = f"HTTP {r.status_code}: {j.get('description', r.text[:140])}"
                if r.status_code == 429:
                    time.sleep(int((j.get("parameters") or {}).get("retry_after", 5)) + 1)
                    continue
                break
            except requests.RequestException as e:
                err = self._red(e)
                time.sleep(3 * (a + 1))
        raise RuntimeError(f"Telegram {method} failed - {err}")

    def send_video(self, path, caption, dur, w, h):
        return self._post("sendVideo",
                          {"chat_id": self.chat, "caption": caption, "parse_mode": "HTML",
                           "supports_streaming": "true", "duration": int(round(dur)),
                           "width": w, "height": h},
                          {"video": (path, "video/mp4")})

    def send_photo(self, path, caption):
        return self._post("sendPhoto",
                          {"chat_id": self.chat, "caption": caption, "parse_mode": "HTML"},
                          {"photo": (path, "image/jpeg")}, timeout=(15, 300))

    def send_document(self, path, caption):
        return self._post("sendDocument",
                          {"chat_id": self.chat, "caption": caption, "parse_mode": "HTML"},
                          {"document": (path, "text/plain")}, timeout=(15, 300))


# =====================================================================
# 10. PIPELINE  (sequential GPU usage: Kokoro -> Whisper -> SDXL -> FFmpeg)
# =====================================================================
@dataclass
class Job:
    idx: int
    topic: Topic
    script: Script = None
    voice: Voiceover = None
    segments: list = field(default_factory=list)
    cuts: list = field(default_factory=list)
    total_s: float = 0.0
    path: str = None
    thumb: str = None
    pkg: str = None
    sent: bool = False
    error: str = ""


def run_stage(jobs, label, fn):
    print(f"\n==================== {label} ====================")
    for j in jobs:
        if j.error:
            continue
        print(f"--- short {j.idx}/{len(jobs)}: {j.topic.title[:80]}")
        try:
            fn(j)
        except Exception as e:
            j.error = f"{label.split(': ')[-1]}: {type(e).__name__}: {e}"
            warn(f"short {j.idx} dropped - {j.error}")
            traceback.print_exc()


def preflight(cfg, eng, tg):
    if not FFMPEG:
        raise RuntimeError("ffmpeg not found - run Cell 1")
    for mod in ("kokoro", "faster_whisper"):
        try:
            __import__(mod)
        except ImportError:
            warn(f"python package '{mod}' missing - run Cell 1")
    if not eng["script"].setup():
        raise RuntimeError("no cloud script brain (set GEMINI_API_KEY or GITHUB_TOKEN)")
    eng["visual"].plan()
    if tg:
        good, who = tg.check()
        (ok if good else warn)(f"Telegram: {'@' + who if good else who}")
    elif cfg.send_to_telegram:
        warn("TELEGRAM_BOT_TOKEN / TELEGRAM_CHANNEL missing - files stay in " + cfg.out_dir)


def run_pipeline(cfg=None):
    cfg = cfg or CFG
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    token, chat = get_secret("TELEGRAM_BOT_TOKEN"), get_secret("TELEGRAM_CHANNEL")
    tg = Telegram(token, chat) if (cfg.send_to_telegram and token and chat) else None
    eng = {"script": ScriptEngine(cfg), "voice": VoiceEngine(cfg), "comp": Compositor(cfg),
           "visual": VisualEngine(cfg, rng, get_secret("PIXABAY_API_KEY"))}
    preflight(cfg, eng, tg)

    info(f"collecting {cfg.num_shorts} topics ({cfg.topic_source})...")
    jobs = [Job(i, t) for i, t in enumerate(TopicEngine(cfg).get_topics(cfg.num_shorts), 1)]

    with tempfile.TemporaryDirectory(prefix="opsb_") as tmp:
        def do_script(j):
            j.script = eng["script"].generate(j.topic, HOOK_STYLES[(j.idx - 1) % len(HOOK_STYLES)])

        def do_voice(j):
            j.voice = eng["voice"].synthesize(j.script.text, tmp)
            j.total_s = round((cfg.lead_in + j.voice.duration + cfg.tail) * cfg.fps) / cfg.fps
            j.voice.words = [Word(w.text, w.start + cfg.lead_in, w.end + cfg.lead_in) for w in j.voice.words]
            j.cuts = plan_cuts(j.voice.words, j.total_s, rng, *cfg.cut_range)

        def do_visuals(j):
            j.segments = eng["visual"].collect(j.topic, j.script.scenes, tmp)

        def do_render(j):
            t0 = time.time()
            wav = Path(tmp) / f"mix_{j.idx}.wav"
            build_audio(j.voice.pcm, j.total_s, [c[0] for c in j.cuts], cfg, rng, np_rng, wav)
            stem = f"short_{datetime.now():%Y%m%d_%H%M%S}_{j.idx}"
            out = Path(cfg.out_dir) / f"{stem}.mp4"
            eng["comp"].render(j.segments, j.cuts, j.voice.words, wav, out, tmp, rng)
            j.path = str(out)
            j.thumb = make_thumbnail(out, Path(cfg.out_dir) / f"{stem}_thumb.jpg",
                                     j.script.hook, cfg, at=min(1.6, j.total_s * 0.12))
            j.pkg = post_package(j.script, j.topic, cfg, Path(cfg.out_dir) / f"{stem}_post_package.txt")
            ok(f"{out.name} ({out.stat().st_size / 1048576:.1f} MB) in {time.time() - t0:.0f}s")
            if tg:
                cap = (f"<b>{html.escape(j.script.title)}</b>\n\n{html.escape(j.script.tags)}"
                       f"\n\n@{cfg.brand} — tools in the first link in bio")
                try:
                    tg.send_video(out, cap, j.total_s, cfg.width, cfg.height)
                    j.sent = True
                    ok("sent video")
                except Exception as e:
                    warn(f"{e} - kept at {out}")
                for fn, p, c in ((tg.send_photo, j.thumb, f"Thumbnail — @{cfg.brand}"),
                                 (tg.send_document, j.pkg, f"Post package — @{cfg.brand}")):
                    try:
                        fn(p, c); ok(f"sent {Path(p).name}")
                    except Exception as e:
                        warn(e)
                if j.sent and cfg.delete_after_send:
                    Path(out).unlink(missing_ok=True)

        run_stage(jobs, "STAGE 1/4: cloud scripts", do_script)
        run_stage(jobs, "STAGE 2/4: Kokoro voice + Whisper alignment", do_voice)
        run_stage(jobs, "STAGE 3/4: hybrid visuals (b-roll + SDXL)", do_visuals)
        eng["visual"].release()      # free VRAM before FFmpeg render
        run_stage(jobs, "STAGE 4/4: render, thumbnail, package, post", do_render)

    print("\n==================== SUMMARY ====================")
    res = []
    for j in jobs:
        state = "sent" if j.sent else ("saved " + str(j.path) if j.path else "FAILED - " + j.error)
        print(f"{j.idx}. {j.topic.title[:58]}  ->  {state}")
        res.append({"topic": j.topic.title, "title": j.script.title if j.script else None,
                    "path": j.path, "thumb": j.thumb, "package": j.pkg, "sent": j.sent, "error": j.error})
    return res


# =====================================================================
# 11. SELFTEST — 5s video + thumbnail + package, no GPU / network / keys
# =====================================================================
def selftest(cfg=None):
    cfg = cfg or CFG
    print("\n==================== SELFTEST (offline) ====================")
    if not FFMPEG:
        warn("ffmpeg missing - run Cell 1"); return None
    rng = random.Random(7)
    np_rng = np.random.default_rng(7)
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    text = ("This is a five second offline selftest of the OPsblueprint shorts factory. "
            "If you want the exact AI and security tools I actually run, they are at the first link in my bio.")
    with tempfile.TemporaryDirectory(prefix="opsb_test_") as tmp:
        segs = []
        for i in range(3):
            p = Path(tmp) / f"bg{i}.jpg"
            procedural_bg(rng, (cfg.width, cfg.height)).save(p, quality=90)
            segs.append({"kind": "image", "path": str(p)})
        total = 5.0
        words = spread(text.split()[:18], 0.3, total - 0.6)
        cuts = [(0.0, 1.7), (1.7, 3.4), (3.4, total)]
        tone = (0.05 * np.sin(2 * np.pi * 180 * np.arange(int(4.0 * SR)) / SR)).astype(np.float32)
        wav = Path(tmp) / "a.wav"
        build_audio(tone, total, [c[0] for c in cuts], cfg, rng, np_rng, wav)
        stem = f"selftest_{datetime.now():%Y%m%d_%H%M%S}"
        mp4 = out_dir / f"{stem}.mp4"
        Compositor(cfg).render(segs, cuts, words, wav, mp4, tmp, rng)
        thumb = make_thumbnail(mp4, out_dir / f"{stem}_thumb.jpg", "SELFTEST PASSED", cfg, at=1.0)
        s = Script(text, text, "Selftest Short", normalize_tags(["#Test"]), [], "SELFTEST PASSED")
        s.description = "Offline render selftest."
        s.pinned = "Tools in the first link in my bio."
        pkg = post_package(s, Topic("Selftest", "offline"), cfg, out_dir / f"{stem}_post_package.txt")
    ok(f"video:     {mp4} ({mp4.stat().st_size/1048576:.2f} MB)")
    ok(f"thumbnail: {thumb}")
    ok(f"package:   {pkg}")
    print("SELFTEST OK")
    return {"video": str(mp4), "thumb": thumb, "package": pkg}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tech Shorts Factory v5 — @OPsblueprint")
    ap.add_argument("--selftest", action="store_true", help="offline 5s render test (no GPU/network/keys)")
    ap.add_argument("--shorts", type=int, default=None, help="override number of shorts")
    a = ap.parse_args(sys.argv[1:] if len(sys.argv) > 1 else [])
    if a.shorts:
        CFG.num_shorts = a.shorts
    selftest(CFG) if a.selftest else run_pipeline(CFG)
```

**Quick notes**
- `Script` gains `description`/`pinned` attributes at runtime (set in `_pack`), used by `post_package`.
- Endpoints used are the real production hosts implied by your spec: `generativelanguage.googleapis.com` (Gemini, `responseMimeType: application/json`, temp 1.0, model-downgrade on HTTP 429) and `models.inference.ai.azure.com` (GitHub Models, `gpt-4o-mini`, `response_format=json_object`).
- GPU order is strictly sequential: Kokoro → unload → faster-whisper → unload → SDXL-Lightning → unload → FFmpeg, with `render_workers=1`.
