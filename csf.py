# =============================================================================
import os, sys, re, json, gc, math, random, time, shutil, subprocess, difflib, wave, argparse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from itertools import zip_longest
from pathlib import Path
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, SR = 1080, 1920, 44100
KEYS = ("GEMINI_API_KEY", "GROQ_API_KEY", "GITHUB_TOKEN", "PIXABAY_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHANNEL")
_SEC = {}


def secret(name):
    if name not in _SEC:
        v = ""
        try:
            from google.colab import userdata
            v = userdata.get(name) or ""
        except Exception:
            pass
        _SEC[name] = (v or os.environ.get(name, "")).strip()
    return _SEC[name]


def _scrub(s):
    s = str(s)
    for k in KEYS:
        v = secret(k)
        if len(v) >= 6:
            s = s.replace(v, f"<{k}>")
    s = re.sub(r"(key=)[A-Za-z0-9_\-]{12,}", r"\1<redacted>", s)
    return re.sub(r"(bot)\d+:[A-Za-z0-9_\-]{20,}", r"\1<redacted>", s)


def log(*a):
    print(_scrub(" ".join(str(x) for x in a)), flush=True)


S = requests.Session()
S.headers["User-Agent"] = "OPsblueprint-ShortsFactory/5.2"
RETRY_CODES = (429, 500, 502, 503, 504)


def http_get(url, tries=4, **kw):
    """GET with exponential backoff on 429/5xx and network errors. Returns the last response (caller checks it)."""
    kw.setdefault("timeout", 20)
    host = url.split("/")[2] if "//" in url else url
    for a in range(tries):
        wait = min(30.0, 2.0 ** a + random.random())
        try:
            r = S.get(url, **kw)
            if r.status_code not in RETRY_CODES or a == tries - 1:
                return r
            ra = r.headers.get("Retry-After", "")
            if ra.isdigit():
                wait = min(30.0, float(ra))
            r.close()
            log(f"http {r.status_code} from {host}; retry {a + 1}/{tries - 1} in {wait:.1f}s")
        except requests.RequestException as e:
            if a == tries - 1:
                raise
            log(f"http error from {host}: {type(e).__name__}; retry {a + 1}/{tries - 1} in {wait:.1f}s")
        time.sleep(wait)


# ------------------------------------------------------------------ config
@dataclass
class Config:
    brand: str = "OPsblueprint"
    out_dir: str = "/content/shorts_out"
    num_shorts: int = 3
    render_workers: int = 1
    send_to_telegram: bool = True
    delete_after_send: bool = True
    gemini_models: tuple = ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite")
    groq_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_models: tuple = ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b")
    github_models: tuple = (("https://models.inference.ai.azure.com/chat/completions", "gpt-4o-mini"),
                            ("https://models.github.ai/inference/chat/completions", "openai/gpt-4o-mini"))
    script_words: tuple = (125, 145)          # TOTAL spoken words incl. bio nudge
    scenes_per_script: int = 8
    script_attempts: int = 3
    ctas: tuple = (
        "If you want the exact AI and security tools I actually run, they are at the first link in my bio.",
        "The gear and software I actually trust are listed at the first link in my bio.",
        "If you want the tools I use to stay ahead of this stuff, check the first link in my bio.")
    kokoro_voices: tuple = ("am_michael", "am_adam", "af_heart")
    voice_speed: float = 1.05
    n_clips: tuple = (3, 5)
    n_ai: tuple = (4, 6)
    clip_max_s: float = 6.5
    local_image_size: tuple = (768, 1344)
    cut_range: tuple = (1.5, 3.2)
    zoom: float = 0.10
    fps: int = 30
    caption_words: int = 3
    caption_size: int = 78
    bgm_db: float = -10
    duck_db: float = -5
    lead_in: float = 0.4
    tail: float = 1.0
    video_crf: int = 20
    encode_preset: str = "superfast"
    max_file_mb: int = 45
    disclosure: str = ("Disclosure: links in my bio may be affiliate links, so I may earn a commission at no extra "
                       "cost to you. Voice and imagery in this video are AI-generated.")


CFG = Config()

STYLE = ("editorial macro hardware photography, dark obsidian and brushed titanium aesthetic, subtle cobalt and "
         "amber rim lighting, shallow depth of field, crisp studio lighting, vertical 9:16 composition")
PALETTES = ("cobalt blue rim light with a warm amber accent", "amber key light against cool cobalt shadows",
            "brushed titanium highlights with soft cobalt glow", "deep obsidian background with thin amber edge light",
            "cool cobalt backlight and faint amber reflections on metal")
NEG = ("cartoon, 3d render, clip art, illustration, watermark, text, letters, numbers, UI gibberish, blurry, "
       "low resolution, human faces, stock photo cliche")

DEFAULT_Q = ("server room", "data center", "fiber optic", "circuit board", "microchip")
DEFAULT_SCENES = ("macro shot of a server rack with cobalt status lights in a dark room",
                  "close-up of a brushed titanium heatsink on a circuit board",
                  "long corridor of data center racks with amber edge lighting",
                  "bundle of fiber optic cables glowing cobalt against obsidian",
                  "macro of a microchip die on a dark lab bench",
                  "overhead view of a motherboard with thin amber traces")
TECH_THEMES = {
    "ai": {"kw": ("ai", "gpt", "llm", "openai", "model", "neural", "gemini", "chatbot", "agent", "claude"),
           "q": ("artificial intelligence", "neural network", "microchip", "circuit board"),
           "s": ("macro of a glowing AI accelerator module on a dark bench",
                 "rows of GPU cards in a dark rack with cobalt LEDs",
                 "close-up of a fiber-linked motherboard with amber traces",
                 "titanium heatsink on an accelerator card with cobalt reflections")},
    "security": {"kw": ("security", "breach", "vulnerability", "malware", "ransomware", "encryption", "leak", "exploit",
                        "privacy", "attack", "zero-day", "cyber"),
                 "q": ("server room", "data center", "encryption", "server rack"),
                 "s": ("steel server cage door closing in a dark data hall",
                       "hardware security key on brushed titanium with cobalt rim light",
                       "network switch with amber link lights in a locked rack",
                       "macro of a secure chip on a circuit board under cobalt light")},
    "mobile": {"kw": ("iphone", "android", "phone", "app", "pixel", "samsung", "apple", "social", "meta", "tiktok"),
               "q": ("smartphone screen", "fiber optic", "mobile technology", "circuit board"),
               "s": ("macro of a smartphone camera module on dark obsidian",
                     "disassembled phone mainboard with amber rim light",
                     "fiber optic strands fanning out with cobalt glow",
                     "titanium phone frame edge in studio light")},
    "cloud": {"kw": ("cloud", "aws", "azure", "datacenter", "server", "outage", "kubernetes", "network"),
              "q": ("data center", "server rack", "fiber optic", "server room"),
              "s": ("endless aisle of server racks with cobalt indicator lights",
                    "patch panel with fiber cables and amber link lights",
                    "cooling fans inside a data center rack, shallow depth of field",
                    "overhead cable trays in a dark data hall")},
    "chip": {"kw": ("chip", "gpu", "cpu", "nvidia", "intel", "amd", "processor", "semiconductor", "tsmc", "quantum"),
             "q": ("microchip", "processor", "motherboard", "circuit board"),
             "s": ("macro of a CPU die with cobalt reflections",
                   "silicon wafer on a dark stage with amber edge light",
                   "GPU board with titanium shroud in studio lighting",
                   "gold pins of a processor socket in extreme close-up")},
    "space": {"kw": ("space", "nasa", "satellite", "rocket", "spacex", "starlink", "science", "robot"),
              "q": ("satellite", "science laboratory", "rocket launch", "fiber optic"),
              "s": ("satellite antenna dish at dusk with cobalt sky",
                    "robotic arm joint macro in brushed titanium",
                    "lab bench with precision instruments under amber light",
                    "rocket engine nozzle close-up in dark studio light")},
}
DEFAULT_THEME = {"kw": (), "q": DEFAULT_Q, "s": DEFAULT_SCENES}


def pick_theme(text):
    toks = set(re.findall(r"[a-z0-9\-]+", text.lower()))
    best, bs = DEFAULT_THEME, 0
    for t in TECH_THEMES.values():
        sc = sum(1 for k in t["kw"] if k in toks or (len(k) >= 5 and any(k in w for w in toks)))
        if sc > bs:
            best, bs = t, sc
    return best, bs


def theme_queries(text):
    return pick_theme(text)[0]["q"]


def pad_scenes(scenes, text, need):
    """Drop empty scenes, then pad with theme-specific hardware scenes so a valid script is never discarded."""
    out = [" ".join(str(s).split()) for s in scenes if str(s).strip()]
    pool = list(pick_theme(text)[0]["s"]) + list(DEFAULT_SCENES)
    for s in pool:
        if len(out) >= need:
            break
        if s not in out:
            out.append(s)
    k = 0
    while len(out) < need:
        out.append(pool[k % len(pool)] + ", alternate angle"); k += 1
    return out


EVERGREEN = [
    {"title": "How AI agents plan, use tools and fail", "q": "Intelligent agent"},
    {"title": "Quantum computing and the future of encryption", "q": "Post-quantum cryptography"},
    {"title": "Why humanoid robots struggle to walk and grip", "q": "Humanoid robot"},
    {"title": "Why data centers are straining the power grid", "q": "Data center"},
    {"title": "Zero-day vulnerabilities and the market for exploits", "q": "Zero-day vulnerability"},
    {"title": "Edge AI chips that run models without the cloud", "q": "Neural processing unit"},
    {"title": "Deepfake detection and authentication of real media", "q": "Deepfake"},
    {"title": "Open-source LLMs versus closed models", "q": "Large language model"},
    {"title": "How a GPU actually trains a neural network", "q": "Graphics processing unit"},
    {"title": "Passkeys and the slow death of passwords", "q": "Passkey"},
    {"title": "Undersea cables that carry the internet", "q": "Submarine communications cable"},
    {"title": "How chip fabs print transistors at nanometer scale", "q": "Semiconductor device fabrication"},
    {"title": "Confidential computing and hardware security enclaves", "q": "Trusted execution environment"},
]


# ------------------------------------------------------------------ helpers
class Seen:
    def __init__(self, p):
        self.p = Path(p)
        self.p.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.s = set(json.loads(self.p.read_text()))
        except Exception:
            self.s = set()

    def __contains__(self, k):
        return k in self.s

    def add(self, k):
        self.s.add(k)
        tmp = self.p.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.s)))
        tmp.replace(self.p)


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode:
        raise RuntimeError(_scrub(f"{cmd[0]} failed: {p.stderr[-400:]}"))
    return p


def ff(*a):
    return run(["ffmpeg", "-y", "-loglevel", "error", *map(str, a)])


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def wav_write(path, x, sr):
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def wav_read(path):
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        d = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return d.reshape(-1, ch).mean(1) if ch > 1 else d


def smooth(x, k):
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0))
    y = (c[k:] - c[:-k]) / k
    return np.pad(y, (k // 2, len(x) - len(y) - k // 2)).astype(np.float32)


def cover(im, w, h):
    im = im.convert("RGB")
    s = max(w / im.width, h / im.height)
    im = im.resize((math.ceil(im.width * s), math.ceil(im.height * s)), Image.LANCZOS)
    l, t = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((l, t, l + w, t + h))


def probe_dur(p):
    try:
        return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]).stdout)
    except Exception:
        return 0.0


def fesc(p):
    return str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


_FONTS = {}


def font(size, bold=True):
    k = (size, bold)
    if k not in _FONTS:
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "DejaVuSans-Bold.ttf"):
            try:
                _FONTS[k] = ImageFont.truetype(p, size); break
            except Exception:
                pass
        else:
            _FONTS[k] = ImageFont.load_default()
    return _FONTS[k]


def parse_json_loose(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    i = t.find("{")
    if i < 0:
        raise ValueError("no JSON object in LLM reply")
    obj, _ = json.JSONDecoder().raw_decode(t[i:])
    return obj


# ------------------------------------------------------------------ cloud LLM
class CloudLLM:
    def __init__(self, cfg):
        self.cfg = cfg

    def _gemini(self, prompt):
        err = None
        for m in self.cfg.gemini_models:
            try:
                r = S.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                           headers={"x-goog-api-key": secret("GEMINI_API_KEY")},
                           json={"contents": [{"parts": [{"text": prompt}]}],
                                 "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0}},
                           timeout=90)
                if r.status_code in (401, 403):
                    raise PermissionError(f"gemini auth {r.status_code}")
                if r.status_code != 200:
                    log(f"gemini {m}: HTTP {r.status_code} -> next model"); err = r.status_code; continue
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            except PermissionError:
                raise
            except Exception as e:
                log(f"gemini {m}: {e}"); err = e
        raise RuntimeError(f"all Gemini models failed ({err})")

    def _openai(self, label, url, key, models, prompt):
        err = None
        for m in models:
            try:
                r = S.post(url, headers={"Authorization": f"Bearer {key}"},
                           json={"model": m, "messages": [{"role": "user", "content": prompt}],
                                 "response_format": {"type": "json_object"}, "temperature": 1.0}, timeout=90)
                if r.status_code in (401, 403):
                    raise PermissionError(f"{label} auth {r.status_code}")
                if r.status_code != 200:
                    log(f"{label} {m}: HTTP {r.status_code} {r.text[:100]} -> next"); err = r.status_code; continue
                return r.json()["choices"][0]["message"]["content"]
            except PermissionError:
                raise
            except Exception as e:
                log(f"{label} {m}: {e}"); err = e
        raise RuntimeError(f"all {label} models failed ({err})")

    def _groq(self, p):
        return self._openai("groq", self.cfg.groq_url, secret("GROQ_API_KEY"), self.cfg.groq_models, p)

    def _github(self, p):
        err = None
        for url, m in self.cfg.github_models:
            try:
                return self._openai("github", url, secret("GITHUB_TOKEN"), (m,), p)
            except Exception as e:
                err = e
        raise RuntimeError(f"GitHub Models failed ({err})")

    def available(self):
        return [n for n, k in (("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY"), ("github", "GITHUB_TOKEN")) if secret(k)]

    def json(self, prompt):
        fns = {"gemini": self._gemini, "groq": self._groq, "github": self._github}
        for name in self.available():
            try:
                return parse_json_loose(fns[name](prompt))
            except Exception as e:
                log(f"[{name}] failed: {e}")
        raise RuntimeError("No LLM produced valid JSON (check GEMINI_API_KEY / GROQ_API_KEY / GITHUB_TOKEN)")


# ------------------------------------------------------------------ topics + scripts
class TopicEngine:
    RSS = "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en"
    TREND = "https://trends.google.com/trending/rss?geo=US"

    def __init__(self, cfg):
        self.cfg = cfg
        self.seen = Seen(Path(cfg.out_dir) / ".topics_seen.json")

    @staticmethod
    def norm(t):
        t = re.sub(r"\s+[-|\u2013]\s+[^-|\u2013]{2,40}$", "", t)
        return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()

    def news(self):
        try:
            r = http_get(self.RSS)
            r.raise_for_status()
            out = []
            for it in ET.fromstring(r.content).iter("item"):
                t = (it.findtext("title") or "").strip()
                d = re.sub(r"<[^>]+>", " ", it.findtext("description") or "")
                if t:
                    out.append({"title": t, "summary": re.sub(r"\s+", " ", d).strip()[:300]})
            return out
        except Exception as e:
            log("rss", e); return []

    def hn(self):
        try:
            ids = http_get("https://hacker-news.firebaseio.com/v0/topstories.json").json()[:30]

            def one(i):
                try:
                    return http_get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json", tries=3, timeout=15).json()
                except Exception:
                    return None
            with ThreadPoolExecutor(8) as ex:
                items = list(ex.map(one, ids))
            return [{"title": i["title"], "summary": f"Hacker News score {i.get('score', 0)}"}
                    for i in items if i and i.get("type") == "story" and i.get("title")]
        except Exception as e:
            log("hn", e); return []

    def trending(self):
        """Google Daily Trending RSS, filtered to tech-related items (trends are mostly sports/celebs otherwise)."""
        try:
            r = http_get(self.TREND)
            r.raise_for_status()
            out = []
            for it in ET.fromstring(r.content).iter("item"):
                term = (it.findtext("title") or "").strip()
                news = [(e.text or "").strip() for e in it.iter() if e.tag.endswith("news_item_title") and (e.text or "").strip()]
                if term and pick_theme(term + " " + " ".join(news))[1] > 0:
                    out.append({"title": news[0] if news else term,
                                "summary": f"Trending search: {term}. " + " ".join(news[1:3])[:250]})
            return out
        except Exception as e:
            log("trending", e); return []

    def wiki(self, q):
        try:
            pages = http_get("https://en.wikipedia.org/w/rest.php/v1/search/title", params={"q": q, "limit": 1}, timeout=15).json().get("pages") or []
            if not pages:
                return ""
            s = http_get("https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(pages[0]["key"]), timeout=15).json()
            return s.get("extract", "")[:500]
        except Exception:
            return ""

    def candidates(self):
        need = self.cfg.num_shorts
        out, batch = [], set()

        def take(items, limit=30):
            for t in items:
                n = self.norm(t["title"])
                if n and n not in batch and n not in self.seen:
                    batch.add(n); out.append(t)
                    if len(out) >= limit:
                        break
        take([x for pair in zip_longest(self.news(), self.hn()) for x in pair if x])     # tier 1
        log(f"topics tier 1 (news + HN, unseen): {len(out)}")
        if len(out) < need:                                                                # tier 2
            take(self.trending())
            log(f"topics tier 2 (Google trending): {len(out)} total")
        ever = [dict(e, summary="", evergreen=True) for e in EVERGREEN]                    # tier 3
        random.shuffle(ever)
        ever.sort(key=lambda e: self.norm(e["title"]) in self.seen)                        # unseen first
        spare = max(0, need - len(out)) + 2                                                # fill slots + 2 spares
        used = 0
        for e in ever:
            if used >= spare:
                break
            n = self.norm(e["title"])
            if n not in batch:
                batch.add(n); out.append(e); used += 1
        log(f"topics tier 3 (evergreen): +{used}")
        return out


@dataclass
class Script:
    text: str
    body: str
    title: str
    tags: list
    scenes: list
    description: str
    pinned: str
    hook: str


SAFETY = ("No breaking news facts were provided for this topic\u2014explain the core technology, real-world architecture, "
          "and practical trade-offs clearly without inventing recent dates, fake statistics, unverified company "
          "announcements, or fabricated quotes.")


class ScriptEngine:
    def __init__(self, cfg):
        self.cfg, self.llm, self.topics = cfg, CloudLLM(cfg), TopicEngine(cfg)

    def setup(self):
        if not self.llm.available():
            raise RuntimeError("Set GEMINI_API_KEY, GROQ_API_KEY or GITHUB_TOKEN (no local LLM exists in v5).")

    def prompt(self, t, lo, hi, fb=""):
        c = self.cfg
        thin = bool(t.get("evergreen")) or len((t.get("summary") or "").strip()) < 60
        ctx = "(none provided)" if t.get("evergreen") else t.get("summary", "")
        return (f"You are the lead investigative tech creator behind the channel {c.brand}. Write a YouTube Shorts script "
                "in a sharp, first-person, investigative voice (I dug into this, here is what I found). "
                "Use ONLY the facts supplied below; never invent numbers, quotes or companies.\n\n"
                f"{'TOPIC' if t.get('evergreen') else 'HEADLINE'}: {t['title']}\nCONTEXT: {ctx}\n"
                f"BACKGROUND (Wikipedia): {t.get('wiki', '')}\n\n"
                f"RULES:\n- body must be {lo}-{hi} words, spoken sentences only, no emojis, no hashtags, no stage directions.\n"
                "- First sentence is a hook that creates a curiosity gap. Include one 'twist' around 60% of the way.\n"
                "- Do NOT mention any bio or link in body (it is appended automatically).\n"
                f"- scenes: exactly {c.scenes_per_script} short visual descriptions of hardware objects or environments "
                "(chips, servers, cables, devices, lab benches). No people, no text, no logos.\n"
                "- hook: 3 or 4 word bold thumbnail banner.\n"
                "- tags: 8 hashtags. pinned: a pinned comment that asks a question.\n"
                + (f"- {SAFETY}\n" if thin else "")
                + f"{fb}\nReturn ONLY a JSON object with keys: title, hook, body, scenes, description, tags, pinned.")

    def make(self, t):
        c = self.cfg
        cta = random.choice(c.ctas)
        lo, hi = c.script_words[0] - len(cta.split()), c.script_words[1] - len(cta.split())
        fb = ""
        for a in range(c.script_attempts):
            d = self.llm.json(self.prompt(t, lo, hi, fb))
            body = " ".join(str(d.get("body", "")).split())
            sents = re.split(r"(?<=[.!?])\s+", body)
            while len(sents) > 1 and len(" ".join(sents).split()) > hi:
                sents.pop()
            body = " ".join(sents)
            n = len(body.split())
            if lo <= n <= hi:
                raw = d.get("scenes") if isinstance(d.get("scenes"), list) else []
                scenes = pad_scenes(raw, f"{t['title']} {body}", max(c.n_ai))
                if len([x for x in raw if str(x).strip()]) < max(c.n_ai):
                    log(f"scenes padded: LLM gave {len([x for x in raw if str(x).strip()])}, now {len(scenes)}")
                title = str(d.get("title", t["title"]))[:95]
                rt = d.get("tags") if isinstance(d.get("tags"), list) else []
                tags = [("#" + re.sub(r"\W", "", str(x))) for x in rt if re.sub(r"\W", "", str(x))][:10]
                hook = " ".join(str(d.get("hook", "")).split()[:4]) or " ".join(title.split()[:4])
                return Script(f"{body} {cta}", body, title, tags + [f"#{c.brand}"], scenes,
                              str(d.get("description", title)), str(d.get("pinned", "What would you want me to investigate next?")), hook)
            fb = f"PREVIOUS ATTEMPT FAILED: body had {n} words. Required body {lo}-{hi} words."
            log(f"script attempt {a + 1}: {n} words (need {lo}-{hi})")
        raise RuntimeError("script validation failed")

    def make_all(self):
        self.setup()
        out = []
        for t in self.topics.candidates():
            if len(out) >= self.cfg.num_shorts:
                break
            try:
                t["wiki"] = self.topics.wiki(t.get("q", t["title"]))
                s = self.make(t)
                self.topics.seen.add(self.topics.norm(t["title"]))
                out.append((t, s)); log(f"script ok: {s.title}" + (" [evergreen]" if t.get("evergreen") else ""))
            except Exception as e:
                log("topic skipped:", t["title"][:60], "|", e)
        if not out:
            raise RuntimeError("No scripts generated")
        return out


# ------------------------------------------------------------------ job
@dataclass
class Job:
    idx: int
    topic: dict
    script: Script
    stem: str
    work: Path
    seed: int
    wav: Path = None
    dur: float = 0.0
    words: list = field(default_factory=list)
    assets: list = field(default_factory=list)
    path: Path = None
    thumb: Path = None
    pkg: Path = None
    ok: bool = True


# ------------------------------------------------------------------ voice (Kokoro + Whisper)
def fill(words, st, en, t0, t1):
    n, i = len(words), 0
    while i < n:
        if st[i] is not None:
            i += 1; continue
        j = i
        while j < n and st[j] is None:
            j += 1
        a = en[i - 1] if i > 0 else t0
        b = st[j] if j < n else t1
        wt = [len(w) + 1 for w in words[i:j]]
        tot, c = sum(wt), a
        for k, w in zip(range(i, j), wt):
            d = max(0.05, b - a) * w / tot
            st[k], en[k] = c, c + d
            c += d
        i = j
    out, prev = [], t0
    for w, s, e in zip(words, st, en):
        s = max(s, prev); e = max(e, s + 0.05)
        out.append((w, s, e)); prev = e
    return out


def align_words(words, ww, t0, t1):
    tok = lambda w: re.sub(r"[^a-z0-9']", "", w.lower())
    a, b = [tok(w) for w in words], [tok(w[0]) for w in ww]
    st, en = [None] * len(words), [None] * len(words)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            st[blk.a + k], en[blk.a + k] = ww[blk.b + k][1], ww[blk.b + k][2]
    return fill(words, st, en, t0, t1)


class VoiceEngine:
    def __init__(self, cfg):
        self.cfg, self.tts, self.stt = cfg, None, None

    def _speak(self, j, voice):
        c = self.cfg
        parts = [np.zeros(int(c.lead_in * 24000), np.float32)]
        got = 0
        for s in re.split(r"(?<=[.!?])\s+", j.script.text):
            for _, _, a in self.tts(s, voice=voice, speed=c.voice_speed):
                x = np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a, np.float32).reshape(-1)
                parts.append(x); got += len(x)
            parts.append(np.zeros(int(0.14 * 24000), np.float32))
        if got == 0:
            raise RuntimeError(f"voice {voice} produced no audio")
        parts.append(np.zeros(int(c.tail * 24000), np.float32))
        return np.concatenate(parts)

    def synth(self, j):
        c = self.cfg
        if self.tts is None:
            from kokoro import KPipeline
            self.tts = KPipeline(lang_code="a")
        n = len(c.kokoro_voices)
        last = None
        for k in range(n):                                   # primary voice first, then the remaining voices
            v = c.kokoro_voices[(j.idx + k) % n]
            try:
                audio = self._speak(j, v)
                break
            except Exception as e:
                last = e; log(f"[{j.stem}] kokoro voice {v} failed: {e}"); free_gpu()
        else:
            raise RuntimeError(f"all Kokoro voices failed ({last})")
        raw, j.wav = j.work / "voice24.wav", j.work / "voice.wav"
        wav_write(raw, audio, 24000)
        ff("-i", raw, "-ar", SR, "-ac", 1, j.wav)
        j.dur = len(wav_read(j.wav)) / SR

    def release_tts(self):
        if self.tts is not None:
            del self.tts
        self.tts = None
        free_gpu()

    def align(self, j):
        words = j.script.text.split()
        t0, t1 = self.cfg.lead_in, j.dur - self.cfg.tail
        try:
            if self.stt is None:
                import torch
                from faster_whisper import WhisperModel
                cuda = torch.cuda.is_available()
                self.stt = WhisperModel("base.en", device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")
            segs, _ = self.stt.transcribe(str(j.wav), word_timestamps=True, language="en", beam_size=1)
            ww = [(w.word, w.start, w.end) for s in segs for w in (s.words or [])]
            j.words = align_words(words, ww, t0, t1)
        except Exception as e:
            log("whisper failed, using weighted spread:", e)
            j.words = fill(words, [None] * len(words), [None] * len(words), t0, t1)

    def release_stt(self):
        if self.stt is not None:
            del self.stt
        self.stt = None
        free_gpu()


# ------------------------------------------------------------------ visuals
PIX_VID, PIX_IMG = "https://pixabay.com/api/videos/", "https://pixabay.com/api/"
ALLOWED_CATS = ("computer", "science")


def procedural_bg(path, seed, size=(768, 1344)):
    """Last-resort visual: dark obsidian/cobalt architectural perspective grid with server-tower silhouettes."""
    r = random.Random(seed)
    w, h = size
    yy = np.linspace(0, 1, h)[:, None, None]
    c0 = np.array([6, 8, 14], np.float64)
    c1 = np.array([r.randint(8, 18), r.randint(24, 42), r.randint(60, 105)], np.float64)
    bg = Image.fromarray((c0 + (c1 - c0) * yy ** 1.6 * np.ones((h, w, 1))).astype(np.uint8))
    ln = Image.new("RGB", (w, h)); d = ImageDraw.Draw(ln)
    hz, vx = int(h * r.uniform(0.38, 0.5)), int(w * r.uniform(0.35, 0.65))
    cob, amb = (40, 110, 255), (255, 170, 40)
    for i in range(1, 15):                                   # floor lines with perspective spacing
        y = hz + (h - hz) * (i / 14) ** 2.2
        d.line([(0, y), (w, y)], fill=amb if i % 5 == 0 else cob, width=2)
    for i in range(-14, 15):                                 # lines converging on the vanishing point
        d.line([(vx, hz), (vx + i * w * 0.22, h)], fill=cob, width=2)
    for _ in range(r.randint(6, 10)):                        # server-tower silhouettes above the horizon
        bw, bh, x = r.randint(40, 110), r.randint(80, max(90, int(hz * 0.9))), r.randint(-20, w)
        d.rectangle([x, hz - bh, x + bw, hz], outline=cob, width=2)
        for _k in range(r.randint(2, 6)):
            y = hz - r.randint(10, bh - 5)
            d.line([(x + 6, y), (x + bw - 6, y)], fill=amb, width=2)
    glow = ln.filter(ImageFilter.GaussianBlur(10))
    ln = ImageChops.add(ln.point(lambda v: int(v * 0.55)), glow)
    ImageChops.add(bg, ln).save(path, quality=93)


class VisualEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seen = Seen(Path(cfg.out_dir) / ".pixabay_seen.json")

    def _dl(self, url, dest):
        with http_get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            if int(r.headers.get("content-length", 0)) > 150e6:
                raise RuntimeError("clip too large")
            with open(dest, "wb") as f:
                for ch in r.iter_content(1 << 20):
                    f.write(ch)

    def _pixabay_videos(self, queries, n, work):
        key, out, c = secret("PIXABAY_API_KEY"), [], self.cfg
        if not key:
            return out
        qs = list(queries); random.shuffle(qs)
        for q in qs * 2:
            if len(out) >= n:
                break
            try:
                hits = http_get(PIX_VID, params={"key": key, "q": q, "per_page": 40, "safesearch": "true"}, timeout=30).json().get("hits", [])
            except Exception as e:
                log("pixabay videos:", e); continue
            cands = []
            for h in hits:
                if f"v{h['id']}" in self.seen:
                    continue
                for size in ("medium", "large"):
                    d = (h.get("videos") or {}).get(size) or {}
                    if d.get("url") and (d.get("width", 0) >= 1080 or d.get("height", 0) >= 1080):   # skip low-res
                        cands.append((d.get("height", 0) <= d.get("width", 0), random.random(), h["id"], d["url"]))  # vertical first
                        break
            cands.sort()
            taken = 0
            for _, _, vid, url in cands:
                if len(out) >= n or taken >= 2:
                    break
                src, dst = work / f"v{vid}_src.mp4", work / f"v{vid}.mp4"
                try:
                    self._dl(url, src)
                    ff("-t", c.clip_max_s, "-i", src, "-an", "-vf",
                       f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={c.fps},setsar=1",
                       "-c:v", "libx264", "-preset", "veryfast", "-crf", 18, "-pix_fmt", "yuv420p", dst)
                    self.seen.add(f"v{vid}")
                    out.append({"kind": "clip", "path": str(dst), "dur": probe_dur(dst)}); taken += 1
                except Exception as e:
                    log("clip skipped:", e)
                finally:
                    src.unlink(missing_ok=True)
        return out

    def _pixabay_images(self, queries, n, work):
        key, out = secret("PIXABAY_API_KEY"), []
        if not key or n <= 0:
            return out
        for a in range(8):
            if len(out) >= n:
                break
            cat = ALLOWED_CATS[a % 2]
            assert cat in ALLOWED_CATS
            try:
                hits = http_get(PIX_IMG, params={"key": key, "q": random.choice(queries), "category": cat, "image_type": "photo",
                                                 "orientation": "vertical", "per_page": 40, "safesearch": "true"}, timeout=30).json().get("hits", [])
            except Exception as e:
                log("pixabay images:", e); continue
            random.shuffle(hits)
            for h in hits:
                if len(out) >= n:
                    break
                if f"i{h['id']}" in self.seen or not h.get("largeImageURL"):
                    continue
                dst = work / f"i{h['id']}.jpg"
                try:
                    self._dl(h["largeImageURL"], dst)
                    self.seen.add(f"i{h['id']}")
                    out.append({"kind": "img", "path": str(dst), "dur": 0})
                except Exception as e:
                    log("image skipped:", e)
        return out


class LocalImageGen:
    def __init__(self, cfg):
        self.cfg, self.pipe = cfg, None

    def load(self):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA GPU")
            from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler
            from huggingface_hub import hf_hub_download
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            p = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
                                                          torch_dtype=torch.float16, variant="fp16")
            p.load_lora_weights(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors"))
            p.fuse_lora()
            p.scheduler = EulerDiscreteScheduler.from_config(p.scheduler.config, timestep_spacing="trailing")
            p.to("cuda"); p.enable_vae_tiling(); p.set_progress_bar_config(disable=True)
            self.pipe = p
        except Exception as e:
            log("SDXL unavailable, Pixabay images / procedural art will backfill:", e)
            self.pipe = None

    def generate(self, scene, seed, dest):
        if self.pipe is None:
            return False
        import torch
        try:
            pal = random.Random(seed).choice(PALETTES)
            with torch.inference_mode():
                im = self.pipe(prompt=f"{scene}, {pal}, {STYLE}", negative_prompt=NEG, num_inference_steps=4,
                               guidance_scale=0.0, width=self.cfg.local_image_size[0], height=self.cfg.local_image_size[1],
                               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            im.save(dest, quality=95)
            return True
        except Exception as e:
            log("sdxl image failed:", e); free_gpu(); return False

    def release(self):
        if self.pipe is not None:
            del self.pipe
        self.pipe = None
        free_gpu()


def build_assets(j, cfg, vis, gen):
    rng = random.Random(j.seed)
    q = theme_queries(j.topic["title"] + " " + " ".join(j.script.scenes))
    n_c, n_a = rng.randint(*cfg.n_clips), rng.randint(*cfg.n_ai)
    try:
        clips = vis._pixabay_videos(q, n_c, j.work)
    except Exception as e:
        log(f"[{j.stem}] pixabay videos failed: {e}"); clips = []
    sc = j.script.scenes or pad_scenes([], j.topic["title"], n_a)
    idx = sorted({round(i * (len(sc) - 1) / max(1, n_a - 1)) for i in range(n_a)})
    ai = []
    for k, i in enumerate(idx):
        dst = j.work / f"ai{k}.jpg"
        if gen.generate(sc[i], j.seed + k, dst):
            ai.append({"kind": "ai", "path": str(dst), "dur": 0})
    try:
        imgs = vis._pixabay_images(q, (n_c - len(clips)) + (n_a - len(ai)), j.work)
    except Exception as e:
        log(f"[{j.stem}] pixabay images failed: {e}"); imgs = []
    j.assets = clips + ai + imgs
    nproc = 0
    if len(j.assets) < 3:                                    # last resort: never fail for lack of visuals
        nproc = max(3, n_a) - len(j.assets)
        for k in range(nproc):
            p = j.work / f"proc{k}.jpg"
            procedural_bg(p, j.seed + 900 + k, cfg.local_image_size)
            j.assets.append({"kind": "proc", "path": str(p), "dur": 0})
    log(f"[{j.stem}] assets: {len(clips)} clips, {len(ai)} SDXL, {len(imgs)} pixabay images, {nproc} procedural")


# ------------------------------------------------------------------ audio synthesis
def synth_bgm(dur, rng):
    bpm, root = rng.randint(84, 112), rng.choice([110.0, 116.54, 123.47, 130.81, 138.59, 146.83])
    beat = 60 / bpm; bar = beat * 4
    prog = rng.choice([[0, 8, 3, 10], [0, 5, 8, 7], [0, 3, 8, 10], [0, 10, 8, 7]])
    n = int(dur * SR); out = np.zeros(n, np.float32)
    for i in range(int(dur / bar) + 2):
        s, e = int(i * bar * SR), min(n, int((i + 1) * bar * SR))
        if s >= n:
            break
        r = root * 2 ** (prog[i % 4] / 12)
        tt = np.arange(e - s) / SR; L = (e - s) / SR
        env = np.minimum(1, tt / 0.4) * np.minimum(1, (L - tt) / 0.4)
        seg = sum(np.sin(2 * np.pi * r * 2 * c * tt) + 0.4 * np.sin(2 * np.pi * r * 4 * c * tt) for c in (1, 2 ** (3 / 12), 2 ** (7 / 12))) / 4.2
        seg = seg + 0.8 * np.sin(2 * np.pi * r * tt)
        out[s:e] += (seg * env * 0.45).astype(np.float32)
    kt = np.arange(int(0.18 * SR)) / SR
    kick = (np.sin(2 * np.pi * 55 * kt * (1 + 2 * np.exp(-kt * 25))) * np.exp(-kt * 18) * 0.7).astype(np.float32)
    hat = (np.random.default_rng(rng.randint(0, 10 ** 6)).standard_normal(int(0.04 * SR)) * np.exp(-np.arange(int(0.04 * SR)) / SR * 90) * 0.12).astype(np.float32)
    for b in np.arange(0, dur, beat):
        i = int(b * SR); e = min(n, i + len(kick)); out[i:e] += kick[:e - i]
        i = int((b + beat / 2) * SR); e = min(n, i + len(hat))
        if i < n:
            out[i:e] += hat[:e - i]
    return out / max(1e-6, np.abs(out).max()) * 0.5


def sfx(kind, rng):
    g = np.random.default_rng(rng.randint(0, 10 ** 6))
    if kind == "whoosh":
        x = np.linspace(0, 1, int(0.45 * SR))
        return (smooth(g.standard_normal(len(x)), 12) * 6 * np.sin(np.pi * x) ** 2 * 0.6).astype(np.float32)
    if kind == "riser":
        T = 1.2; x = np.linspace(0, 1, int(T * SR))
        return (np.sin(2 * np.pi * (200 * T * x + 1100 * T * x ** 2)) * x ** 2 * 0.35 + g.standard_normal(len(x)) * x ** 3 * 0.2).astype(np.float32)
    t = np.arange(int(0.8 * SR)) / SR
    return (np.sin(2 * np.pi * 45 * t * (1 + 2.5 * np.exp(-t * 9))) * np.exp(-t * 5) * 0.9
            + g.standard_normal(len(t)) * np.exp(-t * 40) * 0.3).astype(np.float32)


def plan_sfx(cuts, fps, total_f):
    ev, twist = [(0.05, "drop")], next((f for f in cuts[1:-1] if f >= 0.6 * total_f), None)
    for f in cuts[1:-1]:
        if f == twist:
            ev += [(max(0, f / fps - 1.2), "riser"), (f / fps, "drop")]
        else:
            ev.append((f / fps, "whoosh"))
    return ev


def master_audio(j, cuts, cfg, rng, total_s, out):
    n = round(total_s * SR)
    v = np.zeros(n, np.float32); x = wav_read(j.wav)[:n]; v[:len(x)] = x
    env = smooth(np.abs(v), SR // 20)
    act = np.clip(smooth((env > 0.02).astype(np.float32), SR // 7), 0, 1)
    duck = 1 - act * (1 - 10 ** (cfg.duck_db / 20))                      # sidechain-style ducking from voice envelope
    mix = v + synth_bgm(total_s, rng)[:n] * 10 ** (cfg.bgm_db / 20) * duck
    for t, k in plan_sfx(cuts, cfg.fps, round(total_s * cfg.fps)):
        s, i = sfx(k, rng), int(t * SR)
        if i < n:
            e = min(n, i + len(s)); mix[i:e] += s[:e - i] * (0.25 if k == "whoosh" else 0.4)
    wav_write(out, mix, SR)


# ------------------------------------------------------------------ planning
def plan_cuts(words, total_f, cfg, rng):
    """Cuts as exact integer frames: cut_frame = round(cut_time * fps); time = cut_frame / fps."""
    fps = cfg.fps
    min_f, max_f = math.ceil(cfg.cut_range[0] * fps), math.floor(cfg.cut_range[1] * fps)
    ends = [round(e * fps) for (w, s, e) in words if re.search(r"[.!?]$", w)]
    wb = [round(s * fps) for (_, s, _) in words]
    cuts, cur = [0], 0
    while total_f - cur > max_f:
        lo, hi = cur + min_f, min(cur + max_f, total_f - min_f)
        if hi < lo:
            break
        tgt = rng.randint(lo, hi)
        pool = [f for f in ends if lo <= f <= hi] or [f for f in wb if lo <= f <= hi] or [tgt]
        cur = min(pool, key=lambda f: abs(f - tgt)); cuts.append(cur)
    cuts.append(total_f)
    return cuts


def plan_shots(cuts, assets, rng):
    shots, use, hist = [], [0] * len(assets), []
    for a, b in zip(cuts, cuts[1:]):
        c = [i for i in range(len(assets)) if not hist or i != hist[-1]] or [0]
        if not hist:
            c = [i for i in c if assets[i]["kind"] == "ai"] or c           # hook = SDXL art
        run = 0
        for i in reversed(hist):
            if assets[i]["kind"] != "ai":
                break
            run += 1
        if run >= 2:                                                       # never >2 AI images back-to-back
            c = [i for i in c if assets[i]["kind"] != "ai"] or c
        m = min(use[i] for i in c)
        i = rng.choice([k for k in c if use[k] == m])
        shots.append((i, a, b, use[i])); use[i] += 1; hist.append(i)
    return shots


# ------------------------------------------------------------------ captions
def cs_ts(t):
    cs = max(0, int(round(t * 100)))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_ass(words, total_s, cfg, path):
    hdr = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 2\n\n[V4+ Styles]\n"
           "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,"
           "StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
           f"Style: Cap,DejaVu Sans,{cfg.caption_size},&H0000F2FF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,520,1\n"
           "Style: WM,DejaVu Sans,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,1,0,1,2,0,7,34,34,30,1\n\n"
           "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    ev, k = [], cfg.caption_words
    for i in range(0, len(words), k):
        g = words[i:i + k]
        st = g[0][1]
        en = words[i + k][1] if i + k < len(words) else min(total_s, g[-1][2] + 0.25)
        en = max(en, st + 0.1)
        parts, acc = [], 0
        for m, (w, s, e) in enumerate(g):
            nxt = g[m + 1][1] if m + 1 < len(g) else en
            kcs = max(1, int(round((nxt - s) * 100)))
            acc += kcs
            off = acc * 10                                  # ms from line start when this word ends
            # explicit \1c yellow per word stops an earlier word's \t from leaking white into later words;
            # \k lights the word (Primary=yellow), \t flips Primary back to white right after it is spoken.
            parts.append("{\\1c&H0000F2FF&\\k%d\\t(%d,%d,\\1c&H00FFFFFF&)}%s" % (kcs, off, off + 1, re.sub(r"[{}\\]", "", w).upper()))
        ev.append(f"Dialogue: 0,{cs_ts(st)},{cs_ts(en)},Cap,,0,0,0,,{' '.join(parts)}")
    ev.append("Dialogue: 1,0:00:00.00,%s,WM,,0,0,0,,{\\an7\\pos(34,30)\\alpha&H80&}@%s" % (cs_ts(total_s), cfg.brand))
    Path(path).write_text(hdr + "\n".join(ev) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ compositor
_BAR = None


def bar_ok():
    """Probe whether this ffmpeg animates drawbox width with t; otherwise use an overlay bar."""
    global _BAR
    if _BAR is None:
        try:
            p = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x16:r=10:d=1", "-vf",
                                "format=yuv420p,drawbox=x=0:y=0:w='iw*(t/1.000)':h=8:color=white:t=fill,format=gray",
                                "-f", "rawvideo", "-"], capture_output=True)
            fr = [p.stdout[i:i + 1024] for i in range(0, len(p.stdout), 1024)]
            cnt = lambda b: int((np.frombuffer(b, np.uint8) > 128).sum())
            _BAR = p.returncode == 0 and len(fr) >= 10 and cnt(fr[-1]) > cnt(fr[4])
        except Exception:
            _BAR = False
        log("progress bar mode:", "drawbox expression" if _BAR else "overlay fallback")
    return _BAR


class Compositor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = (int(W * 1.12), int(H * 1.12))      # headroom >= 1 + cfg.zoom (0.10)
        self.cache = {}

    def _img(self, a):
        if a["path"] not in self.cache:
            self.cache[a["path"]] = cover(Image.open(a["path"]), *self.base)
        return self.cache[a["path"]]

    def _kb(self, a, n, u):
        im, rng = self._img(a), random.Random(hash(a["path"]) + u)
        Bw, Bh = im.size
        zin, dx, dy = (rng.random() < 0.5) ^ (u % 2 == 1), rng.choice([-0.9, 0.9]), rng.choice([-0.45, 0.45])
        for k in range(n):
            p = k / max(1, n - 1); e = p * p * (3 - 2 * p)
            z = 1 + self.cfg.zoom * (e if zin else 1 - e)
            rw, rh = Bw / z, Bh / z
            ox, oy = (Bw - rw) * (0.5 + dx * (p - 0.5)), (Bh - rh) * (0.5 + dy * (p - 0.5))
            yield im.resize((W, H), Image.BILINEAR, box=(ox, oy, ox + rw, oy + rh)).tobytes()

    def _clip(self, a, n, u):
        fs, ss = W * H * 3, (0 if u == 0 else min(u * 3.2, max(0, a["dur"] - 3.2)))
        p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-stream_loop", "-1", "-ss", f"{ss:.2f}", "-i", a["path"],
                              "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        last = None
        try:
            for _ in range(n):
                b = p.stdout.read(fs)
                if len(b) == fs:
                    last = b
                yield last if last is not None else bytes(fs)
        finally:
            p.kill(); p.stdout.close(); p.wait()

    def render(self, assets, shots, mix, ass, out, total_s):
        c = self.cfg
        fd = "/usr/share/fonts/truetype/dejavu"
        # hard size cap: peak bitrate bounded so the mp4 stays under cfg.max_file_mb (Telegram bot limit is 50 MB)
        maxrate = int(max(1_200_000, min(6_000_000, c.max_file_mb * 8e6 / max(total_s, 1) - 250_000)))
        base = f"[0:v]format=yuv420p,vignette=PI/5,noise=alls=6:allf=t+u,ass={fesc(ass)}" + (f":fontsdir={fesc(fd)}" if os.path.isdir(fd) else "")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(c.fps), "-i", "-", "-i", str(mix)]
        if bar_ok():
            g = base + f",drawbox=x=0:y=0:w='iw*(t/{total_s:.3f})':h=8:color=0xFFF200@0.9:t=fill[v]"
        else:
            cmd += ["-f", "lavfi", "-i", f"color=c=0xFFF200@0.9:s={W}x8:r={c.fps}:d={total_s:.3f}"]
            g = base + f"[b];[2:v]format=rgba[bar];[b][bar]overlay=x='-w+w*t/{total_s:.3f}':y=0:eof_action=repeat[v]"
        cmd += ["-filter_complex", g, "-map", "[v]", "-map", "1:a", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=44100",
                "-c:v", "libx264", "-preset", c.encode_preset, "-crf", str(c.video_crf), "-profile:v", "high",
                "-maxrate", str(maxrate), "-bufsize", str(maxrate * 2), "-pix_fmt", "yuv420p", "-r", str(c.fps),
                "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-t", f"{total_s:.3f}", "-movflags", "+faststart", str(out)]
        errp = Path(str(out) + ".log")
        errf = open(errp, "wb")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
        try:
            for ai, a, b, u in shots:
                gen = self._clip(assets[ai], b - a, u) if assets[ai]["kind"] == "clip" else self._kb(assets[ai], b - a, u)
                for fr in gen:
                    proc.stdin.write(fr)
        except BrokenPipeError:
            pass
        except Exception:
            proc.kill(); raise
        finally:
            try:
                proc.stdin.close()                       # always close stdin first, otherwise wait() can hang forever
            except Exception:
                pass
            rc = proc.wait()
            errf.close(); self.cache.clear(); gc.collect()
        if rc:
            tail = errp.read_text(errors="ignore")[-500:]
            raise RuntimeError(_scrub("ffmpeg render failed: " + tail))
        errp.unlink(missing_ok=True)


# ------------------------------------------------------------------ thumbnail / package / telegram
def make_thumbnail(cfg, src, hook, out):
    im = cover(Image.open(src), W, H).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    for y in range(H // 2, H):
        d.line([(0, y), (W, y)], fill=(0, 0, 0, int(220 * ((y - H // 2) / (H // 2)) ** 1.4)))
    im = Image.alpha_composite(im, ov); d = ImageDraw.Draw(im)
    ws = hook.upper().split()[:4] or [cfg.brand.upper()]
    lines = [" ".join(ws[:2]), " ".join(ws[2:])] if len(ws) > 2 else [" ".join(ws)]
    y = 1130
    d.rectangle([60, y - 30, 300, y - 18], fill=(255, 242, 0, 255))
    for i, ln in enumerate(lines):
        sz = 170
        while d.textlength(ln, font=font(sz)) > 940 and sz > 60:
            sz -= 6
        d.text((60, y), ln, font=font(sz), fill=(255, 242, 0, 255) if i == 0 else (255, 255, 255, 255), stroke_width=8, stroke_fill=(0, 0, 0, 255))
        y += int(sz * 1.2)
    f = font(46); tx = f"@{cfg.brand}"; tw = d.textlength(tx, font=f)
    d.rounded_rectangle([60, 1700, 60 + tw + 60, 1780], radius=40, fill=(10, 12, 18, 235), outline=(255, 242, 0, 255), width=3)
    d.text((90, 1712), tx, font=f, fill=(255, 255, 255, 255))
    im.convert("RGB").save(out, quality=92)


def hook_source(assets, work):
    for kind in ("ai", "img", "proc"):
        for a in assets:
            if a["kind"] == kind:
                return a["path"]
    p = work / "hook.png"
    ff("-i", assets[0]["path"], "-frames:v", 1, p)
    return str(p)


def post_package(cfg, s):
    return "\n".join([f"TITLE:\n{s.title}\n", f"DESCRIPTION:\n{s.description}\n", f"PINNED COMMENT:\n{s.pinned}\n",
                      f"FTC DISCLOSURE:\n{cfg.disclosure}\n", "TAGS:\n" + " ".join(s.tags), ""])


def tg(method, field_name, path, caption="", extra=None):
    tok, chat = secret("TELEGRAM_BOT_TOKEN"), secret("TELEGRAM_CHANNEL")
    if not (tok and chat):
        return False
    try:
        with open(path, "rb") as f:
            r = S.post(f"https://api.telegram.org/bot{tok}/{method}", data={"chat_id": chat, "caption": caption[:1000], **(extra or {})},
                       files={field_name: f}, timeout=300)
        ok = r.ok and r.json().get("ok")
        if not ok:
            log(f"telegram {method} failed: {r.text[:150]}")
        return bool(ok)
    except Exception as e:
        log(f"telegram {method} error: {e}"); return False


def produce(cfg, j, out_dir):
    rng = random.Random(j.seed)
    total_f = round(j.dur * cfg.fps); total_s = total_f / cfg.fps
    cuts = plan_cuts(j.words, total_f, cfg, rng)
    shots = plan_shots(cuts, j.assets, rng)
    mix, ass = j.work / "mix.wav", j.work / "caps.ass"
    master_audio(j, cuts, cfg, rng, total_s, mix)
    write_ass(j.words, total_s, cfg, ass)
    j.path, j.thumb, j.pkg = out_dir / f"{j.stem}.mp4", out_dir / f"{j.stem}_thumb.jpg", out_dir / f"{j.stem}_post_package.txt"
    Compositor(cfg).render(j.assets, shots, mix, ass, j.path, total_s)
    make_thumbnail(cfg, hook_source(j.assets, j.work), j.script.hook, j.thumb)
    j.pkg.write_text(post_package(cfg, j.script), encoding="utf-8")
    mb = j.path.stat().st_size / 1e6
    if mb > cfg.max_file_mb:
        log(f"WARN {j.path.name} is {mb:.1f} MB (> {cfg.max_file_mb})")


def do_render(cfg, j):
    out_dir = Path(cfg.out_dir)
    try:
        produce(cfg, j, out_dir)
        if not cfg.send_to_telegram:
            log(f"DONE {j.stem} (telegram off, files kept) -> {j.path}")
            return
        cap = f"{j.script.title}\n@{cfg.brand}"
        steps = [("sendPhoto", "photo", j.thumb, cap, None),
                 ("sendVideo", "video", j.path, cap, {"supports_streaming": "true", "width": W, "height": H}),
                 ("sendDocument", "document", j.pkg, "Post package", None)]
        for m, f, p, c, ex in steps:
            if Path(p).stat().st_size > 50e6:                # Telegram bot upload limit
                log(f"skip {m}: {Path(p).name} exceeds 50 MB"); continue
            if tg(m, f, p, c, ex) and cfg.delete_after_send:
                Path(p).unlink(missing_ok=True)        # delete each file right after its own upload succeeds
        log(f"DONE {j.stem}")
    except Exception as e:
        log(f"render failed {j.stem}: {e}")
    finally:
        shutil.rmtree(j.work, ignore_errors=True); gc.collect()


# ------------------------------------------------------------------ pipeline
def _stage(jobs, label, fn):
    for j in jobs:
        if j.ok:
            try:
                fn(j)
            except Exception as e:
                j.ok = False; log(f"[{label}] {j.stem} failed: {e}")


def run_pipeline(cfg=CFG):
    out = Path(cfg.out_dir); (out / "work").mkdir(parents=True, exist_ok=True)
    if not secret("PIXABAY_API_KEY"):
        log("WARN: PIXABAY_API_KEY missing - stock clips/backfill disabled (SDXL / procedural only)")
    items = ScriptEngine(cfg).make_all()
    ts = time.strftime("%Y%m%d_%H%M%S")
    jobs = []
    for i, (t, s) in enumerate(items):
        stem = f"short_{ts}_{i + 1}"
        w = out / "work" / stem; w.mkdir(parents=True, exist_ok=True)
        jobs.append(Job(i, t, s, stem, w, random.randint(1, 10 ** 6)))
    voice = VoiceEngine(cfg)
    try:                                                         # 1) Kokoro
        _stage(jobs, "kokoro", voice.synth)
    finally:
        voice.release_tts()
    try:                                                         # 2) Whisper
        _stage(jobs, "whisper", voice.align)
    finally:
        voice.release_stt()
    vis, gen = VisualEngine(cfg), LocalImageGen(cfg)
    try:                                                         # 3) Pixabay + SDXL-Lightning (+ procedural)
        if any(j.ok for j in jobs):
            gen.load()
        _stage(jobs, "visuals", lambda j: build_assets(j, cfg, vis, gen))
    finally:
        gen.release()
    live = [j for j in jobs if j.ok]                             # 4) FFmpeg
    with ThreadPoolExecutor(max(1, cfg.render_workers)) as ex:
        list(ex.map(lambda j: do_render(cfg, j), live))
    log(f"finished: {len(live)}/{len(jobs)} shorts rendered -> {out}")


# ------------------------------------------------------------------ selftest (no GPU / network / keys)
def selftest(cfg=CFG):
    cfg = replace(cfg, out_dir="/tmp/shorts_selftest", delete_after_send=False, send_to_telegram=False)
    out = Path(cfg.out_dir); work = out / "work"
    shutil.rmtree(out, ignore_errors=True); work.mkdir(parents=True)
    text = "This is a quick self test of the OPsblueprint renderer. Captions, cuts and audio should line up."
    words, t, x = [], cfg.lead_in, []
    x.append(np.zeros(int(cfg.lead_in * SR), np.float32))
    for k, w in enumerate(text.split()):
        n = int(0.26 * SR); tt = np.arange(n) / SR
        x.append((np.sin(2 * np.pi * (220 + 25 * k) * tt) * np.minimum(1, tt / 0.02) * np.exp(-tt * 3) * 0.4).astype(np.float32))
        x.append(np.zeros(int(0.04 * SR), np.float32)); words.append((w, t, t + 0.26)); t += 0.30
    sig = np.concatenate(x); sig = np.pad(sig, (0, max(0, 5 * SR - len(sig))))[:5 * SR]
    wav = work / "voice.wav"; wav_write(wav, sig, SR)
    ff("-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:r={cfg.fps}:d=6", "-c:v", "libx264", "-pix_fmt", "yuv420p", work / "clip.mp4")
    assets = [{"kind": "clip", "path": str(work / "clip.mp4"), "dur": 6.0}]
    for k in range(3):                                           # also exercises procedural_bg()
        p = work / f"bg{k}.jpg"; procedural_bg(p, 100 + k, (360, 640))
        assets.append({"kind": "proc", "path": str(p), "dur": 0})
    s = Script(text, text, "Selftest Short", ["#test", f"#{cfg.brand}"], [], "Selftest description.", "Selftest pinned comment?", "SELF TEST PASSED")
    j = Job(0, {"title": "selftest"}, s, "selftest", work, 7, wav=wav, dur=5.0, words=words, assets=assets)
    produce(cfg, j, out)
    log("SELFTEST OK:", j.path, j.thumb, j.pkg)
    return j.path, j.thumb, j.pkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a, _ = ap.parse_known_args()
    selftest(CFG) if a.selftest else run_pipeline(CFG)


if __name__ == "__main__":
    main()
