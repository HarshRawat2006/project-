#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project ZORA - Unified Voice Assistant (Linux-friendly)
------------------------------------------------------
Core features:
- Continuous Speech-to-Text (STT) with automatic language detection
- Sentiment detection (multilingual-capable with fallbacks)
- Natural language interpretation (intent detection)
- Local automation & app control (open/close apps, searches, media)
- Response generation & Text-to-Speech (TTS) with tone and language consistency

Design notes:
- Modular classes with clear responsibilities
- Lazy loading of heavy ML components (transformers, whisper)
- Cross-platform fallbacks, but tuned for Linux
- Minimal logs suitable for dev/prod toggling

Run:
  python project_zora.py

CLI:
  python project_zora.py --help

Environment (optional):
  ZORA_SENTIMENT_MODEL   - HF model (default: cardiffnlp/twitter-roberta-base-sentiment-latest)
  ZORA_WHISPER_MODEL     - faster-whisper model size (default: small)
  ZORA_TTS_ENGINE        - gtts|pyttsx3 (default: gtts)
  ZORA_LANGUAGE_HINT     - force STT language code (e.g. en, hi), overrides detection if set
  OPENAI_API_KEY         - if set and --llm=auto or --llm=openai, uses OpenAI for replies
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import queue
import shlex
import atexit
import signal
import logging
import threading
import tempfile
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Callable

# ---------------------------- Logging ---------------------------------------
LOG_LEVEL = os.getenv("ZORA_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("zora")

# ------------------------ Optional Dependencies ----------------------------
# Speech
try:
    import speech_recognition as sr  # Microphone access and basic STT
    HAS_SR = True
except Exception:
    HAS_SR = False

# Faster Whisper (optional, for robust STT + language detection)
try:
    from faster_whisper import WhisperModel  # type: ignore
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False

# Transformers sentiment (lazy init)
try:
    from transformers import pipeline  # type: ignore
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# Langdetect for textual language detection fallback
try:
    from langdetect import detect as lang_detect  # type: ignore
    HAS_LANGDETECT = True
except Exception:
    HAS_LANGDETECT = False

# TTS options
try:
    from gtts import gTTS  # type: ignore
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

try:
    import pyttsx3  # type: ignore
    HAS_PYTTSX3 = True
except Exception:
    HAS_PYTTSX3 = False

# Optional OpenAI for response generation
try:
    import openai  # type: ignore
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# ----------------------------- Config --------------------------------------
@dataclass
class ZoraConfig:
    sentiment_model: str = os.getenv(
        "ZORA_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    whisper_model: str = os.getenv("ZORA_WHISPER_MODEL", "small")
    tts_engine: str = os.getenv("ZORA_TTS_ENGINE", "gtts").lower()  # gtts|pyttsx3
    language_hint: Optional[str] = os.getenv("ZORA_LANGUAGE_HINT")
    device: str = "cuda" if os.getenv("ZORA_USE_CUDA", "0") == "1" else "cpu"
    # Main loop tuning
    wake_word: Optional[str] = None  # e.g. "zora" to gate STT (None = always listen)
    min_phrase_ms: int = 800  # audio chunk minimum duration
    silence_timeout_s: float = 2.0
    # Actions
    default_browser: Optional[str] = None  # e.g. "google-chrome", "firefox"; None => system default
    # LLM
    llm: str = os.getenv("ZORA_LLM", "none").lower()  # none|openai|auto
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ----------------------------- Utilities -----------------------------------
LANG_CODE_MAP: Dict[str, str] = {
    # langdetect -> gTTS language code mapping (subset)
    "en": "en",
    "hi": "hi",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "ru": "ru",
    "ja": "ja",
    "ko": "ko",
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
}


def normalize_lang_code(code: Optional[str]) -> str:
    if not code:
        return "en"
    c = code.lower()
    # Accept already suitable codes
    if c in LANG_CODE_MAP.values():
        return c
    # Simplify region tags, e.g. en-us -> en
    c = c.split("-")[0]
    return LANG_CODE_MAP.get(c, c or "en")


# ----------------------------- STT -----------------------------------------
class SpeechToText:
    """Microphone listener with optional faster-whisper backend for detection."""

    def __init__(self, cfg: ZoraConfig):
        self.cfg = cfg
        self._whisper_model = None  # lazy
        self._recognizer = sr.Recognizer() if HAS_SR else None
        self._mic = sr.Microphone() if HAS_SR else None
        # Calibrate microphone if available
        if self._recognizer and self._mic:
            try:
                with self._mic as source:
                    self._recognizer.adjust_for_ambient_noise(source, duration=0.8)
                logger.info("Microphone calibrated for ambient noise")
            except Exception as e:
                logger.warning(f"Microphone calibration failed: {e}")

    def _ensure_whisper(self):
        if not HAS_WHISPER:
            return None
        if self._whisper_model is None:
            compute_type = "float16" if self.cfg.device == "cuda" else "int8"
            self._whisper_model = WhisperModel(self.cfg.whisper_model, device=self.cfg.device, compute_type=compute_type)
            logger.info(f"Loaded faster-whisper model: {self.cfg.whisper_model}")
        return self._whisper_model

    def _recognize_with_google(self, audio: "sr.AudioData", lang_hint: Optional[str]) -> Tuple[str, Optional[str]]:
        """Recognize using Google via speech_recognition. Returns (text, lang)."""
        if not HAS_SR or not self._recognizer:
            raise RuntimeError("speech_recognition not available")
        # Google API does not auto-detect; use hint if provided else default en-US
        language = None
        if lang_hint:
            # Map 'en' -> 'en-US' best-effort
            base = lang_hint.split("-")[0]
            region_default = {
                "en": "en-US",
                "hi": "hi-IN",
                "es": "es-ES",
                "fr": "fr-FR",
                "de": "de-DE",
            }.get(base, f"{base}-{base.upper()}")
            language = region_default
        else:
            language = "en-US"
        text = self._recognizer.recognize_google(audio, language=language)
        logger.debug(f"Google STT transcript: {text}")
        # Language detection from text as fallback
        lang = None
        if HAS_LANGDETECT:
            try:
                lang = lang_detect(text)
            except Exception:
                lang = None
        return text, lang

    def _recognize_with_whisper(self, wav_path: str) -> Tuple[str, Optional[str]]:
        model = self._ensure_whisper()
        if model is None:
            raise RuntimeError("faster-whisper not available")
        # language=None enables autodetect
        segments, info = model.transcribe(
            wav_path,
            language=None,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=5,
        )
        # Build text from segments
        transcript_parts: List[str] = []
        for seg in segments:
            if seg.text:
                transcript_parts.append(seg.text.strip())
        text = " ".join(transcript_parts).strip()
        lang = (info.language or None) if hasattr(info, "language") else None
        logger.debug(f"Whisper STT transcript: {text}")
        return text, lang

    def listen_once(self, timeout: Optional[float] = None, phrase_time_limit: Optional[float] = None) -> Tuple[Optional[str], Optional[str]]:
        """Capture a single utterance and return (text, language)."""
        if not HAS_SR or not self._recognizer or not self._mic:
            logger.error("speech_recognition is required for microphone capture")
            return None, None
        with self._mic as source:
            try:
                audio = self._recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except Exception as e:
                logger.debug(f"Listen timeout or error: {e}")
                return None, None
        # Persist temp wav for whisper if needed
        text: Optional[str] = None
        lang: Optional[str] = None
        if HAS_WHISPER:
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
                tmp.write(wav_bytes)
                tmp.flush()
                tmp.close()
                try:
                    text, lang = self._recognize_with_whisper(tmp.name)
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Whisper recognition failed, falling back to Google: {e}")
        if text is None:
            # Fallback to Google
            try:
                text, lang = self._recognize_with_google(audio, self.cfg.language_hint)
            except Exception as e:
                logger.debug(f"Google recognition failed: {e}")
                return None, None
        return text, normalize_lang_code(lang)


# --------------------------- Sentiment --------------------------------------
class SentimentAnalyzer:
    """Multilingual-friendly sentiment analyzer with mapping to positive/neutral/negative."""

    def __init__(self, cfg: ZoraConfig):
        self.cfg = cfg
        self._pipe = None  # lazy

    def _ensure_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        if not HAS_TRANSFORMERS:
            logger.warning("transformers not installed; falling back to simple rule sentiment")
            self._pipe = None
            return None
        model_name = self.cfg.sentiment_model
        try:
            self._pipe = pipeline("text-classification", model=model_name, top_k=None)
            logger.info(f"Loaded sentiment model: {model_name}")
        except Exception as e:
            logger.warning(f"Primary sentiment model failed ({model_name}): {e}")
            try:
                fallback = "nlptown/bert-base-multilingual-uncased-sentiment"
                self._pipe = pipeline("sentiment-analysis", model=fallback)
                logger.info(f"Loaded fallback sentiment model: {fallback}")
            except Exception as e2:
                logger.error(f"No sentiment model available: {e2}")
                self._pipe = None
        return self._pipe

    @staticmethod
    def _map_label(label: str) -> str:
        l = label.lower()
        if "pos" in l or l.endswith("5") or l.endswith("4") or "positive" in l:
            return "positive"
        if "neg" in l or l.endswith("1") or "negative" in l:
            return "negative"
        return "neutral"

    def analyze(self, text: str) -> Tuple[str, float]:
        pipe = self._ensure_pipeline()
        if pipe is None:
            # Rudimentary fallback based on keywords
            pos_keywords = ["great", "good", "love", "awesome", "excellent", "thanks"]
            neg_keywords = ["bad", "terrible", "hate", "awful", "sad", "angry", "worst"]
            score = 0.5
            t = text.lower()
            if any(w in t for w in pos_keywords):
                return "positive", 0.8
            if any(w in t for w in neg_keywords):
                return "negative", 0.8
            return "neutral", score
        try:
            preds = pipe(text)[0]
            # Some models return list of dicts; some return dict
            if isinstance(preds, list):
                best = max(preds, key=lambda x: x.get("score", 0.0))
            else:
                best = preds
            label = str(best.get("label", "neutral"))
            score = float(best.get("score", 0.5))
            return self._map_label(label), score
        except Exception as e:
            logger.debug(f"Sentiment inference error: {e}")
            return "neutral", 0.5


# --------------------------- Intent Parsing --------------------------------
IntentHandler = Callable[[str, str], Dict[str, Any]]  # (text, lang) -> result


class IntentInterpreter:
    """Regex + keyword-based interpreter.

    Intents:
      - play music on spotify
      - search cats on youtube
      - open chrome and search weather / search on chrome weather
      - open/close apps (chrome, firefox, spotify, code, vlc)
      - open websites/urls
      - google <query>
    """

    def __init__(self):
        self.patterns: List[Tuple[str, re.Pattern]] = [
            ("spotify_play", re.compile(r"\b(?:open\s+)?spotify\b.*?\bplay\b\s*(?:['\"“”]?)(.+?)(?:['\"“”])?$", re.I)),
            ("youtube_search", re.compile(r"\b(?:search|play)\b\s*(.+?)\s*(?:on\s+)?youtube\b", re.I)),
            ("google_search", re.compile(r"\b(?:google|search)\b\s*(.+)$", re.I)),
            ("open_chrome_search", re.compile(r"\b(?:open\s+chrome\s+and\s+search|search\s+on\s+chrome)\b\s*(.+)$", re.I)),
            ("open_app", re.compile(r"\b(?:open|launch|start)\b\s+(chrome|firefox|spotify|code|vscode|vlc)\b", re.I)),
            ("close_app", re.compile(r"\b(?:close|quit|exit)\b\s+(chrome|firefox|spotify|code|vscode|vlc)\b", re.I)),
            ("open_url", re.compile(r"\b(open|launch)\b\s+((?:https?://)?[\w.-]+\.[a-z]{2,}[^\s]*)", re.I)),
            ("time", re.compile(r"\b(time|what\s+time\s+is\s+it)\b", re.I)),
        ]

    def interpret(self, text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        for name, pat in self.patterns:
            m = pat.search(text)
            if not m:
                continue
            if name in {"spotify_play", "youtube_search", "google_search", "open_chrome_search", "open_url"}:
                query = m.group(m.lastindex or 1).strip()
                return name, {"query": query}
            if name in {"open_app", "close_app"}:
                app = m.group(1).lower()
                return name, {"app": app}
            if name == "time":
                return name, {}
        # Fallback intent heuristics
        if re.search(r"\b(play)\b", text, re.I) and re.search(r"\bspotify\b", text, re.I):
            return "spotify_play", {"query": text}
        if re.search(r"\byoutube\b", text, re.I):
            q = re.sub(r".*\byoutube\b", "", text, flags=re.I).strip() or text
            return "youtube_search", {"query": q}
        if re.search(r"\bsearch\b|\bgoogle\b", text, re.I):
            q = re.sub(r".*?(?:search|google)\b", "", text, flags=re.I).strip() or text
            return "google_search", {"query": q}
        return None, {}


# --------------------------- App Control -----------------------------------
class AppController:
    """System-level actions (Linux-friendly, cross-platform aware)."""

    def __init__(self, default_browser: Optional[str] = None):
        self.system = platform.system().lower()
        self.default_browser = default_browser

    # --- helpers ---
    def _open_url(self, url: str, browser: Optional[str] = None) -> None:
        if not re.match(r"^https?://", url, re.I):
            url = f"https://{url}"
        try:
            if self.system == "linux":
                if browser:
                    subprocess.Popen([browser, url])
                else:
                    subprocess.Popen(["xdg-open", url])
            elif self.system == "darwin":
                if browser:
                    subprocess.Popen(["open", "-a", browser, url])
                else:
                    subprocess.Popen(["open", url])
            else:  # windows
                if browser:
                    subprocess.Popen([browser, url], shell=True)
                else:
                    os.startfile(url)  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Failed to open URL: {e}")

    def _launch_app(self, app: str) -> bool:
        try:
            if self.system == "linux":
                candidates = {
                    "chrome": ["google-chrome", "chromium", "chromium-browser"],
                    "firefox": ["firefox"],
                    "spotify": ["spotify"],
                    "code": ["code"],
                    "vscode": ["code"],
                    "vlc": ["vlc"],
                }.get(app, [app])
                for c in candidates:
                    if shutil_which(c):
                        subprocess.Popen([c])
                        return True
                return False
            elif self.system == "darwin":
                app_map = {
                    "chrome": "Google Chrome",
                    "firefox": "Firefox",
                    "spotify": "Spotify",
                    "code": "Visual Studio Code",
                    "vscode": "Visual Studio Code",
                    "vlc": "VLC",
                }
                name = app_map.get(app, app)
                subprocess.Popen(["open", "-a", name])
                return True
            else:  # windows
                subprocess.Popen([app], shell=True)
                return True
        except Exception as e:
            logger.error(f"Failed to launch app {app}: {e}")
            return False

    def _kill_app(self, app: str) -> bool:
        try:
            if self.system == "linux":
                subprocess.Popen(["pkill", "-f", app])
                return True
            elif self.system == "darwin":
                subprocess.Popen(["pkill", "-x", app])
                return True
            else:  # windows
                subprocess.Popen(["taskkill", "/IM", f"{app}.exe", "/F"], shell=True)
                return True
        except Exception as e:
            logger.error(f"Failed to close app {app}: {e}")
            return False

    # --- public actions ---
    def act(self, intent: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if intent == "spotify_play":
                q = params.get("query") or ""
                if not q:
                    return {"status": "error", "reason": "no query"}
                # Try native spotify client, else open web
                if self._launch_app("spotify"):
                    # Best effort: open spotify search url too
                    url = f"https://open.spotify.com/search/{url_quote(q)}"
                    self._open_url(url)
                else:
                    url = f"https://open.spotify.com/search/{url_quote(q)}"
                    self._open_url(url)
                return {"status": "ok", "service": "spotify", "query": q}

            if intent == "youtube_search":
                q = params.get("query") or ""
                if not q:
                    return {"status": "error", "reason": "no query"}
                url = f"https://www.youtube.com/results?search_query={url_quote(q)}"
                self._open_url(url)
                return {"status": "ok", "service": "youtube", "query": q}

            if intent == "google_search":
                q = params.get("query") or ""
                if not q:
                    return {"status": "error", "reason": "no query"}
                url = f"https://www.google.com/search?q={url_quote(q)}"
                self._open_url(url, browser=self.default_browser)
                return {"status": "ok", "service": "google", "query": q}

            if intent == "open_chrome_search":
                q = params.get("query") or ""
                if not q:
                    return {"status": "error", "reason": "no query"}
                url = f"https://www.google.com/search?q={url_quote(q)}"
                self._open_url(url, browser="google-chrome")
                return {"status": "ok", "service": "google", "query": q, "browser": "google-chrome"}

            if intent == "open_app":
                app = params.get("app") or ""
                ok = self._launch_app(app)
                return {"status": "ok" if ok else "error", "action": "open_app", "app": app}

            if intent == "close_app":
                app = params.get("app") or ""
                ok = self._kill_app(app)
                return {"status": "ok" if ok else "error", "action": "close_app", "app": app}

            if intent == "open_url":
                target = params.get("query") or ""
                if not target:
                    return {"status": "error", "reason": "no url"}
                self._open_url(target)
                return {"status": "ok", "url": target}

            if intent == "time":
                from datetime import datetime
                return {"status": "ok", "time": datetime.now().strftime("%H:%M:%S")}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        return {"status": "skipped", "reason": "no-op"}


# --------------------------- Response + TTS ---------------------------------
class ResponseGenerator:
    """Generate a short response, optionally via OpenAI, tuned by sentiment and language."""

    def __init__(self, cfg: ZoraConfig):
        self.cfg = cfg
        self._openai_ready = False
        if self.cfg.llm in ("auto", "openai") and HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            try:
                openai.api_key = os.getenv("OPENAI_API_KEY")
                self._openai_ready = True
            except Exception:
                self._openai_ready = False

    def _tone_prefix(self, sentiment: str) -> str:
        return {
            "positive": "😊",
            "neutral": "",
            "negative": "🤝",
        }.get(sentiment, "")

    def _prompt(self, text: str, sentiment: str, lang: str) -> str:
        style = {
            "positive": "cheerful and concise",
            "neutral": "clear and concise",
            "negative": "empathetic and concise",
        }.get(sentiment, "clear and concise")
        return (
            f"Respond in language '{lang}'. Keep it {style}. "
            f"User said: {text}"
        )

    def generate(self, text: str, sentiment: str, lang: str) -> str:
        lang = normalize_lang_code(lang)
        if self._openai_ready:
            try:
                rsp = openai.ChatCompletion.create(
                    model=self.cfg.openai_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful voice assistant."},
                        {"role": "user", "content": self._prompt(text, sentiment, lang)},
                    ],
                    temperature=0.4,
                    max_tokens=120,
                )
                msg = rsp["choices"][0]["message"]["content"].strip()
                return msg
            except Exception as e:
                logger.debug(f"OpenAI generation failed: {e}")
        # Fallback handcrafted reply
        tone = {
            "positive": "Great!",
            "neutral": "Okay.",
            "negative": "I understand.",
        }.get(sentiment, "Okay.")
        return f"{tone} I'll take care of that."


class TextToSpeech:
    """TTS with gTTS default and pyttsx3 fallback; async playback."""

    def __init__(self, cfg: ZoraConfig):
        self.cfg = cfg
        self.engine_name = (cfg.tts_engine if HAS_GTTS or HAS_PYTTSX3 else "none")
        if self.engine_name not in {"gtts", "pyttsx3"}:
            self.engine_name = "gtts" if HAS_GTTS else ("pyttsx3" if HAS_PYTTSX3 else "none")
        self._queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stop = threading.Event()
        self._thread.start()
        atexit.register(self.shutdown)

    def speak_async(self, text: str, lang: str) -> None:
        lang = normalize_lang_code(lang)
        if self.engine_name == "none":
            logger.info(f"TTS disabled: {text}")
            return
        self._queue.put((text, lang))

    def shutdown(self) -> None:
        try:
            self._stop.set()
            self._queue.put(("", ""))
        except Exception:
            pass

    def _worker(self):
        while not self._stop.is_set():
            try:
                text, lang = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not text:
                continue
            try:
                if self.engine_name == "gtts" and HAS_GTTS:
                    self._speak_gtts(text, lang)
                elif self.engine_name == "pyttsx3" and HAS_PYTTSX3:
                    self._speak_pyttsx3(text, lang)
                else:
                    logger.info(f"[TTS] {text}")
            except Exception as e:
                logger.warning(f"TTS failed: {e}")

    def _speak_gtts(self, text: str, lang: str) -> None:
        fn = None
        try:
            tts = gTTS(text=text, lang=lang)
            fd, path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            fn = path
            tts.save(fn)
            # Play using OS tools
            system = platform.system().lower()
            if system == "linux":
                # Try mpv, mpg123, or xdg-open
                if shutil_which("mpv"):
                    subprocess.Popen(["mpv", "--really-quiet", fn])
                elif shutil_which("mpg123"):
                    subprocess.Popen(["mpg123", "-q", fn])
                else:
                    subprocess.Popen(["xdg-open", fn])
            elif system == "darwin":
                subprocess.Popen(["afplay", fn])
            else:
                os.startfile(fn)  # type: ignore[attr-defined]
        finally:
            # schedule deletion later to allow player to read
            if fn:
                threading.Timer(20.0, lambda: safe_unlink(fn)).start()

    def _speak_pyttsx3(self, text: str, lang: str) -> None:
        engine = pyttsx3.init()  # type: ignore
        try:
            # Try to select a voice matching language
            try:
                voices = engine.getProperty("voices")  # type: ignore
                for v in voices:
                    # voice languages vary by engine; best-effort prefix match
                    if hasattr(v, "languages"):
                        langs = [str(x).lower() for x in v.languages]
                        if any(lang.split("-")[0] in l for l in langs):
                            engine.setProperty("voice", v.id)
                            break
                    elif hasattr(v, "id") and lang in str(v.id).lower():
                        engine.setProperty("voice", v.id)
                        break
            except Exception:
                pass
            engine.setProperty("rate", 190)  # type: ignore
            engine.say(text)  # type: ignore
            engine.runAndWait()  # type: ignore
        finally:
            try:
                engine.stop()  # type: ignore
            except Exception:
                pass


# --------------------------- Orchestrator ----------------------------------
@dataclass
class Analysis:
    sentiment: str
    sentiment_score: float


class ZoraAssistant:
    def __init__(self, cfg: ZoraConfig):
        self.cfg = cfg
        self.stt = SpeechToText(cfg)
        self.intent = IntentInterpreter()
        self.apps = AppController(default_browser=cfg.default_browser)
        self.sentiment = SentimentAnalyzer(cfg)
        self.reply = ResponseGenerator(cfg)
        self.tts = TextToSpeech(cfg)

    def process_text(self, text: str, lang: Optional[str]) -> Dict[str, Any]:
        if not text:
            return {"status": "empty"}
        sentiment, s_score = self.sentiment.analyze(text)
        intent_name, params = self.intent.interpret(text)
        action_result = {"status": "skipped"}
        if intent_name:
            action_result = self.apps.act(intent_name, params)
        reply_text = self.reply.generate(text, sentiment, lang or "en")
        self.tts.speak_async(reply_text, lang or "en")
        return {
            "status": "ok",
            "text": text,
            "lang": lang,
            "analysis": Analysis(sentiment=sentiment, sentiment_score=s_score).__dict__,
            "intent": intent_name,
            "action": action_result,
            "reply": reply_text,
        }

    def run_loop(self, stop_event: Optional[threading.Event] = None) -> None:
        logger.info("ZORA is listening. Say something. (Ctrl+C to exit)")
        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                text, lang = self.stt.listen_once(timeout=None, phrase_time_limit=None)
                if not text:
                    continue
                logger.info(f"Heard: {text} (lang={lang})")
                res = self.process_text(text, lang)
                logger.debug(json.dumps(res, ensure_ascii=False))
            except KeyboardInterrupt:
                logger.info("Exiting on keyboard interrupt.")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(0.5)


# --------------------------- Helpers ---------------------------------------
def shutil_which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)


def url_quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s)


def safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except Exception:
        pass


# --------------------------- CLI -------------------------------------------
def build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Project ZORA - Unified Voice Assistant")
    parser.add_argument("--once", action="store_true", help="Listen once and exit")
    parser.add_argument("--text", type=str, help="Bypass STT and process this text directly", default=None)
    parser.add_argument("--lang", type=str, help="Language code for --text path (e.g., en, hi)")
    parser.add_argument("--llm", type=str, choices=["none", "openai", "auto"], default=os.getenv("ZORA_LLM", "none"))
    parser.add_argument("--tts", type=str, choices=["gtts", "pyttsx3"], default=os.getenv("ZORA_TTS_ENGINE", "gtts"))
    parser.add_argument("--sentiment-model", type=str, default=os.getenv("ZORA_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"))
    parser.add_argument("--whisper-model", type=str, default=os.getenv("ZORA_WHISPER_MODEL", "small"))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = ZoraConfig(
        sentiment_model=args.sentiment_model,
        whisper_model=args.whisper_model,
        tts_engine=args.tts,
        llm=args.llm,
    )

    bot = ZoraAssistant(cfg)

    if args.text:
        res = bot.process_text(args.text, normalize_lang_code(args.lang or cfg.language_hint))
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if not HAS_SR:
        logger.error("speech_recognition is required for microphone mode. Install with: pip install SpeechRecognition pyaudio")
        return 2

    if args.once:
        text, lang = bot.stt.listen_once()
        if text:
            res = bot.process_text(text, lang)
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            logger.info("No speech captured.")
        return 0

    stop_event = threading.Event()

    def handle_sig(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    bot.run_loop(stop_event=stop_event)
    return 0


if __name__ == "__main__":
    sys.exit(main())
