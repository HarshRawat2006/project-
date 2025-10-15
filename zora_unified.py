#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project ZORA - Unified Voice Assistant (Linux-first)

Features
- Continuous STT (Google via speech_recognition; optional Whisper via faster-whisper)
- Automatic language detection (Whisper or langdetect fallback)
- Sentiment + (optional) emotion analysis via HuggingFace transformers
- Intent parsing for common tasks (YouTube, Spotify, Google/Chrome search, open apps, time, notes, etc.)
- Action execution via subprocess/webbrowser (Linux-first; basic macOS/Windows fallbacks)
- TTS responses in the same language (gTTS primary, pyttsx3 fallback)
- Modular functions/classes, simple logging, and a responsive main loop

Notes
- Dependencies are optional; the assistant degrades gracefully if some are missing.
- Runs best on Linux with a working microphone and audio output.

CLI examples
- python zora_unified.py --engine auto --loop
- python zora_unified.py --engine whisper --once
- python zora_unified.py --no-actions --print-only
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import math
import queue
import shutil
import atexit
import signal
import random
import string
import logging
import platform
import tempfile
import threading
import subprocess
import webbrowser
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

# ---------------------------- Optional imports & environment ----------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Optional STT packages
try:
    import speech_recognition as sr  # type: ignore
except Exception:
    sr = None  # type: ignore

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore

# Optional NLP packages
try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None  # type: ignore

try:
    from langdetect import detect as detect_lang  # type: ignore
except Exception:
    detect_lang = None  # type: ignore

# Optional TTS packages
try:
    from gtts import gTTS  # type: ignore
    from gtts.lang import tts_langs as gtts_langs  # type: ignore
except Exception:
    gTTS = None  # type: ignore
    gtts_langs = None  # type: ignore

try:
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None  # type: ignore

# Optional search helpers
try:
    from duckduckgo_search import DDGS  # type: ignore
except Exception:
    DDGS = None  # type: ignore

try:
    from pytube import Search as YTSearch  # type: ignore
except Exception:
    YTSearch = None  # type: ignore

# ---------------------------- Config & logging ----------------------------
DEFAULT_STT_ENGINE = os.getenv("ZORA_STT_ENGINE", "auto")  # auto|google|whisper
WHISPER_MODEL_NAME = os.getenv("ZORA_WHISPER_MODEL", "small")
SENTIMENT_MODEL_EN = os.getenv("ZORA_SENTIMENT_MODEL_EN", "cardiffnlp/twitter-roberta-base-sentiment-latest")
SENTIMENT_MODEL_MULTI = os.getenv("ZORA_SENTIMENT_MODEL_MULTI", "lxyuan/distilbert-base-multilingual-cased-sentiments-student")
EMOTION_MODEL_EN = os.getenv("ZORA_EMOTION_MODEL_EN", "j-hartmann/emotion-english-distilroberta-base")

LOG_LEVEL = os.getenv("ZORA_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("zora")

# ---------------------------- Data classes ----------------------------
@dataclass
class TranscriptionResult:
    text: str
    language: str  # ISO-639-1 when possible (e.g., 'en', 'hi')
    confidence: float
    engine: str  # 'google' | 'whisper'

@dataclass
class AnalysisResult:
    sentiment: str  # positive|neutral|negative
    sentiment_score: float
    emotion: str  # joy|sadness|anger|fear|surprise|disgust|neutral
    emotion_score: float

@dataclass
class Intent:
    kind: str
    data: Dict[str, Any]
    raw: str

# ---------------------------- Utilities ----------------------------
def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))

# ---------------------------- STT Service ----------------------------
class STTService:
    def __init__(self, engine: str = DEFAULT_STT_ENGINE, mic_index: Optional[int] = None) -> None:
        self.engine = self._decide_engine(engine)
        self.mic_index = mic_index
        self._whisper_model: Optional[Any] = None
        logger.info(f"STT engine: {self.engine}")

    def _decide_engine(self, engine: str) -> str:
        if engine == "auto":
            if WhisperModel is not None:
                return "whisper"
            if sr is not None:
                return "google"
            return "none"
        return engine

    def _ensure_whisper(self) -> None:
        if self._whisper_model is None and WhisperModel is not None:
            device = "cuda" if shutil.which("nvidia-smi") else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info(f"Loading Whisper model '{WHISPER_MODEL_NAME}' ({device}/{compute_type}) ...")
            self._whisper_model = WhisperModel(WHISPER_MODEL_NAME, device=device, compute_type=compute_type)

    def listen_once(self, timeout: Optional[float] = 10.0, phrase_time_limit: Optional[float] = 12.0) -> Optional[TranscriptionResult]:
        if self.engine == "none":
            logger.error("No STT engine available. Install 'speech_recognition' or 'faster-whisper'.")
            return None
        if sr is None:
            if self.engine == "google":
                logger.error("speech_recognition not installed.")
                return None
        if self.engine == "whisper" and WhisperModel is None:
            logger.warning("faster-whisper not installed; falling back to Google STT if available.")
            self.engine = "google"

        if self.engine == "whisper":
            return self._listen_whisper(timeout, phrase_time_limit)
        else:
            return self._listen_google(timeout, phrase_time_limit)

    # Google STT via speech_recognition
    def _listen_google(self, timeout: Optional[float], phrase_time_limit: Optional[float]) -> Optional[TranscriptionResult]:
        assert sr is not None
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone(device_index=self.mic_index) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                logger.info("Listening ... (Google STT)")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except Exception as e:
            logger.error(f"Microphone error: {e}")
            return None

        try:
            # Google Web Speech API requires a language; we'll pass None->default en-US.
            # We will detect language from the returned text using langdetect.
            text = recognizer.recognize_google(audio)  # type: ignore
            logger.info(f"Heard: {text}")
        except sr.UnknownValueError:  # type: ignore
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:  # type: ignore
            logger.error(f"Google STT request failed: {e}")
            return None

        language = "en"
        if detect_lang is not None:
            try:
                language = detect_lang(text)
            except Exception:
                language = "en"
        return TranscriptionResult(text=text, language=language, confidence=0.6, engine="google")

    # Whisper STT via faster-whisper
    def _listen_whisper(self, timeout: Optional[float], phrase_time_limit: Optional[float]) -> Optional[TranscriptionResult]:
        self._ensure_whisper()
        if self._whisper_model is None:
            return None
        assert sr is not None
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone(device_index=self.mic_index) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.4)
                logger.info("Listening ... (Whisper)")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except Exception as e:
            logger.error(f"Microphone error: {e}")
            return None

        # Save audio to a temp WAV file for whisper
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(audio.get_wav_data())
            segments, info = self._whisper_model.transcribe(
                tmp_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=450),
            )
            os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return None

        text_parts = []
        try:
            for seg in segments:
                text_parts.append(seg.text)
        except Exception:
            pass
        text = " ".join(t.strip() for t in text_parts if t and t.strip())
        language = getattr(info, "language", None) or "en"
        logger.info(f"Heard[{language}]: {text}")
        return TranscriptionResult(text=text, language=language, confidence=float(getattr(info, "language_probability", 0.8)), engine="whisper")

# ---------------------------- Sentiment & Emotion ----------------------------
class SentimentEmotionAnalyzer:
    def __init__(self) -> None:
        self._sentiment_en = None
        self._sentiment_multi = None
        self._emotion_en = None
        if pipeline is not None:
            try:
                self._sentiment_en = pipeline("text-classification", model=SENTIMENT_MODEL_EN, top_k=None)
            except Exception as e:
                logger.warning(f"Could not load EN sentiment model: {e}")
            try:
                self._sentiment_multi = pipeline("text-classification", model=SENTIMENT_MODEL_MULTI, top_k=None)
            except Exception as e:
                logger.warning(f"Could not load MULTI sentiment model: {e}")
            try:
                self._emotion_en = pipeline("text-classification", model=EMOTION_MODEL_EN, top_k=None)
            except Exception as e:
                logger.warning(f"Could not load EN emotion model: {e}")
        else:
            logger.warning("transformers not installed; sentiment defaults to neutral.")

    def analyze(self, text: str, language: str) -> AnalysisResult:
        sentiment_label = "neutral"
        sentiment_score = 0.5
        emotion_label = "neutral"
        emotion_score = 0.5

        # Sentiment
        pipe = None
        if language.startswith("en") and self._sentiment_en is not None:
            pipe = self._sentiment_en
        elif self._sentiment_multi is not None:
            pipe = self._sentiment_multi

        if pipe is not None:
            try:
                preds = pipe(text)[0]
                best = max(preds, key=lambda x: float(x.get("score", 0.0)))
                label = str(best.get("label", "")).lower()
                score = float(best.get("score", 0.0))
                if "pos" in label or "positive" in label:
                    sentiment_label = "positive"
                elif "neg" in label or "negative" in label:
                    sentiment_label = "negative"
                elif "neu" in label or "neutral" in label:
                    sentiment_label = "neutral"
                else:
                    # Some multilingual heads emit 1-5 stars
                    try:
                        stars = int(re.sub(r"\D", "", label) or "3")
                        if stars <= 2:
                            sentiment_label = "negative"
                        elif stars == 3:
                            sentiment_label = "neutral"
                        else:
                            sentiment_label = "positive"
                    except Exception:
                        sentiment_label = "neutral"
                sentiment_score = clamp(score)
            except Exception as e:
                logger.debug(f"Sentiment inference failed: {e}")

        # Emotion (EN model only; fallback to sentiment)
        if language.startswith("en") and self._emotion_en is not None:
            try:
                preds = self._emotion_en(text)[0]
                best = max(preds, key=lambda x: float(x.get("score", 0.0)))
                emotion_label = str(best.get("label", "")).lower()
                emotion_score = clamp(float(best.get("score", 0.0)))
            except Exception as e:
                logger.debug(f"Emotion inference failed: {e}")
        else:
            # Map sentiment to a coarse emotion when non-EN or model unavailable
            if sentiment_label == "positive":
                emotion_label, emotion_score = "joy", sentiment_score
            elif sentiment_label == "negative":
                emotion_label, emotion_score = "sadness", sentiment_score
            else:
                emotion_label, emotion_score = "neutral", 0.5

        return AnalysisResult(
            sentiment=sentiment_label,
            sentiment_score=sentiment_score,
            emotion=emotion_label,
            emotion_score=emotion_score,
        )

# ---------------------------- Intent Interpreter ----------------------------
class IntentInterpreter:
    def __init__(self) -> None:
        self.patterns: Dict[str, re.Pattern] = {
            # YouTube
            "youtube_play": re.compile(r"\b(?:play|search)\b\s+(.+?)\s+\b(?:on\s+)?youtube\b|\byoutube\b\s+(?:for\s+)?(.+)$", re.I),
            # Spotify
            "spotify_play": re.compile(r"\b(?:open\s+)?spotify\b.*?\bplay\b\s*(?:\"|\'|“|”)?([^\"\'“”]+)(?:\"|\'|“|”)?", re.I),
            # Chrome search
            "chrome_search": re.compile(r"\b(?:open\s+)?chrome\b.*?\bsearch\b\s*(.+)$", re.I),
            # Google search (generic)
            "google_search": re.compile(r"\b(?:google|search\s+(?:for\s+)?)\b\s*(.+)$", re.I),
            # Open app
            "open_app": re.compile(r"\b(?:open|launch|start)\b\s+(chrome|firefox|spotify|code|vscode|terminal|gnome-terminal|xterm|calculator|nautilus|files)\b", re.I),
            # Open website
            "open_website": re.compile(r"\b(?:open|launch)\b\s+((?:https?://)?[\w.-]+\.[a-z]{2,}(?:/\S*)?)", re.I),
            # Play local music
            "play_music_local": re.compile(r"\bplay\b\s+music\b", re.I),
            # Time
            "time": re.compile(r"\b(time|what\s+time\s+is\s+it)\b", re.I),
            # Exit
            "exit": re.compile(r"\b(exit|quit|stop|goodbye)\b", re.I),
        }

    def interpret(self, text: str) -> Optional[Intent]:
        t = text.strip()
        for kind, pat in self.patterns.items():
            m = pat.search(t)
            if not m:
                continue
            if kind == "youtube_play":
                query = (m.group(1) or m.group(2) or "").strip()
                return Intent(kind=kind, data={"query": query}, raw=t)
            if kind == "spotify_play":
                song = (m.group(1) or "").strip()
                return Intent(kind=kind, data={"song": song}, raw=t)
            if kind == "chrome_search":
                query = (m.group(1) or "").strip()
                return Intent(kind=kind, data={"query": query}, raw=t)
            if kind == "google_search":
                query = (m.group(1) or "").strip()
                return Intent(kind=kind, data={"query": query}, raw=t)
            if kind == "open_app":
                app = (m.group(1) or "").lower().strip()
                return Intent(kind=kind, data={"app": app}, raw=t)
            if kind == "open_website":
                url = (m.group(1) or "").strip()
                return Intent(kind=kind, data={"url": url}, raw=t)
            if kind == "play_music_local":
                return Intent(kind=kind, data={}, raw=t)
            if kind == "time":
                return Intent(kind=kind, data={}, raw=t)
            if kind == "exit":
                return Intent(kind=kind, data={}, raw=t)
        return None

# ---------------------------- Action Executor ----------------------------
class ActionExecutor:
    def __init__(self) -> None:
        self.system = platform.system().lower()

    def _which_first(self, candidates: Tuple[str, ...]) -> Optional[str]:
        for c in candidates:
            p = shutil.which(c)
            if p:
                return p
        return None

    def open_url(self, url: str) -> Dict[str, Any]:
        if not re.match(r"^https?://", url, re.I):
            url = f"https://{url}"
        webbrowser.open(url)
        return {"status": "ok", "url": url}

    def open_app(self, app: str) -> Dict[str, Any]:
        app = app.lower()
        if self.system == "linux":
            mapping = {
                "chrome": ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"),
                "firefox": ("firefox",),
                "spotify": ("spotify",),
                "code": ("code",),
                "vscode": ("code",),
                "terminal": ("gnome-terminal", "x-terminal-emulator", "konsole", "xterm"),
                "gnome-terminal": ("gnome-terminal",),
                "xterm": ("xterm",),
                "calculator": ("gnome-calculator", "kcalc"),
                "nautilus": ("nautilus",),
                "files": ("nautilus", "thunar", "dolphin"),
            }
            exe = self._which_first(mapping.get(app, (app,)))
            if not exe:
                return {"status": "error", "reason": f"app not found: {app}"}
            subprocess.Popen([exe])
            return {"status": "ok", "launched": exe}
        elif self.system == "darwin":
            try:
                subprocess.Popen(["open", "-a", app])
                return {"status": "ok", "launched": app}
            except Exception as e:
                return {"status": "error", "reason": str(e)}
        else:  # windows minimal
            try:
                subprocess.Popen(app, shell=True)
                return {"status": "ok", "launched": app}
            except Exception as e:
                return {"status": "error", "reason": str(e)}

    def search_youtube(self, query: str) -> Dict[str, Any]:
        if not query:
            return self.open_url("https://www.youtube.com")
        # Try to get first result URL via pytube
        yt_url: Optional[str] = None
        if YTSearch is not None:
            try:
                s = YTSearch(query)
                results = getattr(s, "results", None) or []
                if not results:
                    s.get_next_results()
                    results = getattr(s, "results", None) or []
                if results:
                    vid = results[0].video_id
                    if vid:
                        yt_url = f"https://www.youtube.com/watch?v={vid}&autoplay=1"
            except Exception:
                yt_url = None
        if yt_url:
            webbrowser.open(yt_url)
            return {"status": "ok", "query": query, "url": yt_url, "autoplay": True}
        url = f"https://www.youtube.com/results?search_query={self._quote(query)}"
        webbrowser.open(url)
        return {"status": "ok", "query": query, "url": url, "autoplay": False}

    def play_spotify(self, song: str) -> Dict[str, Any]:
        if not song:
            return self.open_app("spotify")
        url = f"https://open.spotify.com/search/{self._quote(song)}"
        webbrowser.open(url)
        return {"status": "ok", "query": song, "url": url}

    def google_search(self, query: str, prefer_first_result: bool = True) -> Dict[str, Any]:
        if prefer_first_result and DDGS is not None:
            try:
                with DDGS() as ddgs:
                    res = next(ddgs.text(query, max_results=1), None)
                if res and res.get("href"):
                    webbrowser.open(res["href"])  # open first result
                    return {"status": "ok", "query": query, "url": res["href"], "first_result": True}
            except Exception:
                pass
        url = f"https://www.google.com/search?q={self._quote(query)}"
        webbrowser.open(url)
        return {"status": "ok", "query": query, "url": url, "first_result": False}

    def chrome_search(self, query: str) -> Dict[str, Any]:
        exe = self._which_first(("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"))
        url = f"https://www.google.com/search?q={self._quote(query)}"
        if exe:
            subprocess.Popen([exe, url])
            return {"status": "ok", "browser": exe, "url": url}
        # fallback
        return self.open_url(url)

    def play_local_music(self) -> Dict[str, Any]:
        music_dir = os.path.expanduser("~/Music")
        if not os.path.isdir(music_dir):
            return {"status": "error", "reason": f"music folder not found: {music_dir}"}
        candidates = []
        for root, _dirs, files in os.walk(music_dir):
            for f in files:
                if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg")):
                    candidates.append(os.path.join(root, f))
        if not candidates:
            return {"status": "error", "reason": "no audio files found in ~/Music"}
        target = sorted(candidates)[0]
        self._open_media_file(target)
        return {"status": "ok", "file": target}

    def tell_time(self) -> Dict[str, Any]:
        import datetime as _dt
        now = _dt.datetime.now().strftime("%H:%M:%S")
        return {"status": "ok", "time": now}

    def _quote(self, s: str) -> str:
        try:
            from urllib.parse import quote
            return quote(s)
        except Exception:
            return s

    def _open_media_file(self, path: str) -> None:
        if self.system == "linux":
            if shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", path])
                return
            # CLI players fallbacks
            for player in ("mpg123", "mpv", "ffplay", "play"):
                exe = shutil.which(player)
                if exe:
                    args = [exe, path]
                    if player == "ffplay":
                        args = [exe, "-nodisp", "-autoexit", "-loglevel", "quiet", path]
                    subprocess.Popen(args)
                    return
        elif self.system == "darwin":
            subprocess.Popen(["open", path])
        else:
            os.startfile(path)  # type: ignore[attr-defined]

# ---------------------------- Speech (TTS) Service ----------------------------
class SpeechService:
    def __init__(self) -> None:
        self.supported_langs = None
        if gtts_langs is not None:
            try:
                self.supported_langs = gtts_langs()
            except Exception:
                self.supported_langs = None
        self._lock = threading.Lock()

    def speak_async(self, text: str, language: str, tone: str = "neutral") -> None:
        th = threading.Thread(target=self._speak_blocking, args=(text, language, tone), daemon=True)
        th.start()

    def _speak_blocking(self, text: str, language: str, tone: str = "neutral") -> None:
        lang = self._resolve_tts_language(language)
        text_prefixed = self._apply_tone_prefix(text, tone)
        if gTTS is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    path = tmp.name
                gTTS(text=text_prefixed, lang=lang, slow=False).save(path)
                self._play_audio_file(path)
                return
            except Exception as e:
                logger.warning(f"gTTS failed: {e}")
        # Fallback to pyttsx3
        if pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                # Slight tone-driven rate tweak
                rate = engine.getProperty('rate')
                if tone == "cheerful":
                    engine.setProperty('rate', rate + 20)
                elif tone == "empathetic":
                    engine.setProperty('rate', rate - 10)
                engine.say(text)
                engine.runAndWait()
                return
            except Exception as e:
                logger.warning(f"pyttsx3 failed: {e}")
        # Final fallback
        logger.info(f"Assistant: {text}")

    def _play_audio_file(self, path: str) -> None:
        system = platform.system().lower()
        try:
            if system == "linux":
                if shutil.which("xdg-open"):
                    subprocess.Popen(["xdg-open", path])
                    return
                for player in ("mpg123", "mpv", "ffplay", "play"):
                    exe = shutil.which(player)
                    if exe:
                        args = [exe, path]
                        if player == "ffplay":
                            args = [exe, "-nodisp", "-autoexit", "-loglevel", "quiet", path]
                        subprocess.Popen(args)
                        return
            elif system == "darwin":
                subprocess.Popen(["open", path])
            else:
                os.startfile(path)  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"Audio playback failed: {e}. Text: {path}")

    def _resolve_tts_language(self, code: str) -> str:
        c = (code or "en").split("-")[0].lower()
        if self.supported_langs and c in self.supported_langs:
            return c
        # Map some common codes
        mapping = {
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
            "zh": "zh-CN",
            "ar": "ar",
        }
        return mapping.get(c, "en")

    def _apply_tone_prefix(self, text: str, tone: str) -> str:
        prefixes = {
            "cheerful": "",
            "empathetic": "",
            "serious": "",
            "neutral": "",
        }
        return f"{prefixes.get(tone, '')}{text}"

# ---------------------------- Zora Assistant ----------------------------
class ZoraAssistant:
    def __init__(self, stt: STTService, analyzer: SentimentEmotionAnalyzer, interpreter: IntentInterpreter, executor: ActionExecutor, speaker: SpeechService, speak_enabled: bool = True, actions_enabled: bool = True) -> None:
        self.stt = stt
        self.analyzer = analyzer
        self.interpreter = interpreter
        self.executor = executor
        self.speaker = speaker
        self.speak_enabled = speak_enabled
        self.actions_enabled = actions_enabled
        self._stop_flag = threading.Event()

    def stop(self) -> None:
        self._stop_flag.set()

    def process_text(self, text: str, language: str) -> Dict[str, Any]:
        analysis = self.analyzer.analyze(text, language)
        intent = self.interpreter.interpret(text)
        action_result: Dict[str, Any] = {"status": "skipped", "reason": "no intent"}

        if intent and self.actions_enabled:
            try:
                action_result = self._execute_intent(intent)
            except Exception as e:
                action_result = {"status": "error", "error": str(e)}

        reply = self._craft_reply(text, analysis, intent)
        tone = self._tone_from_analysis(analysis)
        if self.speak_enabled and reply:
            self.speaker.speak_async(reply, language=language, tone=tone)
        return {
            "reply": reply,
            "analysis": analysis.__dict__,
            "intent": (intent.kind if intent else None),
            "action_result": action_result,
        }

    def _tone_from_analysis(self, analysis: AnalysisResult) -> str:
        if analysis.sentiment == "positive":
            return "cheerful"
        if analysis.sentiment == "negative":
            return "empathetic"
        return "serious"

    def _craft_reply(self, _text: str, analysis: AnalysisResult, intent: Optional[Intent]) -> str:
        openers = {
            "anger": "I hear your frustration.",
            "disgust": "I understand that felt unpleasant.",
            "fear": "I get that this feels worrying.",
            "joy": "That's great to hear!",
            "neutral": "Alright.",
            "sadness": "I'm sorry you're dealing with this.",
            "surprise": "That's unexpected!",
        }
        opener = openers.get(analysis.emotion, "Understood.")
        strategy = "I'll handle it now." if analysis.sentiment != "positive" else "Let's keep that momentum."
        if intent is None:
            return f"{opener} How can I help further?"
        # Short summaries per intent
        summaries = {
            "youtube_play": "Opening YouTube",
            "spotify_play": "Opening Spotify",
            "chrome_search": "Searching on Chrome",
            "google_search": "Searching the web",
            "open_app": "Launching app",
            "open_website": "Opening website",
            "play_music_local": "Playing music from your library",
            "time": "Telling the time",
            "exit": "Goodbye",
        }
        return f"{opener} {summaries.get(intent.kind, 'Okay')}. {strategy}"

    def _execute_intent(self, intent: Intent) -> Dict[str, Any]:
        k = intent.kind
        d = intent.data
        if k == "youtube_play":
            return self.executor.search_youtube(d.get("query", ""))
        if k == "spotify_play":
            return self.executor.play_spotify(d.get("song", ""))
        if k == "chrome_search":
            return self.executor.chrome_search(d.get("query", ""))
        if k == "google_search":
            return self.executor.google_search(d.get("query", ""))
        if k == "open_app":
            return self.executor.open_app(d.get("app", ""))
        if k == "open_website":
            return self.executor.open_url(d.get("url", ""))
        if k == "play_music_local":
            return self.executor.play_local_music()
        if k == "time":
            return self.executor.tell_time()
        if k == "exit":
            self.stop()
            return {"status": "ok", "stopping": True}
        return {"status": "skipped", "reason": f"unknown intent: {k}"}

    def run_once(self) -> Optional[Dict[str, Any]]:
        if self._stop_flag.is_set():
            return None
        heard = self.stt.listen_once()
        if not heard or not heard.text:
            return None
        return self.process_text(heard.text, language=heard.language)

    def run_loop(self) -> None:
        logger.info('Say something like: "play lofi on youtube", "open chrome and search weather", "open spotify and play shape of you". Say "exit" to quit.')
        while not self._stop_flag.is_set():
            try:
                result = self.run_once()
                if result and result.get("intent") == "exit":
                    break
            except KeyboardInterrupt:
                logger.info("Interrupted by user.")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")

# ---------------------------- CLI ----------------------------
def parse_args(argv: Optional[list[str]] = None):  # Python 3.9+
    import argparse
    p = argparse.ArgumentParser(description="Project ZORA - Unified Voice Assistant")
    p.add_argument("--engine", choices=["auto", "google", "whisper"], default=DEFAULT_STT_ENGINE, help="STT engine")
    p.add_argument("--mic-index", type=int, default=None, help="Microphone device index (speech_recognition)")
    p.add_argument("--once", action="store_true", help="Run a single listen/process/respond step")
    p.add_argument("--loop", action="store_true", help="Run continuously until 'exit'")
    p.add_argument("--no-actions", action="store_true", help="Do not execute system actions")
    p.add_argument("--print-only", action="store_true", help="Do not speak responses (print only)")
    p.add_argument("--text", type=str, help="Process a provided text instead of using the mic")
    p.add_argument("--lang", type=str, default="", help="Language code for --text path (e.g., en, hi)")
    p.add_argument("--log-level", type=str, default=LOG_LEVEL, help="Logging level")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    stt = STTService(engine=args.engine, mic_index=args.mic_index)
    analyzer = SentimentEmotionAnalyzer()
    interpreter = IntentInterpreter()
    executor = ActionExecutor()
    speaker = SpeechService()

    assistant = ZoraAssistant(
        stt=stt,
        analyzer=analyzer,
        interpreter=interpreter,
        executor=executor,
        speaker=speaker,
        speak_enabled=not args.print_only,
        actions_enabled=not args.no_actions,
    )

    # Text-only mode
    if args.text:
        language = args.lang or (detect_lang(args.text) if detect_lang else "en")
        res = assistant.process_text(args.text, language=language)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    # STT modes
    if args.once:
        res = assistant.run_once()
        if res:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    # Default to loop if --loop or no explicit mode
    assistant.run_loop() if args.loop or True else None
    return 0


if __name__ == "__main__":
    sys.exit(main())
