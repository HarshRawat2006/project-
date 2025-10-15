#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project ZORA - Unified Voice Assistant
=====================================

A comprehensive voice assistant that combines speech-to-text, sentiment analysis,
natural language interpretation, automation, and text-to-speech capabilities.

Features:
- Continuous speech recognition with language detection
- Real-time sentiment and emotion analysis
- Intent-based command interpretation
- System automation and app control
- Multilingual text-to-speech responses
- Contextual and emotional response generation

Author: Project ZORA Team
Version: 1.0.0
"""

import os
import re
import sys
import json
import time
import logging
import threading
import webbrowser
import subprocess
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Callable, Optional, List
from datetime import datetime

# Core dependencies
import requests
from dotenv import load_dotenv

# Speech Recognition
try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False
    print("⚠️  speech_recognition not available. Install with: pip install SpeechRecognition")

# Language Detection
try:
    from langdetect import detect, LangDetectError
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("⚠️  langdetect not available. Install with: pip install langdetect")

# Sentiment Analysis
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers not available. Install with: pip install transformers torch")

# Alternative sentiment analysis
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("⚠️  textblob not available. Install with: pip install textblob")

# Text-to-Speech
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False
    print("⚠️  gtts not available. Install with: pip install gtts")

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False
    print("⚠️  pyttsx3 not available. Install with: pip install pyttsx3")

# Optional integrations
try:
    from pytube import Search as YTSearch
    HAS_PYTUBE = True
except ImportError:
    HAS_PYTUBE = False
    YTSearch = None

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    DDGS = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    HAS_SPOTIFY = True
except ImportError:
    HAS_SPOTIFY = False
    spotipy = None
    SpotifyOAuth = None

# Load environment variables
load_dotenv()

# Configuration
EMOTION_MODEL = os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
WAKE_WORD = os.getenv("WAKE_WORD", "zora").lower()

# Language mappings for TTS
LANGUAGE_CODES = {
    'en': 'en',
    'es': 'es', 
    'fr': 'fr',
    'de': 'de',
    'it': 'it',
    'pt': 'pt',
    'ru': 'ru',
    'ja': 'ja',
    'ko': 'ko',
    'zh': 'zh',
    'hi': 'hi',
    'ar': 'ar'
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zora.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('ZORA')


@dataclass
class SpeechInput:
    """Represents processed speech input"""
    text: str
    language: str
    confidence: float
    timestamp: datetime


@dataclass
class EmotionAnalysis:
    """Represents emotion and sentiment analysis results"""
    emotion: str
    emotion_score: float
    sentiment: str
    sentiment_score: float


@dataclass
class CommandResult:
    """Represents the result of command execution"""
    success: bool
    action: str
    details: Dict[str, Any]
    response_text: str


class LanguageDetector:
    """Handles language detection for speech input"""
    
    def __init__(self):
        self.default_language = 'en'
        
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        if not HAS_LANGDETECT or not text.strip():
            return self.default_language
            
        try:
            detected = detect(text)
            return detected if detected in LANGUAGE_CODES else self.default_language
        except LangDetectError:
            return self.default_language


class SpeechToText:
    """Handles speech-to-text conversion with language detection"""
    
    def __init__(self):
        if not HAS_SPEECH_RECOGNITION:
            raise ImportError("speech_recognition is required for STT functionality")
            
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.language_detector = LanguageDetector()
        
        # Adjust for ambient noise
        logger.info("🎤 Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        logger.info("✅ Microphone calibrated")
        
    def listen_for_wake_word(self, timeout: int = 5) -> bool:
        """Listen for wake word activation"""
        try:
            with self.microphone as source:
                logger.info(f"🎧 Listening for wake word '{WAKE_WORD}'...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
                
            text = self.recognizer.recognize_google(audio, language="en-US").lower()
            return WAKE_WORD in text
            
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            return False
    
    def listen_for_command(self, timeout: int = 10) -> Optional[SpeechInput]:
        """Listen for voice command after wake word detection"""
        try:
            with self.microphone as source:
                logger.info("🎤 Listening for command...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            # Try to recognize with multiple languages
            for lang_code in ['en-US', 'es-ES', 'fr-FR', 'de-DE', 'hi-IN']:
                try:
                    text = self.recognizer.recognize_google(audio, language=lang_code)
                    if text.strip():
                        detected_lang = self.language_detector.detect_language(text)
                        logger.info(f"🗣️  Recognized: '{text}' (Language: {detected_lang})")
                        
                        return SpeechInput(
                            text=text,
                            language=detected_lang,
                            confidence=0.9,  # Google API doesn't provide confidence
                            timestamp=datetime.now()
                        )
                except sr.UnknownValueError:
                    continue
                    
            logger.warning("❌ Could not understand audio in any supported language")
            return None
            
        except sr.RequestError as e:
            logger.error(f"❌ Speech recognition service error: {e}")
            return None
        except sr.WaitTimeoutError:
            logger.warning("⏰ Listening timeout")
            return None


class SentimentAnalyzer:
    """Handles emotion and sentiment analysis"""
    
    def __init__(self):
        self.emotion_pipeline = None
        self.sentiment_pipeline = None
        
        # Initialize transformers pipelines if available
        if HAS_TRANSFORMERS:
            try:
                logger.info("🧠 Loading emotion analysis model...")
                self.emotion_pipeline = pipeline("text-classification", model=EMOTION_MODEL, top_k=None)
                logger.info("🧠 Loading sentiment analysis model...")
                self.sentiment_pipeline = pipeline("text-classification", model=SENTIMENT_MODEL, top_k=None)
                logger.info("✅ Sentiment analysis models loaded")
            except Exception as e:
                logger.warning(f"⚠️  Could not load transformers models: {e}")
                self.emotion_pipeline = None
                self.sentiment_pipeline = None
    
    def analyze(self, text: str) -> EmotionAnalysis:
        """Analyze emotion and sentiment of text"""
        emotion, emotion_score = self._predict_emotion(text)
        sentiment, sentiment_score = self._predict_sentiment(text)
        
        return EmotionAnalysis(
            emotion=emotion,
            emotion_score=emotion_score,
            sentiment=sentiment,
            sentiment_score=sentiment_score
        )
    
    def _predict_emotion(self, text: str) -> Tuple[str, float]:
        """Predict emotion using transformers or fallback"""
        if self.emotion_pipeline:
            try:
                preds = self.emotion_pipeline(text)[0]
                best = max(preds, key=lambda x: x["score"])
                return best["label"].lower(), float(best["score"])
            except Exception as e:
                logger.warning(f"Emotion prediction error: {e}")
        
        # Fallback to simple keyword-based emotion detection
        text_lower = text.lower()
        if any(word in text_lower for word in ['happy', 'great', 'awesome', 'love', 'excellent']):
            return 'joy', 0.8
        elif any(word in text_lower for word in ['sad', 'terrible', 'awful', 'hate', 'bad']):
            return 'sadness', 0.8
        elif any(word in text_lower for word in ['angry', 'mad', 'furious', 'annoyed']):
            return 'anger', 0.8
        elif any(word in text_lower for word in ['scared', 'afraid', 'worried', 'nervous']):
            return 'fear', 0.8
        else:
            return 'neutral', 0.6
    
    def _predict_sentiment(self, text: str) -> Tuple[str, float]:
        """Predict sentiment using transformers or TextBlob fallback"""
        if self.sentiment_pipeline:
            try:
                preds = self.sentiment_pipeline(text)[0]
                best = max(preds, key=lambda x: x["score"])
                label = str(best.get("label", "")).lower()
                
                if "pos" in label or "positive" in label:
                    mapped = "positive"
                elif "neg" in label or "negative" in label:
                    mapped = "negative"
                elif "neu" in label or "neutral" in label:
                    mapped = "neutral"
                else:
                    mapped = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}.get(label, "neutral")
                
                return mapped, float(best["score"])
            except Exception as e:
                logger.warning(f"Sentiment prediction error: {e}")
        
        # Fallback to TextBlob
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    return "positive", abs(polarity)
                elif polarity < -0.1:
                    return "negative", abs(polarity)
                else:
                    return "neutral", 1 - abs(polarity)
            except Exception:
                pass
        
        # Final fallback
        return "neutral", 0.5


class IntentClassifier:
    """Classifies user intents from speech input"""
    
    def __init__(self):
        self.intent_patterns = {
            # Media and Entertainment
            "spotify_play": re.compile(r"\b(?:play|open)\s+(?:spotify|music)\b.*?(?:play\s+)?(.*?)(?:\s+on\s+spotify)?$", re.I),
            "youtube_play": re.compile(r"\b(?:play|open|watch)\b.*?\b(?:on\s+)?youtube\b(?:\s+(.+))?$", re.I),
            "youtube_search": re.compile(r"\b(?:search|find)\b.*?\b(?:on\s+)?youtube\b(?:\s+(.+))?$", re.I),
            
            # Web and Search
            "google_search": re.compile(r"\b(?:google|search)\b(?:\s+for)?\s+(.+)$", re.I),
            "web_search": re.compile(r"\b(?:search|look\s+up|find)\b\s+(.+)$", re.I),
            "open_website": re.compile(r"\b(?:open|go\s+to)\s+((?:https?://)?[\w.-]+\.[a-z]{2,}\S*)$", re.I),
            
            # Applications
            "open_app": re.compile(r"\b(?:open|launch|start)\s+(notepad|calculator|vscode|code|spotify|word|excel|powerpoint|chrome|edge|firefox|cmd|powershell|terminal)$", re.I),
            
            # System Operations
            "system_info": re.compile(r"\b(?:what\s+)?(?:time|date|weather|system\s+info)\b", re.I),
            "volume_control": re.compile(r"\b(?:volume|sound)\s+(?:up|down|mute|unmute|set\s+to\s+\d+)$", re.I),
            
            # File Operations
            "open_folder": re.compile(r"\b(?:open|show)\s+(?:folder|directory)\s+(.+)$", re.I),
            "create_note": re.compile(r"\b(?:create|make|write)\s+(?:note|reminder)\s+(.+)$", re.I),
            
            # Communication
            "send_email": re.compile(r"\b(?:send|compose)\s+email\b(?:\s+to\s+(.+?))?(?:\s+subject\s+(.+?))?(?:\s+body\s+(.+?))?$", re.I),
            
            # Information
            "define_word": re.compile(r"\b(?:define|what\s+is|meaning\s+of)\s+(.+)$", re.I),
            "wikipedia": re.compile(r"\b(?:wiki|wikipedia)\s+(.+)$", re.I),
            "weather": re.compile(r"\b(?:weather|forecast)\b(?:\s+in\s+(.+?))?$", re.I),
            
            # Assistant Control
            "stop_listening": re.compile(r"\b(?:stop|quit|exit|goodbye|sleep)\b", re.I),
            "help": re.compile(r"\b(?:help|what\s+can\s+you\s+do)\b", re.I),
        }
    
    def classify_intent(self, text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Classify intent and extract parameters from text"""
        text = text.strip()
        
        for intent, pattern in self.intent_patterns.items():
            match = pattern.search(text)
            if match:
                # Extract parameters based on intent
                params = self._extract_parameters(intent, match, text)
                logger.info(f"🎯 Intent detected: {intent} with params: {params}")
                return intent, params
        
        logger.info("❓ No specific intent detected, treating as general query")
        return "general_query", {"query": text}
    
    def _extract_parameters(self, intent: str, match: re.Match, full_text: str) -> Dict[str, Any]:
        """Extract parameters based on intent type"""
        params = {}
        
        if intent in ["spotify_play", "youtube_play", "youtube_search"]:
            if match.groups():
                params["query"] = match.group(1).strip() if match.group(1) else ""
            
        elif intent in ["google_search", "web_search", "define_word", "wikipedia"]:
            if match.groups():
                params["query"] = match.group(1).strip()
                
        elif intent == "open_website":
            if match.groups():
                url = match.group(1).strip()
                if not url.startswith(('http://', 'https://')):
                    url = f"https://{url}"
                params["url"] = url
                
        elif intent == "open_app":
            if match.groups():
                params["app"] = match.group(1).strip().lower()
                
        elif intent == "open_folder":
            if match.groups():
                params["path"] = match.group(1).strip()
                
        elif intent == "create_note":
            if match.groups():
                params["content"] = match.group(1).strip()
                
        elif intent == "send_email":
            groups = match.groups()
            if groups:
                params["to"] = groups[0].strip() if groups[0] else ""
                params["subject"] = groups[1].strip() if len(groups) > 1 and groups[1] else ""
                params["body"] = groups[2].strip() if len(groups) > 2 and groups[2] else ""
                
        elif intent == "weather":
            if match.groups() and match.group(1):
                params["location"] = match.group(1).strip()
            else:
                params["location"] = "current location"
        
        return params


class AutomationEngine:
    """Handles system automation and app control"""
    
    def __init__(self):
        self.spotify_client = None
        if HAS_SPOTIFY and SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
            try:
                auth_manager = SpotifyOAuth(
                    client_id=SPOTIFY_CLIENT_ID,
                    client_secret=SPOTIFY_CLIENT_SECRET,
                    redirect_uri=SPOTIFY_REDIRECT_URI,
                    scope="user-modify-playback-state user-read-playback-state"
                )
                self.spotify_client = spotipy.Spotify(auth_manager=auth_manager)
            except Exception as e:
                logger.warning(f"Could not initialize Spotify client: {e}")
    
    def execute_command(self, intent: str, params: Dict[str, Any], analysis: EmotionAnalysis) -> CommandResult:
        """Execute automation command based on intent"""
        try:
            if intent == "spotify_play":
                return self._handle_spotify_play(params)
            elif intent in ["youtube_play", "youtube_search"]:
                return self._handle_youtube(params)
            elif intent in ["google_search", "web_search"]:
                return self._handle_web_search(params)
            elif intent == "open_website":
                return self._handle_open_website(params)
            elif intent == "open_app":
                return self._handle_open_app(params)
            elif intent == "system_info":
                return self._handle_system_info()
            elif intent == "open_folder":
                return self._handle_open_folder(params)
            elif intent == "create_note":
                return self._handle_create_note(params, analysis)
            elif intent == "define_word":
                return self._handle_define_word(params)
            elif intent == "wikipedia":
                return self._handle_wikipedia(params)
            elif intent == "weather":
                return self._handle_weather(params)
            elif intent == "help":
                return self._handle_help()
            elif intent == "general_query":
                return self._handle_general_query(params)
            else:
                return CommandResult(
                    success=False,
                    action=intent,
                    details={"error": "Unknown intent"},
                    response_text="I'm not sure how to handle that request."
                )
                
        except Exception as e:
            logger.error(f"Error executing command {intent}: {e}")
            return CommandResult(
                success=False,
                action=intent,
                details={"error": str(e)},
                response_text="Sorry, I encountered an error while processing your request."
            )
    
    def _handle_spotify_play(self, params: Dict[str, Any]) -> CommandResult:
        """Handle Spotify playback"""
        query = params.get("query", "")
        
        if not query:
            webbrowser.open("https://open.spotify.com")
            return CommandResult(
                success=True,
                action="open_spotify",
                details={"url": "https://open.spotify.com"},
                response_text="Opening Spotify for you."
            )
        
        if self.spotify_client:
            try:
                results = self.spotify_client.search(q=query, type="track", limit=1)
                tracks = results.get("tracks", {}).get("items", [])
                
                if tracks:
                    track = tracks[0]
                    self.spotify_client.start_playback(uris=[track["uri"]])
                    return CommandResult(
                        success=True,
                        action="spotify_play",
                        details={"track": track["name"], "artist": track["artists"][0]["name"]},
                        response_text=f"Now playing {track['name']} by {track['artists'][0]['name']} on Spotify."
                    )
            except Exception as e:
                logger.warning(f"Spotify API error: {e}")
        
        # Fallback to opening Spotify with search
        url = f"https://open.spotify.com/search/{requests.utils.quote(query)}"
        webbrowser.open(url)
        return CommandResult(
            success=True,
            action="spotify_search",
            details={"query": query, "url": url},
            response_text=f"Searching for '{query}' on Spotify."
        )
    
    def _handle_youtube(self, params: Dict[str, Any]) -> CommandResult:
        """Handle YouTube operations"""
        query = params.get("query", "")
        
        if not query:
            webbrowser.open("https://www.youtube.com")
            return CommandResult(
                success=True,
                action="open_youtube",
                details={"url": "https://www.youtube.com"},
                response_text="Opening YouTube for you."
            )
        
        # Try to get direct video URL using pytube
        if HAS_PYTUBE:
            try:
                search = YTSearch(query)
                results = search.results
                if results:
                    video = results[0]
                    url = f"https://www.youtube.com/watch?v={video.video_id}"
                    webbrowser.open(url)
                    return CommandResult(
                        success=True,
                        action="youtube_play",
                        details={"title": video.title, "url": url},
                        response_text=f"Playing '{video.title}' on YouTube."
                    )
            except Exception as e:
                logger.warning(f"YouTube search error: {e}")
        
        # Fallback to search results
        url = f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"
        webbrowser.open(url)
        return CommandResult(
            success=True,
            action="youtube_search",
            details={"query": query, "url": url},
            response_text=f"Searching for '{query}' on YouTube."
        )
    
    def _handle_web_search(self, params: Dict[str, Any]) -> CommandResult:
        """Handle web search"""
        query = params.get("query", "")
        
        if not query:
            return CommandResult(
                success=False,
                action="web_search",
                details={"error": "No search query provided"},
                response_text="What would you like me to search for?"
            )
        
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        webbrowser.open(url)
        return CommandResult(
            success=True,
            action="web_search",
            details={"query": query, "url": url},
            response_text=f"Searching for '{query}' on Google."
        )
    
    def _handle_open_website(self, params: Dict[str, Any]) -> CommandResult:
        """Handle opening websites"""
        url = params.get("url", "")
        
        if not url:
            return CommandResult(
                success=False,
                action="open_website",
                details={"error": "No URL provided"},
                response_text="Please specify a website to open."
            )
        
        webbrowser.open(url)
        return CommandResult(
            success=True,
            action="open_website",
            details={"url": url},
            response_text=f"Opening {url} for you."
        )
    
    def _handle_open_app(self, params: Dict[str, Any]) -> CommandResult:
        """Handle opening applications"""
        app = params.get("app", "").lower()
        
        app_commands = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "terminal": "cmd.exe",
            "chrome": "chrome.exe",
            "edge": "msedge.exe",
            "firefox": "firefox.exe",
            "vscode": "code",
            "code": "code",
            "word": "winword.exe",
            "excel": "excel.exe",
            "powerpoint": "powerpnt.exe",
            "spotify": "spotify.exe"
        }
        
        command = app_commands.get(app)
        if not command:
            return CommandResult(
                success=False,
                action="open_app",
                details={"error": f"Unknown application: {app}"},
                response_text=f"I don't know how to open {app}."
            )
        
        try:
            if os.name == 'nt':  # Windows
                subprocess.Popen(command, shell=True)
            else:  # Linux/Mac
                subprocess.Popen(command.split())
            
            return CommandResult(
                success=True,
                action="open_app",
                details={"app": app, "command": command},
                response_text=f"Opening {app} for you."
            )
        except Exception as e:
            return CommandResult(
                success=False,
                action="open_app",
                details={"error": str(e)},
                response_text=f"Could not open {app}. {str(e)}"
            )
    
    def _handle_system_info(self) -> CommandResult:
        """Handle system information requests"""
        current_time = datetime.now().strftime("%I:%M %p")
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        
        return CommandResult(
            success=True,
            action="system_info",
            details={"time": current_time, "date": current_date},
            response_text=f"It's currently {current_time} on {current_date}."
        )
    
    def _handle_open_folder(self, params: Dict[str, Any]) -> CommandResult:
        """Handle opening folders"""
        path = params.get("path", "")
        
        # Handle special folder names
        special_folders = {
            "desktop": os.path.expanduser("~/Desktop"),
            "documents": os.path.expanduser("~/Documents"),
            "downloads": os.path.expanduser("~/Downloads"),
            "pictures": os.path.expanduser("~/Pictures"),
            "music": os.path.expanduser("~/Music"),
            "videos": os.path.expanduser("~/Videos"),
        }
        
        if path.lower() in special_folders:
            path = special_folders[path.lower()]
        else:
            path = os.path.expanduser(path)
        
        if not os.path.exists(path):
            return CommandResult(
                success=False,
                action="open_folder",
                details={"error": f"Path does not exist: {path}"},
                response_text=f"I couldn't find the folder {path}."
            )
        
        try:
            if os.name == 'nt':  # Windows
                os.startfile(path)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', path])
            else:  # Linux
                subprocess.Popen(['xdg-open', path])
            
            return CommandResult(
                success=True,
                action="open_folder",
                details={"path": path},
                response_text=f"Opening {os.path.basename(path)} folder."
            )
        except Exception as e:
            return CommandResult(
                success=False,
                action="open_folder",
                details={"error": str(e)},
                response_text=f"Could not open folder. {str(e)}"
            )
    
    def _handle_create_note(self, params: Dict[str, Any], analysis: EmotionAnalysis) -> CommandResult:
        """Handle creating notes"""
        content = params.get("content", "")
        
        if not content:
            return CommandResult(
                success=False,
                action="create_note",
                details={"error": "No note content provided"},
                response_text="What would you like me to note down?"
            )
        
        # Create notes directory if it doesn't exist
        notes_dir = os.path.join(os.getcwd(), "notes")
        os.makedirs(notes_dir, exist_ok=True)
        
        # Create note file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"note_{timestamp}.txt"
        filepath = os.path.join(notes_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Note created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Emotion: {analysis.emotion} (confidence: {analysis.emotion_score:.2f})\n")
                f.write(f"Sentiment: {analysis.sentiment} (confidence: {analysis.sentiment_score:.2f})\n")
                f.write(f"\nContent:\n{content}\n")
            
            return CommandResult(
                success=True,
                action="create_note",
                details={"filepath": filepath, "content": content},
                response_text=f"I've saved your note: '{content[:50]}...'" if len(content) > 50 else f"I've saved your note: '{content}'"
            )
        except Exception as e:
            return CommandResult(
                success=False,
                action="create_note",
                details={"error": str(e)},
                response_text=f"Could not create note. {str(e)}"
            )
    
    def _handle_define_word(self, params: Dict[str, Any]) -> CommandResult:
        """Handle word definition requests"""
        query = params.get("query", "")
        
        if not query:
            return CommandResult(
                success=False,
                action="define_word",
                details={"error": "No word provided"},
                response_text="What word would you like me to define?"
            )
        
        url = f"https://www.dictionary.com/browse/{requests.utils.quote(query)}"
        webbrowser.open(url)
        return CommandResult(
            success=True,
            action="define_word",
            details={"word": query, "url": url},
            response_text=f"Looking up the definition of '{query}' for you."
        )
    
    def _handle_wikipedia(self, params: Dict[str, Any]) -> CommandResult:
        """Handle Wikipedia searches"""
        query = params.get("query", "")
        
        if not query:
            webbrowser.open("https://en.wikipedia.org")
            return CommandResult(
                success=True,
                action="wikipedia",
                details={"url": "https://en.wikipedia.org"},
                response_text="Opening Wikipedia for you."
            )
        
        url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(query.replace(' ', '_'))}"
        webbrowser.open(url)
        return CommandResult(
            success=True,
            action="wikipedia",
            details={"query": query, "url": url},
            response_text=f"Looking up '{query}' on Wikipedia."
        )
    
    def _handle_weather(self, params: Dict[str, Any]) -> CommandResult:
        """Handle weather requests"""
        location = params.get("location", "current location")
        
        # Open weather search on Google
        query = f"weather {location}" if location != "current location" else "weather"
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        webbrowser.open(url)
        
        return CommandResult(
            success=True,
            action="weather",
            details={"location": location, "url": url},
            response_text=f"Getting weather information for {location}."
        )
    
    def _handle_help(self) -> CommandResult:
        """Handle help requests"""
        help_text = """
        I can help you with:
        • Playing music on Spotify or YouTube
        • Searching the web or specific websites
        • Opening applications and folders
        • Creating notes and reminders
        • Getting system information like time and date
        • Looking up definitions and Wikipedia articles
        • Checking the weather
        
        Just speak naturally and I'll try to understand what you need!
        """
        
        return CommandResult(
            success=True,
            action="help",
            details={"help_text": help_text},
            response_text="Here's what I can help you with: I can play music, search the web, open apps, create notes, get system info, look up definitions, and check weather. Just speak naturally!"
        )
    
    def _handle_general_query(self, params: Dict[str, Any]) -> CommandResult:
        """Handle general queries"""
        query = params.get("query", "")
        
        # For now, just search the web for general queries
        if query:
            url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
            webbrowser.open(url)
            return CommandResult(
                success=True,
                action="general_query",
                details={"query": query, "url": url},
                response_text=f"I'm searching for information about '{query}'."
            )
        else:
            return CommandResult(
                success=False,
                action="general_query",
                details={"error": "No query provided"},
                response_text="I didn't catch that. Could you please repeat your request?"
            )


class TextToSpeech:
    """Handles text-to-speech conversion with multilingual support"""
    
    def __init__(self):
        self.tts_engine = None
        
        # Initialize pyttsx3 for offline TTS
        if HAS_PYTTSX3:
            try:
                self.tts_engine = pyttsx3.init()
                # Configure speech rate and volume
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('volume', 0.9)
                logger.info("✅ Offline TTS engine initialized")
            except Exception as e:
                logger.warning(f"Could not initialize pyttsx3: {e}")
                self.tts_engine = None
    
    def speak(self, text: str, language: str = 'en', emotion: str = 'neutral', use_offline: bool = True) -> bool:
        """Convert text to speech with emotion and language support"""
        if not text.strip():
            return False
        
        # Adjust text based on emotion
        text = self._apply_emotional_tone(text, emotion)
        
        # Try offline TTS first if requested and available
        if use_offline and self.tts_engine:
            try:
                self._speak_offline(text, emotion)
                return True
            except Exception as e:
                logger.warning(f"Offline TTS failed: {e}")
        
        # Fallback to online TTS
        if HAS_GTTS:
            try:
                self._speak_online(text, language)
                return True
            except Exception as e:
                logger.error(f"Online TTS failed: {e}")
        
        # Final fallback - print text
        logger.warning("TTS not available, printing text instead")
        print(f"🔊 ZORA: {text}")
        return False
    
    def _apply_emotional_tone(self, text: str, emotion: str) -> str:
        """Apply emotional tone to text"""
        tone_prefixes = {
            'joy': "😊 ",
            'excitement': "🎉 ",
            'sadness': "😔 ",
            'anger': "😠 ",
            'fear': "😰 ",
            'surprise': "😲 ",
            'neutral': ""
        }
        
        prefix = tone_prefixes.get(emotion, "")
        return f"{prefix}{text}"
    
    def _speak_offline(self, text: str, emotion: str):
        """Use pyttsx3 for offline speech"""
        if not self.tts_engine:
            raise Exception("Offline TTS engine not available")
        
        # Adjust speech rate based on emotion
        rate = self.tts_engine.getProperty('rate')
        if emotion == 'excitement':
            self.tts_engine.setProperty('rate', rate + 30)
        elif emotion in ['sadness', 'fear']:
            self.tts_engine.setProperty('rate', rate - 30)
        else:
            self.tts_engine.setProperty('rate', 180)
        
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def _speak_online(self, text: str, language: str):
        """Use gTTS for online speech"""
        if not HAS_GTTS:
            raise Exception("gTTS not available")
        
        # Map language codes
        tts_lang = LANGUAGE_CODES.get(language, 'en')
        
        # Create TTS object and save to temporary file
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        temp_file = "temp_speech.mp3"
        tts.save(temp_file)
        
        # Play the audio file
        try:
            if os.name == 'nt':  # Windows
                os.system(f'start /wait "" "{temp_file}"')
            elif sys.platform == 'darwin':  # macOS
                os.system(f'afplay "{temp_file}"')
            else:  # Linux
                os.system(f'mpg321 "{temp_file}" || aplay "{temp_file}"')
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_file)
            except:
                pass


class ResponseGenerator:
    """Generates contextual and emotional responses"""
    
    def __init__(self):
        self.response_templates = {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Good to see you! What would you like me to help with?"
            ],
            'acknowledgment': [
                "Got it! Let me take care of that for you.",
                "Understood! I'll handle that right away.",
                "Sure thing! Working on it now."
            ],
            'success': [
                "Done! Is there anything else you need?",
                "All set! What else can I help with?",
                "Completed successfully! Need anything else?"
            ],
            'error': [
                "I'm sorry, I couldn't complete that task.",
                "Oops! Something went wrong. Let me try a different approach.",
                "I encountered an issue. Could you try rephrasing your request?"
            ],
            'clarification': [
                "Could you please clarify what you'd like me to do?",
                "I'm not sure I understood. Could you rephrase that?",
                "Can you provide more details about what you need?"
            ]
        }
    
    def generate_response(self, command_result: CommandResult, analysis: EmotionAnalysis, language: str = 'en') -> str:
        """Generate contextual response based on command result and emotion"""
        base_response = command_result.response_text
        
        # Adjust response based on user's emotional state
        if analysis.sentiment == 'negative':
            if analysis.emotion in ['sadness', 'fear']:
                base_response = f"I understand this might be difficult. {base_response}"
            elif analysis.emotion == 'anger':
                base_response = f"I'll take care of this right away. {base_response}"
        elif analysis.sentiment == 'positive':
            if analysis.emotion == 'joy':
                base_response = f"Great! {base_response}"
        
        return base_response
    
    def get_wake_response(self, language: str = 'en') -> str:
        """Get response when wake word is detected"""
        responses = {
            'en': ["Yes? How can I help?", "I'm listening!", "What can I do for you?"],
            'es': ["¿Sí? ¿Cómo puedo ayudar?", "¡Te escucho!", "¿Qué puedo hacer por ti?"],
            'fr': ["Oui? Comment puis-je aider?", "Je vous écoute!", "Que puis-je faire pour vous?"],
            'de': ["Ja? Wie kann ich helfen?", "Ich höre zu!", "Was kann ich für Sie tun?"],
            'hi': ["हाँ? मैं कैसे मदद कर सकता हूँ?", "मैं सुन रहा हूँ!", "मैं आपके लिए क्या कर सकता हूँ?"]
        }
        
        import random
        return random.choice(responses.get(language, responses['en']))


class ProjectZORA:
    """Main Project ZORA voice assistant class"""
    
    def __init__(self):
        logger.info("🚀 Initializing Project ZORA...")
        
        # Initialize components
        self.stt = SpeechToText() if HAS_SPEECH_RECOGNITION else None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.intent_classifier = IntentClassifier()
        self.automation_engine = AutomationEngine()
        self.tts = TextToSpeech()
        self.response_generator = ResponseGenerator()
        
        # State management
        self.is_listening = False
        self.is_active = False
        self.command_queue = Queue()
        
        # Statistics
        self.session_stats = {
            'commands_processed': 0,
            'successful_commands': 0,
            'start_time': datetime.now()
        }
        
        logger.info("✅ Project ZORA initialized successfully!")
    
    def start(self):
        """Start the voice assistant"""
        if not self.stt:
            logger.error("❌ Speech recognition not available. Cannot start voice assistant.")
            print("❌ Speech recognition is required but not available.")
            print("Please install: pip install SpeechRecognition")
            return
        
        logger.info("🎤 Project ZORA is starting up...")
        self.tts.speak("Hello! I'm ZORA, your voice assistant. Say my name to activate me.", use_offline=True)
        
        self.is_active = True
        
        try:
            # Start command processing thread
            command_thread = threading.Thread(target=self._process_commands, daemon=True)
            command_thread.start()
            
            # Main listening loop
            self._main_listening_loop()
            
        except KeyboardInterrupt:
            logger.info("👋 Shutting down Project ZORA...")
            self.stop()
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            self.stop()
    
    def stop(self):
        """Stop the voice assistant"""
        self.is_active = False
        self.is_listening = False
        
        # Print session statistics
        duration = datetime.now() - self.session_stats['start_time']
        success_rate = (self.session_stats['successful_commands'] / 
                       max(self.session_stats['commands_processed'], 1)) * 100
        
        logger.info(f"📊 Session Summary:")
        logger.info(f"   Duration: {duration}")
        logger.info(f"   Commands processed: {self.session_stats['commands_processed']}")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        
        self.tts.speak("Goodbye! It was nice talking with you.", use_offline=True)
        logger.info("👋 Project ZORA shut down complete")
    
    def _main_listening_loop(self):
        """Main loop for listening for wake word and commands"""
        logger.info(f"👂 Listening for wake word: '{WAKE_WORD}'")
        
        while self.is_active:
            try:
                # Listen for wake word
                if self.stt.listen_for_wake_word(timeout=5):
                    logger.info(f"🎯 Wake word '{WAKE_WORD}' detected!")
                    
                    # Respond to wake word
                    wake_response = self.response_generator.get_wake_response()
                    self.tts.speak(wake_response, use_offline=True)
                    
                    # Listen for command
                    speech_input = self.stt.listen_for_command(timeout=10)
                    
                    if speech_input:
                        # Add command to processing queue
                        self.command_queue.put(speech_input)
                        self.session_stats['commands_processed'] += 1
                    else:
                        self.tts.speak("I didn't catch that. Please try again.", use_offline=True)
                
            except Exception as e:
                logger.error(f"Error in main listening loop: {e}")
                time.sleep(1)  # Brief pause before retrying
    
    def _process_commands(self):
        """Process commands from the queue"""
        while self.is_active:
            try:
                # Get command from queue (blocking with timeout)
                speech_input = self.command_queue.get(timeout=1)
                
                logger.info(f"🔄 Processing command: '{speech_input.text}'")
                
                # Check for stop command
                if any(word in speech_input.text.lower() for word in ['stop', 'quit', 'exit', 'goodbye', 'sleep']):
                    self.tts.speak("Goodbye!", language=speech_input.language, use_offline=True)
                    self.is_active = False
                    break
                
                # Analyze sentiment and emotion
                analysis = self.sentiment_analyzer.analyze(speech_input.text)
                logger.info(f"💭 Emotion: {analysis.emotion} ({analysis.emotion_score:.2f}), "
                          f"Sentiment: {analysis.sentiment} ({analysis.sentiment_score:.2f})")
                
                # Classify intent and extract parameters
                intent, params = self.intent_classifier.classify_intent(speech_input.text)
                
                # Execute command
                command_result = self.automation_engine.execute_command(intent, params, analysis)
                
                # Generate and speak response
                response = self.response_generator.generate_response(
                    command_result, analysis, speech_input.language
                )
                
                self.tts.speak(
                    response,
                    language=speech_input.language,
                    emotion=analysis.emotion,
                    use_offline=True
                )
                
                # Update statistics
                if command_result.success:
                    self.session_stats['successful_commands'] += 1
                
                logger.info(f"✅ Command processed successfully: {command_result.action}")
                
            except Empty:
                # No commands in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing command: {e}")
    
    def process_text_command(self, text: str, language: str = 'en') -> str:
        """Process a text command directly (for testing or text interface)"""
        try:
            # Create speech input object
            speech_input = SpeechInput(
                text=text,
                language=language,
                confidence=1.0,
                timestamp=datetime.now()
            )
            
            # Analyze sentiment and emotion
            analysis = self.sentiment_analyzer.analyze(text)
            
            # Classify intent and extract parameters
            intent, params = self.intent_classifier.classify_intent(text)
            
            # Execute command
            command_result = self.automation_engine.execute_command(intent, params, analysis)
            
            # Generate response
            response = self.response_generator.generate_response(command_result, analysis, language)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing text command: {e}")
            return f"Sorry, I encountered an error: {str(e)}"


def main():
    """Main entry point for Project ZORA"""
    print("🤖 Project ZORA - Unified Voice Assistant")
    print("=" * 50)
    
    # Check dependencies
    missing_deps = []
    if not HAS_SPEECH_RECOGNITION:
        missing_deps.append("speech_recognition")
    if not HAS_TRANSFORMERS and not HAS_TEXTBLOB:
        missing_deps.append("transformers OR textblob")
    if not HAS_GTTS and not HAS_PYTTSX3:
        missing_deps.append("gtts OR pyttsx3")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies and try again.")
        return
    
    # Create and start ZORA
    zora = ProjectZORA()
    
    # Check if we should run in text mode
    if len(sys.argv) > 1 and sys.argv[1] == "--text":
        print("\n🔤 Running in text mode (no voice)")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                text = input("You: ").strip()
                if text.lower() in ['quit', 'exit', 'stop']:
                    break
                if text:
                    response = zora.process_text_command(text)
                    print(f"ZORA: {response}")
            except KeyboardInterrupt:
                break
        
        print("\n👋 Goodbye!")
    else:
        # Run in voice mode
        zora.start()


if __name__ == "__main__":
    main()