#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            PROJECT ZORA                                      ║
║              Unified Voice Assistant with Sentiment Analysis                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

A production-ready, modular voice assistant that:
- Listens to user speech continuously
- Detects spoken language automatically
- Analyzes sentiment and emotional tone
- Interprets intent and executes commands
- Responds with context-aware, emotionally intelligent replies
- Speaks back in the same language as input

Author: Project ZORA Team
Version: 1.0.0
Date: 2025-10-15
"""

# ═══════════════════════════════════════════════════════════════════════════
#                           IMPORTS & DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════

import os
import re
import sys
import json
import time
import platform
import subprocess
import webbrowser
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

# Speech Recognition
import speech_recognition as sr

# Sentiment & Emotion Analysis
from transformers import pipeline

# Text-to-Speech
from gtts import gTTS
try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False
    print("⚠️  pyttsx3 not available - using gTTS only")

# Language Detection
from langdetect import detect, LangDetectException

# Web requests for automation
import requests

# Optional: Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional: YouTube integration
try:
    from pytube import Search as YTSearch
    HAS_PYTUBE = True
except ImportError:
    HAS_PYTUBE = False
    print("⚠️  pytube not available - YouTube autoplay disabled")

# Optional: Spotify integration
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    HAS_SPOTIFY = True
except ImportError:
    HAS_SPOTIFY = False
    spotipy = None
    SpotifyOAuth = None
    print("⚠️  spotipy not available - Spotify integration disabled")


# ═══════════════════════════════════════════════════════════════════════════
#                          CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Model Configuration
EMOTION_MODEL = os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")

# Spotify Configuration (optional)
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
SPOTIFY_DEVICE_ID = os.getenv("SPOTIFY_DEVICE_ID", "")

# Language Mappings (ISO 639-1 codes)
LANGUAGE_MAP = {
    'en': 'en',      # English
    'hi': 'hi',      # Hindi
    'es': 'es',      # Spanish
    'fr': 'fr',      # French
    'de': 'de',      # German
    'it': 'it',      # Italian
    'pt': 'pt',      # Portuguese
    'ru': 'ru',      # Russian
    'ja': 'ja',      # Japanese
    'ko': 'ko',      # Korean
    'zh-cn': 'zh-CN', # Chinese (Simplified)
    'ar': 'ar',      # Arabic
}

# Wake word for activation (optional)
WAKE_WORD = "zora"

# Assistant name
ASSISTANT_NAME = "ZORA"

# Audio output directory
AUDIO_OUTPUT_DIR = "./audio_output"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#                            DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpeechInput:
    """Represents processed speech input from the user"""
    text: str
    language: str
    confidence: float
    timestamp: str


@dataclass
class SentimentAnalysis:
    """Represents sentiment and emotion analysis results"""
    emotion: str
    emotion_score: float
    sentiment: str
    sentiment_score: float


@dataclass
class CommandIntent:
    """Represents the interpreted intent of a command"""
    intent_type: str
    confidence: float
    parameters: Dict[str, Any]
    raw_text: str


@dataclass
class AssistantResponse:
    """Represents the complete response from the assistant"""
    text: str
    language: str
    sentiment: SentimentAnalysis
    action_result: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════
#                      SPEECH-TO-TEXT (STT) MODULE
# ═══════════════════════════════════════════════════════════════════════════

class SpeechToTextEngine:
    """
    Handles continuous speech recognition with automatic language detection.
    Uses Google Speech Recognition API for accuracy and language support.
    """
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000  # Adjust for ambient noise
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Seconds of silence to consider phrase complete
        
    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[SpeechInput]:
        """
        Listen to microphone and convert speech to text.
        
        Args:
            timeout: Maximum time to wait for speech to start (seconds)
            phrase_time_limit: Maximum duration of the phrase (seconds)
            
        Returns:
            SpeechInput object or None if listening failed
        """
        try:
            with sr.Microphone() as source:
                print(f"\n🎤 {ASSISTANT_NAME} is listening...")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
                
                print("🔄 Processing speech...")
                
                # Try to recognize speech (returns tuple of text and language)
                # First, try with language detection
                text = self.recognizer.recognize_google(audio)
                
                if not text:
                    print("❌ No speech detected")
                    return None
                
                # Detect language
                detected_lang = self._detect_language(text)
                
                # Create SpeechInput object
                speech_input = SpeechInput(
                    text=text,
                    language=detected_lang,
                    confidence=0.9,  # Google API doesn't provide confidence
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                return speech_input
                
        except sr.WaitTimeoutError:
            print("⏱️  No speech detected within timeout period")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition service error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error in speech recognition: {e}")
            return None
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ISO 639-1 language code (defaults to 'en')
        """
        try:
            lang_code = detect(text)
            # Map to supported language
            return LANGUAGE_MAP.get(lang_code, 'en')
        except LangDetectException:
            # Default to English if detection fails
            return 'en'


# ═══════════════════════════════════════════════════════════════════════════
#                    SENTIMENT & EMOTION ANALYSIS MODULE
# ═══════════════════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    """
    Analyzes the emotional tone and sentiment of user input.
    Uses pre-trained transformer models for accurate emotion detection.
    """
    
    def __init__(self):
        print("🔄 Loading sentiment analysis models...")
        try:
            # Load emotion classification model
            self.emotion_pipeline = pipeline(
                "text-classification", 
                model=EMOTION_MODEL, 
                top_k=None
            )
            
            # Load sentiment classification model
            self.sentiment_pipeline = pipeline(
                "text-classification", 
                model=SENTIMENT_MODEL, 
                top_k=None
            )
            print("✅ Sentiment models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading sentiment models: {e}")
            self.emotion_pipeline = None
            self.sentiment_pipeline = None
    
    def analyze(self, text: str) -> SentimentAnalysis:
        """
        Perform comprehensive sentiment and emotion analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentAnalysis object with emotion and sentiment data
        """
        if not self.emotion_pipeline or not self.sentiment_pipeline:
            # Return neutral sentiment if models not loaded
            return SentimentAnalysis(
                emotion="neutral",
                emotion_score=1.0,
                sentiment="neutral",
                sentiment_score=1.0
            )
        
        # Analyze emotion
        emotion, emotion_score = self._predict_emotion(text)
        
        # Analyze sentiment
        sentiment, sentiment_score = self._predict_sentiment(text)
        
        return SentimentAnalysis(
            emotion=emotion,
            emotion_score=emotion_score,
            sentiment=sentiment,
            sentiment_score=sentiment_score
        )
    
    def _predict_emotion(self, text: str) -> Tuple[str, float]:
        """Predict the dominant emotion in the text"""
        try:
            predictions = self.emotion_pipeline(text)[0]
            best = max(predictions, key=lambda x: x["score"])
            return best["label"].lower(), float(best["score"])
        except Exception as e:
            print(f"⚠️  Emotion prediction error: {e}")
            return "neutral", 1.0
    
    def _predict_sentiment(self, text: str) -> Tuple[str, float]:
        """Predict the sentiment (positive/negative/neutral) of the text"""
        try:
            predictions = self.sentiment_pipeline(text)[0]
            best = max(predictions, key=lambda x: x["score"])
            label = str(best.get("label", "")).lower()
            
            # Map label to standard sentiment
            if "pos" in label or "positive" in label:
                mapped = "positive"
            elif "neg" in label or "negative" in label:
                mapped = "negative"
            elif "neu" in label or "neutral" in label:
                mapped = "neutral"
            else:
                # Handle numeric labels
                label_map = {
                    "label_0": "negative",
                    "label_1": "neutral",
                    "label_2": "positive"
                }
                mapped = label_map.get(label, "neutral")
            
            return mapped, float(best["score"])
        except Exception as e:
            print(f"⚠️  Sentiment prediction error: {e}")
            return "neutral", 1.0


# ═══════════════════════════════════════════════════════════════════════════
#                    INTENT RECOGNITION & PARSING MODULE
# ═══════════════════════════════════════════════════════════════════════════

class IntentRecognizer:
    """
    Parses user commands and identifies intent using pattern matching.
    Extracts parameters needed for command execution.
    """
    
    def __init__(self):
        # Define intent patterns with regex
        self.intent_patterns = {
            # Media & Entertainment
            "spotify_play": re.compile(r"\b(?:open\s+)?spotify\b.*?\bplay\b\s*([\"""']?)(.+?)\1\b", re.I),
            "youtube_play": re.compile(r"\b(?:play|open)\b.*?\b(?:on\s+)?youtube\b(?:\s*([\"""'])(.+?)\1|\s+(.+))?$", re.I),
            "youtube_search": re.compile(r"\byoutube\b.*?\bsearch\b\s*(.+)", re.I),
            "play_music": re.compile(r"\b(?:play|start)\b.*?\b(?:music|song|track)\b", re.I),
            
            # Web Search & Navigation
            "google_search": re.compile(r"\b(?:google|search\s+(?:on\s+)?google(?:\s+for)?)\b\s*(.+)?$", re.I),
            "web_search": re.compile(r"\b(?:search|look up|find)\b\s+(.+)", re.I),
            "open_website": re.compile(r"\b(?:open|go to)\s+((?:https?://)?[\w.-]+\.[a-z]{2,}\S*)\b", re.I),
            
            # Application Control
            "open_app": re.compile(r"\b(?:open|launch|start)\s+(notepad|calculator|vscode|code|spotify|chrome|edge|word|excel|powerpoint)\b", re.I),
            "close_app": re.compile(r"\b(?:close|exit|quit)\s+(.+)", re.I),
            
            # System Commands
            "get_time": re.compile(r"\b(?:what(?:'s| is) the )?(?:time|current time)\b", re.I),
            "get_date": re.compile(r"\b(?:what(?:'s| is) the )?(?:date|today|today's date)\b", re.I),
            
            # File & Folder Operations
            "open_folder": re.compile(r"\b(?:open|show)\s+(?:my\s+)?(documents|downloads|desktop|pictures|videos|music)\b", re.I),
            "create_note": re.compile(r"\b(?:create|make|write)\s+(?:a\s+)?note\b", re.I),
            
            # Information Queries
            "wikipedia": re.compile(r"\b(?:wiki|wikipedia)\b\s*(.+)?", re.I),
            "maps": re.compile(r"\b(?:map|maps|directions?|navigate)\b\s+(?:to\s+)?(.+)?", re.I),
            "define": re.compile(r"\b(?:define|meaning of|what is|what's)\b\s+(.+)", re.I),
            
            # Assistant Control
            "exit": re.compile(r"\b(?:exit|quit|stop|bye|goodbye)\b", re.I),
            "help": re.compile(r"\b(?:help|what can you do|commands)\b", re.I),
        }
    
    def recognize(self, text: str) -> CommandIntent:
        """
        Recognize the intent of the user's command.
        
        Args:
            text: User's command text
            
        Returns:
            CommandIntent object with intent type and parameters
        """
        text_lower = text.lower().strip()
        
        # Try to match against each pattern
        for intent_type, pattern in self.intent_patterns.items():
            match = pattern.search(text)
            if match:
                # Extract parameters based on intent type
                parameters = self._extract_parameters(intent_type, match, text)
                
                return CommandIntent(
                    intent_type=intent_type,
                    confidence=0.9,
                    parameters=parameters,
                    raw_text=text
                )
        
        # No specific intent recognized - treat as general query
        return CommandIntent(
            intent_type="general_query",
            confidence=0.5,
            parameters={"query": text},
            raw_text=text
        )
    
    def _extract_parameters(self, intent_type: str, match: re.Match, text: str) -> Dict[str, Any]:
        """Extract relevant parameters from the matched pattern"""
        params = {}
        
        try:
            if intent_type == "spotify_play":
                params["song"] = self._clean_quotes(match.group(2) if match.lastindex >= 2 else "")
            
            elif intent_type in ["youtube_play", "youtube_search"]:
                params["query"] = self._clean_quotes(match.group(2) or match.group(3) or match.group(1) or "")
            
            elif intent_type in ["google_search", "web_search"]:
                params["query"] = self._clean_quotes(match.group(1) or text)
            
            elif intent_type == "open_website":
                url = match.group(1)
                if not url.startswith("http"):
                    url = "https://" + url
                params["url"] = url
            
            elif intent_type == "open_app":
                params["app_name"] = match.group(1).lower()
            
            elif intent_type == "open_folder":
                params["folder"] = match.group(1).lower()
            
            elif intent_type == "wikipedia":
                params["query"] = self._clean_quotes(match.group(1) or "")
            
            elif intent_type == "maps":
                params["location"] = self._clean_quotes(match.group(1) or "")
            
            elif intent_type == "define":
                params["term"] = self._clean_quotes(match.group(1) or "")
        
        except Exception as e:
            print(f"⚠️  Parameter extraction error: {e}")
        
        return params
    
    @staticmethod
    def _clean_quotes(text: str) -> str:
        """Remove quotes and extra whitespace from text"""
        return text.strip().strip('"').strip("'").strip(""").strip(""").strip()


# ═══════════════════════════════════════════════════════════════════════════
#                    AUTOMATION & APP CONTROL MODULE
# ═══════════════════════════════════════════════════════════════════════════

class AutomationEngine:
    """
    Executes system-level commands and automates application control.
    Handles opening apps, searching the web, playing media, etc.
    """
    
    def __init__(self):
        self.platform = platform.system()
        self._spotify_client = None
        
        # Initialize Spotify client if available
        if HAS_SPOTIFY and SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
            try:
                auth_manager = SpotifyOAuth(
                    client_id=SPOTIFY_CLIENT_ID,
                    client_secret=SPOTIFY_CLIENT_SECRET,
                    redirect_uri=SPOTIFY_REDIRECT_URI,
                    scope="user-modify-playback-state user-read-playback-state"
                )
                self._spotify_client = spotipy.Spotify(auth_manager=auth_manager)
            except Exception as e:
                print(f"⚠️  Spotify initialization failed: {e}")
    
    def execute(self, intent: CommandIntent) -> Dict[str, Any]:
        """
        Execute the command based on the recognized intent.
        
        Args:
            intent: CommandIntent object with intent type and parameters
            
        Returns:
            Dictionary with execution result and status
        """
        intent_type = intent.intent_type
        params = intent.parameters
        
        # Map intent types to handler methods
        handlers = {
            "spotify_play": self._handle_spotify,
            "youtube_play": self._handle_youtube,
            "youtube_search": self._handle_youtube,
            "play_music": self._handle_play_music,
            "google_search": self._handle_google_search,
            "web_search": self._handle_web_search,
            "open_website": self._handle_open_website,
            "open_app": self._handle_open_app,
            "get_time": self._handle_get_time,
            "get_date": self._handle_get_date,
            "open_folder": self._handle_open_folder,
            "wikipedia": self._handle_wikipedia,
            "maps": self._handle_maps,
            "define": self._handle_define,
            "exit": self._handle_exit,
            "help": self._handle_help,
        }
        
        # Execute the appropriate handler
        handler = handlers.get(intent_type, self._handle_unknown)
        
        try:
            return handler(params)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing command: {str(e)}"
            }
    
    # ───────────────────────────────────────────────────────────────────────
    # Media & Entertainment Handlers
    # ───────────────────────────────────────────────────────────────────────
    
    def _handle_spotify(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open Spotify and play a song"""
        song = params.get("song", "")
        
        if not song:
            webbrowser.open("https://open.spotify.com")
            return {"status": "success", "message": "Opening Spotify"}
        
        # Try API-based playback first
        if self._spotify_client:
            try:
                results = self._spotify_client.search(q=song, type="track", limit=1)
                items = results.get("tracks", {}).get("items", [])
                
                if items:
                    uri = items[0]["uri"]
                    track_name = items[0]["name"]
                    artist = items[0]["artists"][0]["name"]
                    
                    self._spotify_client.start_playback(
                        device_id=SPOTIFY_DEVICE_ID or None,
                        uris=[uri]
                    )
                    
                    return {
                        "status": "success",
                        "message": f"Playing '{track_name}' by {artist} on Spotify",
                        "autoplay": True
                    }
            except Exception as e:
                print(f"⚠️  Spotify API playback failed: {e}")
        
        # Fallback to web search
        search_url = f"https://open.spotify.com/search/{requests.utils.quote(song)}"
        webbrowser.open(search_url)
        
        return {
            "status": "success",
            "message": f"Opening Spotify search for '{song}'",
            "autoplay": False
        }
    
    def _handle_youtube(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open YouTube and play/search for a video"""
        query = params.get("query", "")
        
        if not query:
            webbrowser.open("https://www.youtube.com")
            return {"status": "success", "message": "Opening YouTube"}
        
        # Try to get direct video link with autoplay
        if HAS_PYTUBE:
            try:
                search = YTSearch(query)
                results = search.results
                
                if not results:
                    search.get_next_results()
                    results = search.results
                
                if results:
                    video_id = results[0].video_id
                    video_url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
                    webbrowser.open(video_url)
                    
                    return {
                        "status": "success",
                        "message": f"Playing '{results[0].title}' on YouTube",
                        "autoplay": True
                    }
            except Exception as e:
                print(f"⚠️  YouTube direct play failed: {e}")
        
        # Fallback to search results
        search_url = f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"
        webbrowser.open(search_url)
        
        return {
            "status": "success",
            "message": f"Searching YouTube for '{query}'",
            "autoplay": False
        }
    
    def _handle_play_music(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Play music from local library or online"""
        # Try to open default music player with a music file
        if self.platform == "Windows":
            music_dir = os.path.expanduser("~/Music")
            if os.path.exists(music_dir):
                try:
                    songs = [f for f in os.listdir(music_dir) if f.endswith(('.mp3', '.wav', '.flac', '.m4a'))]
                    if songs:
                        music_file = os.path.join(music_dir, songs[0])
                        os.startfile(music_file)
                        return {"status": "success", "message": f"Playing {songs[0]}"}
                except Exception:
                    pass
        
        # Fallback to opening Spotify
        webbrowser.open("https://open.spotify.com")
        return {"status": "success", "message": "Opening Spotify for music"}
    
    # ───────────────────────────────────────────────────────────────────────
    # Web Search & Navigation Handlers
    # ───────────────────────────────────────────────────────────────────────
    
    def _handle_google_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a Google search"""
        query = params.get("query", "")
        if not query:
            webbrowser.open("https://www.google.com")
            return {"status": "success", "message": "Opening Google"}
        
        search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        webbrowser.open(search_url)
        
        return {"status": "success", "message": f"Searching Google for '{query}'"}
    
    def _handle_web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a web search"""
        return self._handle_google_search(params)
    
    def _handle_open_website(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open a specific website"""
        url = params.get("url", "")
        if not url:
            return {"status": "error", "message": "No URL provided"}
        
        webbrowser.open(url)
        return {"status": "success", "message": f"Opening {url}"}
    
    # ───────────────────────────────────────────────────────────────────────
    # Application Control Handlers
    # ───────────────────────────────────────────────────────────────────────
    
    def _handle_open_app(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open a specific application"""
        app_name = params.get("app_name", "notepad")
        
        if self.platform == "Windows":
            return self._open_app_windows(app_name)
        elif self.platform == "Darwin":  # macOS
            return self._open_app_macos(app_name)
        else:  # Linux
            return self._open_app_linux(app_name)
    
    def _open_app_windows(self, app_name: str) -> Dict[str, Any]:
        """Open application on Windows"""
        app_commands = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "chrome": "chrome",
            "edge": "microsoft-edge:",
            "vscode": "code",
            "code": "code",
            "spotify": "spotify",
            "word": "winword",
            "excel": "excel",
            "powerpoint": "powerpnt",
        }
        
        command = app_commands.get(app_name, app_name)
        
        try:
            subprocess.Popen(f'start "" {command}', shell=True)
            return {"status": "success", "message": f"Opening {app_name}"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to open {app_name}: {str(e)}"}
    
    def _open_app_macos(self, app_name: str) -> Dict[str, Any]:
        """Open application on macOS"""
        app_commands = {
            "chrome": "Google Chrome",
            "vscode": "Visual Studio Code",
            "code": "Visual Studio Code",
            "spotify": "Spotify",
        }
        
        app = app_commands.get(app_name, app_name)
        
        try:
            subprocess.Popen(["open", "-a", app])
            return {"status": "success", "message": f"Opening {app_name}"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to open {app_name}: {str(e)}"}
    
    def _open_app_linux(self, app_name: str) -> Dict[str, Any]:
        """Open application on Linux"""
        app_commands = {
            "chrome": "google-chrome",
            "vscode": "code",
            "code": "code",
            "spotify": "spotify",
        }
        
        command = app_commands.get(app_name, app_name)
        
        try:
            subprocess.Popen([command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"status": "success", "message": f"Opening {app_name}"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to open {app_name}: {str(e)}"}
    
    # ───────────────────────────────────────────────────────────────────────
    # System Information Handlers
    # ───────────────────────────────────────────────────────────────────────
    
    def _handle_get_time(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get current time"""
        current_time = datetime.now().strftime("%I:%M %p")
        return {
            "status": "success",
            "message": f"The current time is {current_time}",
            "data": {"time": current_time}
        }
    
    def _handle_get_date(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get current date"""
        current_date = datetime.now().strftime("%B %d, %Y")
        day_of_week = datetime.now().strftime("%A")
        return {
            "status": "success",
            "message": f"Today is {day_of_week}, {current_date}",
            "data": {"date": current_date, "day": day_of_week}
        }
    
    # ───────────────────────────────────────────────────────────────────────
    # File & Folder Handlers
    # ───────────────────────────────────────────────────────────────────────
    
    def _handle_open_folder(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open a system folder"""
        folder_name = params.get("folder", "documents")
        home = os.path.expanduser("~")
        
        folder_map = {
            "documents": os.path.join(home, "Documents"),
            "downloads": os.path.join(home, "Downloads"),
            "desktop": os.path.join(home, "Desktop"),
            "pictures": os.path.join(home, "Pictures"),
            "videos": os.path.join(home, "Videos"),
            "music": os.path.join(home, "Music"),
        }
        
        folder_path = folder_map.get(folder_name)
        
        if not folder_path or not os.path.exists(folder_path):
            return {"status": "error", "message": f"Folder '{folder_name}' not found"}
        
        try:
            if self.platform == "Windows":
                os.startfile(folder_path)
            elif self.platform == "Darwin":
                subprocess.Popen(["open", folder_path])
            else:
                subprocess.Popen(["xdg-open", folder_path])
            
            return {"status": "success", "message": f"Opening {folder_name} folder"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to open folder: {str(e)}"}
    
    # ───────────────────────────────────────────────────────────────────────
    # Information Query Handlers
    # ───────────────────────────────────────────────────────────────────────
    
    def _handle_wikipedia(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search Wikipedia"""
        query = params.get("query", "")
        
        if not query:
            webbrowser.open("https://en.wikipedia.org")
            return {"status": "success", "message": "Opening Wikipedia"}
        
        url = f"https://en.wikipedia.org/w/index.php?search={requests.utils.quote(query)}"
        webbrowser.open(url)
        
        return {"status": "success", "message": f"Searching Wikipedia for '{query}'"}
    
    def _handle_maps(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Open Google Maps with location"""
        location = params.get("location", "")
        
        if not location:
            webbrowser.open("https://www.google.com/maps")
            return {"status": "success", "message": "Opening Google Maps"}
        
        url = f"https://www.google.com/maps/search/{requests.utils.quote(location)}"
        webbrowser.open(url)
        
        return {"status": "success", "message": f"Finding '{location}' on Google Maps"}
    
    def _handle_define(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Define a term"""
        term = params.get("term", "")
        
        if not term:
            return {"status": "error", "message": "No term provided"}
        
        url = f"https://www.dictionary.com/browse/{requests.utils.quote(term)}"
        webbrowser.open(url)
        
        return {"status": "success", "message": f"Looking up definition of '{term}'"}
    
    # ───────────────────────────────────────────────────────────────────────
    # Control & Misc Handlers
    # ───────────────────────────────────────────────────────────────────────
    
    def _handle_exit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Exit the assistant"""
        return {"status": "exit", "message": "Goodbye! Shutting down ZORA."}
    
    def _handle_help(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Show help information"""
        help_text = """
        I can help you with:
        • Opening applications (Chrome, Spotify, etc.)
        • Playing music and videos
        • Searching the web
        • Getting time and date
        • Opening folders
        • Searching Wikipedia, Maps
        • And much more!
        """
        return {"status": "success", "message": help_text.strip()}
    
    def _handle_unknown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unknown commands"""
        return {
            "status": "unknown",
            "message": "I'm not sure how to help with that. Try asking me something else!"
        }


# ═══════════════════════════════════════════════════════════════════════════
#                    RESPONSE GENERATION MODULE
# ═══════════════════════════════════════════════════════════════════════════

class ResponseGenerator:
    """
    Generates context-aware, emotionally intelligent responses.
    Adapts tone based on user's sentiment and command outcome.
    """
    
    def __init__(self):
        # Response templates by emotion
        self.emotion_openers = {
            "anger": [
                "I hear your frustration.",
                "I understand this is frustrating.",
                "Let me help you with that right away."
            ],
            "joy": [
                "That's great to hear!",
                "Wonderful!",
                "I'm happy to help!"
            ],
            "sadness": [
                "I'm sorry you're dealing with this.",
                "I understand.",
                "Let me help you with that."
            ],
            "fear": [
                "Don't worry, I've got this.",
                "Let me handle that for you.",
                "I'll take care of it."
            ],
            "surprise": [
                "That's interesting!",
                "Okay!",
                "Got it!"
            ],
            "disgust": [
                "I understand.",
                "Let me help with that.",
                "I'll handle it."
            ],
            "neutral": [
                "Sure.",
                "Okay.",
                "Understood.",
                "Alright."
            ]
        }
        
        # Response closers by sentiment
        self.sentiment_closers = {
            "positive": [
                "All set! Anything else I can help with?",
                "Done! Let me know if you need anything else.",
                "There you go! What else can I do for you?"
            ],
            "negative": [
                "I'll take care of everything for you.",
                "Don't worry, I've got this handled.",
                "I'm here to help."
            ],
            "neutral": [
                "Ready for the next step whenever you are.",
                "Is there anything else you need?",
                "Let me know if you need anything else."
            ]
        }
    
    def generate(
        self, 
        sentiment: SentimentAnalysis, 
        action_result: Dict[str, Any],
        language: str = "en"
    ) -> str:
        """
        Generate a contextual response based on sentiment and action outcome.
        
        Args:
            sentiment: SentimentAnalysis object
            action_result: Result from automation execution
            language: Target language for response
            
        Returns:
            Generated response text
        """
        import random
        
        # Get appropriate opener based on emotion
        openers = self.emotion_openers.get(sentiment.emotion, self.emotion_openers["neutral"])
        opener = random.choice(openers)
        
        # Get action feedback
        action_message = action_result.get("message", "")
        
        # Get appropriate closer based on sentiment
        closers = self.sentiment_closers.get(sentiment.sentiment, self.sentiment_closers["neutral"])
        closer = random.choice(closers)
        
        # Construct response
        if action_result.get("status") == "success":
            response = f"{opener} {action_message} {closer}"
        elif action_result.get("status") == "error":
            response = f"{opener} I encountered an issue: {action_message}"
        elif action_result.get("status") == "exit":
            response = action_message
        else:
            response = f"{opener} {action_message if action_message else closer}"
        
        return response.strip()


# ═══════════════════════════════════════════════════════════════════════════
#                    TEXT-TO-SPEECH (TTS) MODULE
# ═══════════════════════════════════════════════════════════════════════════

class TextToSpeechEngine:
    """
    Converts text to speech with multilingual support.
    Uses gTTS for online TTS and pyttsx3 for offline (if available).
    """
    
    def __init__(self, use_offline: bool = False):
        self.use_offline = use_offline and HAS_PYTTSX3
        self.counter = 0  # For unique filenames
        
        if self.use_offline:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)  # Speed
                self.engine.setProperty('volume', 0.9)  # Volume
            except Exception as e:
                print(f"⚠️  pyttsx3 initialization failed: {e}")
                self.use_offline = False
    
    def speak(self, text: str, language: str = "en", play_audio: bool = True) -> Optional[str]:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to convert to speech
            language: Language code (ISO 639-1)
            play_audio: Whether to play the audio immediately
            
        Returns:
            Path to the generated audio file (if using gTTS)
        """
        if not text:
            return None
        
        print(f"\n🔊 {ASSISTANT_NAME}: {text}\n")
        
        if self.use_offline:
            return self._speak_offline(text)
        else:
            return self._speak_online(text, language, play_audio)
    
    def _speak_online(self, text: str, language: str, play_audio: bool) -> Optional[str]:
        """Use gTTS for online text-to-speech"""
        try:
            # Generate unique filename
            self.counter += 1
            filename = os.path.join(AUDIO_OUTPUT_DIR, f"response_{self.counter}.mp3")
            
            # Create TTS
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(filename)
            
            # Play audio if requested
            if play_audio:
                self._play_audio(filename)
            
            return filename
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None
    
    def _speak_offline(self, text: str) -> None:
        """Use pyttsx3 for offline text-to-speech"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"❌ Offline TTS error: {e}")
    
    def _play_audio(self, filename: str) -> None:
        """Play audio file using system player"""
        try:
            if platform.system() == "Windows":
                os.system(f'start {filename}')
            elif platform.system() == "Darwin":  # macOS
                os.system(f'afplay {filename}')
            else:  # Linux
                os.system(f'mpg321 {filename} &')
        except Exception as e:
            print(f"⚠️  Could not play audio: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#                         MAIN ASSISTANT CLASS
# ═══════════════════════════════════════════════════════════════════════════

class VoiceAssistant:
    """
    Main orchestrator class that coordinates all modules.
    Handles the continuous listen → process → respond loop.
    """
    
    def __init__(self, use_offline_tts: bool = False):
        print(f"\n{'='*80}")
        print(f"  🤖 Initializing {ASSISTANT_NAME} - Voice Assistant with Sentiment Analysis")
        print(f"{'='*80}\n")
        
        # Initialize all modules
        print("Loading modules...")
        self.stt = SpeechToTextEngine()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.intent_recognizer = IntentRecognizer()
        self.automation = AutomationEngine()
        self.response_generator = ResponseGenerator()
        self.tts = TextToSpeechEngine(use_offline=use_offline_tts)
        
        print(f"\n✅ {ASSISTANT_NAME} is ready!\n")
    
    def process_command(self, speech_input: SpeechInput) -> AssistantResponse:
        """
        Process a single command through the complete pipeline.
        
        Args:
            speech_input: Processed speech input
            
        Returns:
            Complete assistant response
        """
        print(f"\n{'─'*80}")
        print(f"📝 User said: {speech_input.text}")
        print(f"🌐 Language: {speech_input.language}")
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze(speech_input.text)
        print(f"😊 Emotion: {sentiment.emotion} (confidence: {sentiment.emotion_score:.2f})")
        print(f"💭 Sentiment: {sentiment.sentiment} (confidence: {sentiment.sentiment_score:.2f})")
        
        # Recognize intent
        intent = self.intent_recognizer.recognize(speech_input.text)
        print(f"🎯 Intent: {intent.intent_type}")
        if intent.parameters:
            print(f"📋 Parameters: {intent.parameters}")
        
        # Execute automation
        action_result = self.automation.execute(intent)
        print(f"⚙️  Action: {action_result.get('status', 'unknown')}")
        
        # Generate response
        response_text = self.response_generator.generate(
            sentiment=sentiment,
            action_result=action_result,
            language=speech_input.language
        )
        
        # Create response object
        response = AssistantResponse(
            text=response_text,
            language=speech_input.language,
            sentiment=sentiment,
            action_result=action_result
        )
        
        return response
    
    def run(self, continuous: bool = True) -> None:
        """
        Main execution loop.
        
        Args:
            continuous: If True, runs continuously. If False, processes one command and exits.
        """
        print(f"🎤 {ASSISTANT_NAME} is listening... Say '{WAKE_WORD}' or just start speaking!")
        print("   (Say 'exit', 'quit', or 'stop' to shut down)\n")
        
        while True:
            try:
                # Listen for speech
                speech_input = self.stt.listen(timeout=5, phrase_time_limit=10)
                
                if not speech_input:
                    continue
                
                # Check for exit commands
                if any(word in speech_input.text.lower() for word in ["exit", "quit", "stop", "goodbye"]):
                    farewell = "Goodbye! Have a great day!"
                    print(f"\n🔊 {ASSISTANT_NAME}: {farewell}\n")
                    self.tts.speak(farewell, language=speech_input.language)
                    break
                
                # Process the command
                response = self.process_command(speech_input)
                
                # Speak the response
                self.tts.speak(response.text, language=response.language)
                
                print(f"{'─'*80}\n")
                
                # Exit if not in continuous mode
                if not continuous:
                    break
                
                # Check if command was to exit
                if response.action_result.get("status") == "exit":
                    break
                
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Interrupted by user")
                farewell = "Shutting down. Goodbye!"
                print(f"🔊 {ASSISTANT_NAME}: {farewell}\n")
                self.tts.speak(farewell)
                break
            
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                
                if not continuous:
                    break


# ═══════════════════════════════════════════════════════════════════════════
#                           MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for the application"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ZORA - Voice Assistant with Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project_zora.py                    # Run in continuous mode
  python project_zora.py --once             # Process one command and exit
  python project_zora.py --offline-tts      # Use offline TTS (pyttsx3)
  python project_zora.py --test "open chrome"  # Test with text input
        """
    )
    
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one command and exit"
    )
    
    parser.add_argument(
        "--offline-tts",
        action="store_true",
        help="Use offline TTS (pyttsx3) instead of gTTS"
    )
    
    parser.add_argument(
        "--test",
        type=str,
        help="Test with text input instead of speech (for debugging)"
    )
    
    args = parser.parse_args()
    
    # Create assistant instance
    assistant = VoiceAssistant(use_offline_tts=args.offline_tts)
    
    # Test mode - process text directly
    if args.test:
        test_input = SpeechInput(
            text=args.test,
            language="en",
            confidence=1.0,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        response = assistant.process_command(test_input)
        assistant.tts.speak(response.text, language=response.language)
        return
    
    # Run the assistant
    assistant.run(continuous=not args.once)


if __name__ == "__main__":
    main()
