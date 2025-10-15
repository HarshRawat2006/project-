#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project ZORA - Unified Voice Assistant with Sentiment Analysis
=============================================================

A comprehensive voice assistant that combines:
- Speech-to-Text with automatic language detection
- Sentiment analysis and emotion detection
- Natural language interpretation and intent recognition
- Automation and app control
- Text-to-Speech with language consistency
- Continuous listening and response loop

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
import subprocess
import webbrowser
import datetime
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Callable, Optional, List
from pathlib import Path

# Core dependencies
import speech_recognition as sr
import requests
from dotenv import load_dotenv

# Optional dependencies with fallbacks
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers not available - sentiment analysis disabled")

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    print("⚠️  faster-whisper not available - using speech_recognition fallback")

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False
    print("⚠️  gTTS not available - TTS disabled")

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False
    print("⚠️  pyttsx3 not available - offline TTS disabled")

try:
    from pytube import Search as YTSearch
    HAS_PYTUBE = True
except ImportError:
    HAS_PYTUBE = False
    print("⚠️  pytube not available - YouTube features limited")

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    print("⚠️  duckduckgo-search not available - web search limited")

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    HAS_SPOTIFY = True
except ImportError:
    HAS_SPOTIFY = False
    print("⚠️  spotipy not available - Spotify features disabled")

# Load environment variables
load_dotenv()

# Configuration
@dataclass
class Config:
    """Configuration settings for Project ZORA"""
    # Model configurations
    emotion_model: str = os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
    sentiment_model: str = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    
    # API configurations
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    custom_automation_webhook: str = os.getenv("CUSTOM_AUTOMATION_WEBHOOK", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Spotify configurations
    spotify_client_id: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    spotify_client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    spotify_redirect_uri: str = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
    spotify_oauth_token: str = os.getenv("SPOTIFY_OAUTH_TOKEN", "")
    spotify_device_id: str = os.getenv("SPOTIFY_DEVICE_ID", "")
    
    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 1024
    timeout: int = 5
    phrase_timeout: int = 0.3
    
    # Language settings
    default_language: str = "en"
    supported_languages: List[str] = ["en", "hi", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "project_zora.log")

@dataclass
class Analysis:
    """Analysis results for text input"""
    emotion: str
    emotion_score: float
    sentiment: str
    sentiment_score: float
    language: str
    confidence: float

class ProjectZORA:
    """
    Main Project ZORA Voice Assistant Class
    
    Integrates all features into a unified, production-ready voice assistant
    """
    
    def __init__(self, config: Config = None):
        """Initialize Project ZORA with configuration"""
        self.config = config or Config()
        self.setup_logging()
        self.setup_components()
        self.setup_intent_patterns()
        self.setup_automation_handlers()
        
        # State management
        self.is_listening = False
        self.current_language = self.config.default_language
        self.last_analysis = None
        
        self.logger.info("Project ZORA initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_components(self):
        """Initialize all components with error handling"""
        # Speech Recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Whisper model (if available)
        self.whisper_model = None
        if HAS_WHISPER:
            try:
                self.whisper_model = WhisperModel(
                    self.config.whisper_model, 
                    device="cpu", 
                    compute_type="int8"
                )
                self.logger.info("Whisper model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load Whisper model: {e}")
        
        # Sentiment Analysis (if available)
        self.emotion_pipeline = None
        self.sentiment_pipeline = None
        if HAS_TRANSFORMERS:
            try:
                self.emotion_pipeline = pipeline(
                    "text-classification", 
                    model=self.config.emotion_model, 
                    top_k=None
                )
                self.sentiment_pipeline = pipeline(
                    "text-classification", 
                    model=self.config.sentiment_model, 
                    top_k=None
                )
                self.logger.info("Sentiment analysis models loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load sentiment models: {e}")
        
        # Spotify (if available)
        self.spotify_client = None
        if HAS_SPOTIFY and self.config.spotify_oauth_token:
            try:
                self.spotify_client = spotipy.Spotify(auth=self.config.spotify_oauth_token)
                self.logger.info("Spotify client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Spotify: {e}")
    
    def setup_intent_patterns(self):
        """Setup regex patterns for intent recognition"""
        self.intent_patterns = {
            # Media and Entertainment
            "spotify_play": re.compile(r"\b(?:open\s+)?spotify\b.*?\bplay\b\s*([\"\"'']?)(.+?)\1\b", re.I),
            "youtube_play": re.compile(r"\b(?:play|open)\b.*?\b(?:on\s+)?youtube\b(?:\s*([\"\"''])(.+?)\1|\s+(.+))?$", re.I),
            "youtube_quick": re.compile(r"\b(?:play|youtube)\b\s*([\"\"'']?)(.+?)\1\b", re.I),
            "play_music": re.compile(r"\bplay\s+(?:music|song|track)\b", re.I),
            
            # Search and Information
            "google_search": re.compile(r"\b(?:google|search\s+google(?:\s+for)?)\b(?:\s*([\"\"''])(.+?)\1|\s+(.+))?$", re.I),
            "web_search": re.compile(r"\b(?:search|look up|find)\b", re.I),
            "wikipedia": re.compile(r"\b(?:wiki|wikipedia)\b(.*)$", re.I),
            "define": re.compile(r"\b(?:define|meaning of)\b(.*)$", re.I),
            "maps": re.compile(r"\b(?:map|maps|direction|navigate)\b(.*)$", re.I),
            
            # Application Control
            "open_app": re.compile(r"\b(open|launch)\s+(notepad|calculator|vscode|spotify|word|youtube|chrome|edge|cmd|command\s+window|powershell|terminal)\b", re.I),
            "open_website": re.compile(r"\b(open|launch)\s+((?:https?://)?[\w.-]+\.[a-z]{2,}\S*)\b", re.I),
            "open_path": re.compile(r"\b(open|launch)\s+([a-zA-Z]:\\[^:<>\"|?*\n]+)\b", re.I),
            "open_special": re.compile(r"\b(open|launch)\s+(my\s*pc|this\s*pc|videos?|photos?|pictures?|documents?|downloads?|desktop|music)\b", re.I),
            
            # Communication
            "email": re.compile(r"\bemail\b(.*)$", re.I),
            "send_slack": re.compile(r"\b(?:slack|notify team|tell the team)\b", re.I),
            "create_note": re.compile(r"\b(?:note|remember|jot|write down|save this)\b", re.I),
            
            # System Control
            "system_command": re.compile(r"\bsystem\b(.*)$", re.I),
            "run_webhook": re.compile(r"\b(?:trigger|webhook|zapier|ifttt|make scenario)\b", re.I),
            "time": re.compile(r"\b(?:time|what time|current time)\b", re.I),
            "date": re.compile(r"\b(?:date|what date|current date)\b", re.I),
            
            # Control Commands
            "stop": re.compile(r"\b(?:stop|exit|quit|shutdown|turn off)\b", re.I),
            "help": re.compile(r"\b(?:help|what can you do|commands)\b", re.I),
        }
    
    def setup_automation_handlers(self):
        """Setup automation handlers for each intent"""
        self.automation_handlers = {
            "spotify_play": self._handle_spotify_play,
            "youtube_play": self._handle_youtube_play,
            "youtube_quick": self._handle_youtube_quick,
            "play_music": self._handle_play_music,
            "google_search": self._handle_google_search,
            "web_search": self._handle_web_search,
            "wikipedia": self._handle_wikipedia,
            "define": self._handle_define,
            "maps": self._handle_maps,
            "open_app": self._handle_open_app,
            "open_website": self._handle_open_website,
            "open_path": self._handle_open_path,
            "open_special": self._handle_open_special,
            "email": self._handle_email,
            "send_slack": self._handle_send_slack,
            "create_note": self._handle_create_note,
            "system_command": self._handle_system_command,
            "run_webhook": self._handle_run_webhook,
            "time": self._handle_time,
            "date": self._handle_date,
            "stop": self._handle_stop,
            "help": self._handle_help,
        }
    
    # ==================== SPEECH-TO-TEXT ====================
    
    def listen_speech(self, timeout: int = None) -> Tuple[str, str, float]:
        """
        Listen to user speech and convert to text
        
        Returns:
            Tuple of (text, language, confidence)
        """
        timeout = timeout or self.config.timeout
        
        try:
            with self.microphone as source:
                self.logger.info("Listening for speech...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=self.config.phrase_timeout)
            
            # Try Whisper first (if available)
            if self.whisper_model:
                try:
                    segments, info = self.whisper_model.transcribe(
                        audio.get_wav_data(),
                        language=None,  # Auto-detect language
                        beam_size=5
                    )
                    text = " ".join([segment.text for segment in segments])
                    language = info.language if hasattr(info, 'language') else self.config.default_language
                    confidence = 0.9  # Whisper doesn't provide confidence scores
                    self.logger.info(f"Whisper transcription: {text}")
                    return text.strip(), language, confidence
                except Exception as e:
                    self.logger.warning(f"Whisper failed: {e}, falling back to speech_recognition")
            
            # Fallback to speech_recognition
            text = self.recognizer.recognize_google(audio, language="en")
            self.logger.info(f"Speech recognition transcription: {text}")
            return text.strip(), self.config.default_language, 0.8
            
        except sr.WaitTimeoutError:
            self.logger.warning("No speech detected within timeout")
            return "", self.config.default_language, 0.0
        except sr.UnknownValueError:
            self.logger.warning("Could not understand audio")
            return "", self.config.default_language, 0.0
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition service error: {e}")
            return "", self.config.default_language, 0.0
        except Exception as e:
            self.logger.error(f"Unexpected error in speech recognition: {e}")
            return "", self.config.default_language, 0.0
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text
        Simple implementation - can be enhanced with language detection libraries
        """
        # Simple keyword-based language detection
        hindi_keywords = ['है', 'हैं', 'मैं', 'तुम', 'आप', 'क्या', 'कैसे', 'कहाँ', 'कब']
        if any(keyword in text for keyword in hindi_keywords):
            return "hi"
        
        # Default to English
        return "en"
    
    # ==================== SENTIMENT ANALYSIS ====================
    
    def analyze_sentiment(self, text: str) -> Analysis:
        """
        Analyze sentiment and emotion of the input text
        
        Returns:
            Analysis object with emotion, sentiment, and language information
        """
        if not text.strip():
            return Analysis("neutral", 0.0, "neutral", 0.0, self.current_language, 0.0)
        
        # Detect language
        language = self.detect_language(text)
        self.current_language = language
        
        # Get emotion and sentiment
        emotion, emotion_score = self._predict_emotion(text)
        sentiment, sentiment_score = self._predict_sentiment(text)
        
        analysis = Analysis(
            emotion=emotion,
            emotion_score=emotion_score,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            language=language,
            confidence=min(emotion_score, sentiment_score)
        )
        
        self.last_analysis = analysis
        self.logger.info(f"Analysis: {emotion} ({emotion_score:.3f}), {sentiment} ({sentiment_score:.3f}), lang: {language}")
        
        return analysis
    
    def _predict_emotion(self, text: str) -> Tuple[str, float]:
        """Predict emotion using the emotion model"""
        if not self.emotion_pipeline:
            # Fallback emotion detection
            return self._fallback_emotion_detection(text)
        
        try:
            preds = self.emotion_pipeline(text)[0]
            best = max(preds, key=lambda x: x["score"])
            return best["label"].lower(), float(best["score"])
        except Exception as e:
            self.logger.warning(f"Emotion prediction failed: {e}")
            return self._fallback_emotion_detection(text)
    
    def _predict_sentiment(self, text: str) -> Tuple[str, float]:
        """Predict sentiment using the sentiment model"""
        if not self.sentiment_pipeline:
            # Fallback sentiment detection
            return self._fallback_sentiment_detection(text)
        
        try:
            preds = self.sentiment_pipeline(text)[0]
            best = max(preds, key=lambda x: x["score"])
            label = str(best.get("label", "")).lower()
            
            # Map labels to standard sentiment
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
            self.logger.warning(f"Sentiment prediction failed: {e}")
            return self._fallback_sentiment_detection(text)
    
    def _fallback_emotion_detection(self, text: str) -> Tuple[str, float]:
        """Fallback emotion detection using keyword matching"""
        text_lower = text.lower()
        
        emotion_keywords = {
            "joy": ["happy", "excited", "great", "wonderful", "amazing", "fantastic", "love", "like"],
            "sadness": ["sad", "depressed", "unhappy", "miserable", "terrible", "awful", "hate", "dislike"],
            "anger": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated", "rage"],
            "fear": ["scared", "afraid", "worried", "anxious", "nervous", "terrified", "panic"],
            "surprise": ["surprised", "shocked", "amazed", "wow", "incredible", "unbelievable"],
            "disgust": ["disgusted", "sick", "gross", "revolting", "nasty", "horrible"]
        }
        
        scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
            scores[emotion] = score
        
        if not any(scores.values()):
            return "neutral", 0.5
        
        best_emotion = max(scores, key=scores.get)
        return best_emotion, scores[best_emotion]
    
    def _fallback_sentiment_detection(self, text: str) -> Tuple[str, float]:
        """Fallback sentiment detection using keyword matching"""
        text_lower = text.lower()
        
        positive_keywords = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like", "happy", "pleased"]
        negative_keywords = ["bad", "terrible", "awful", "horrible", "hate", "dislike", "sad", "angry", "frustrated", "disappointed"]
        
        positive_score = sum(1 for word in positive_keywords if word in text_lower)
        negative_score = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_score > negative_score:
            return "positive", min(0.9, 0.5 + positive_score * 0.1)
        elif negative_score > positive_score:
            return "negative", min(0.9, 0.5 + negative_score * 0.1)
        else:
            return "neutral", 0.5
    
    # ==================== NATURAL LANGUAGE INTERPRETATION ====================
    
    def interpret_command(self, text: str) -> Optional[str]:
        """
        Parse transcribed text to detect intent
        
        Returns:
            Intent string or None if no intent detected
        """
        if not text.strip():
            return None
        
        text_lower = text.lower()
        
        for intent, pattern in self.intent_patterns.items():
            if pattern.search(text_lower):
                self.logger.info(f"Detected intent: {intent}")
                return intent
        
        self.logger.info("No intent detected")
        return None
    
    def craft_response(self, text: str, analysis: Analysis, intent: str = None) -> str:
        """
        Generate contextual and emotional response
        
        Returns:
            Response text in the same language as input
        """
        emotion = analysis.emotion
        sentiment = analysis.sentiment
        language = analysis.language
        
        # Response templates based on emotion and sentiment
        if language == "hi":
            responses = {
                "joy": "बहुत अच्छा! मैं आपकी मदद करूंगा।",
                "sadness": "मुझे खेद है कि आप परेशान हैं। मैं यहाँ हूँ मदद के लिए।",
                "anger": "मैं समझता हूँ कि आप नाराज़ हैं। चलिए इसे ठीक करते हैं।",
                "fear": "चिंता न करें, मैं आपकी मदद करूंगा।",
                "surprise": "वाह! यह तो अच्छा है।",
                "neutral": "ठीक है, मैं समझ गया।"
            }
        else:  # English and other languages
            responses = {
                "joy": "That's wonderful! I'm here to help you.",
                "sadness": "I'm sorry you're feeling down. Let me help you with that.",
                "anger": "I understand you're frustrated. Let's work through this together.",
                "fear": "Don't worry, I'm here to help you.",
                "surprise": "Wow! That's exciting.",
                "neutral": "Alright, I understand."
            }
        
        base_response = responses.get(emotion, responses["neutral"])
        
        # Add intent-specific responses
        if intent:
            if language == "hi":
                intent_responses = {
                    "spotify_play": "मैं Spotify पर संगीत चलाता हूँ।",
                    "youtube_play": "मैं YouTube पर वीडियो खोलता हूँ।",
                    "google_search": "मैं Google पर खोज करता हूँ।",
                    "open_app": "मैं एप्लिकेशन खोलता हूँ।",
                    "time": "मैं समय बताता हूँ।",
                    "help": "मैं आपकी मदद कर सकता हूँ।"
                }
            else:
                intent_responses = {
                    "spotify_play": "I'll play music on Spotify for you.",
                    "youtube_play": "I'll open YouTube and play the video.",
                    "google_search": "I'll search on Google for you.",
                    "open_app": "I'll open the application for you.",
                    "time": "I'll tell you the current time.",
                    "help": "I can help you with various tasks."
                }
            
            intent_response = intent_responses.get(intent, "")
            if intent_response:
                base_response += f" {intent_response}"
        
        return base_response
    
    # ==================== AUTOMATION & APP CONTROL ====================
    
    def execute_action(self, intent: str, text: str, analysis: Analysis) -> Dict[str, Any]:
        """
        Execute the appropriate action based on intent
        
        Returns:
            Dictionary with action result information
        """
        if intent not in self.automation_handlers:
            return {"status": "error", "reason": "Unknown intent"}
        
        try:
            result = self.automation_handlers[intent](text, analysis)
            self.logger.info(f"Action executed: {intent} - {result}")
            return result
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _handle_spotify_play(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle Spotify play requests"""
        match = self.intent_patterns["spotify_play"].search(text)
        song = self._extract_query(match.group(2) if match else "")
        
        if not song:
            return {"status": "error", "reason": "No song specified"}
        
        if self.spotify_client:
            try:
                results = self.spotify_client.search(q=song, type="track", limit=1)
                tracks = results.get("tracks", {}).get("items", [])
                if tracks:
                    track_uri = tracks[0]["uri"]
                    self.spotify_client.start_playback(device_id=self.config.spotify_device_id or None, uris=[track_uri])
                    return {"status": "success", "action": "spotify_play", "track": tracks[0]["name"]}
            except Exception as e:
                self.logger.warning(f"Spotify API failed: {e}")
        
        # Fallback to opening Spotify web player
        url = f"https://open.spotify.com/search/{requests.utils.quote(song)}"
        webbrowser.open(url)
        return {"status": "success", "action": "spotify_web", "url": url}
    
    def _handle_youtube_play(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle YouTube play requests"""
        match = self.intent_patterns["youtube_play"].search(text)
        query = self._extract_query((match.group(2) or match.group(3) or "") if match else "")
        
        if not query:
            url = "https://www.youtube.com"
            webbrowser.open(url)
            return {"status": "success", "action": "youtube_open", "url": url}
        
        # Try to find and play video directly
        if HAS_PYTUBE:
            try:
                search = YTSearch(query)
                results = search.results[:1]
                if results:
                    video_url = f"https://www.youtube.com/watch?v={results[0].video_id}&autoplay=1"
                    webbrowser.open(video_url)
                    return {"status": "success", "action": "youtube_play", "video": results[0].title, "url": video_url}
            except Exception as e:
                self.logger.warning(f"YouTube search failed: {e}")
        
        # Fallback to search results
        url = f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"
        webbrowser.open(url)
        return {"status": "success", "action": "youtube_search", "query": query, "url": url}
    
    def _handle_youtube_quick(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle quick YouTube requests"""
        return self._handle_youtube_play(text, analysis)
    
    def _handle_play_music(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle general music play requests"""
        # Try Spotify first, then YouTube
        spotify_result = self._handle_spotify_play(text, analysis)
        if spotify_result.get("status") == "success":
            return spotify_result
        
        return self._handle_youtube_play(text, analysis)
    
    def _handle_google_search(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle Google search requests"""
        match = self.intent_patterns["google_search"].search(text)
        query = self._extract_query((match.group(2) or match.group(3) or "") if match else text)
        
        # Try DuckDuckGo first for better results
        if HAS_DDGS:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=1))
                if results and results[0].get("href"):
                    webbrowser.open(results[0]["href"])
                    return {"status": "success", "action": "web_search", "query": query, "url": results[0]["href"]}
            except Exception as e:
                self.logger.warning(f"DuckDuckGo search failed: {e}")
        
        # Fallback to Google
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        webbrowser.open(url)
        return {"status": "success", "action": "google_search", "query": query, "url": url}
    
    def _handle_web_search(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle general web search requests"""
        return self._handle_google_search(text, analysis)
    
    def _handle_wikipedia(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle Wikipedia search requests"""
        match = self.intent_patterns["wikipedia"].search(text)
        query = self._extract_query(match.group(2) if match else "")
        
        if not query:
            url = "https://en.wikipedia.org"
        else:
            url = f"https://en.wikipedia.org/w/index.php?search={requests.utils.quote(query)}"
        
        webbrowser.open(url)
        return {"status": "success", "action": "wikipedia", "query": query, "url": url}
    
    def _handle_define(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle dictionary definition requests"""
        match = self.intent_patterns["define"].search(text)
        query = self._extract_query(match.group(2) if match else "")
        
        if not query:
            return {"status": "error", "reason": "No term to define"}
        
        url = f"https://www.dictionary.com/browse/{requests.utils.quote(query)}"
        webbrowser.open(url)
        return {"status": "success", "action": "define", "term": query, "url": url}
    
    def _handle_maps(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle maps/navigation requests"""
        match = self.intent_patterns["maps"].search(text)
        query = self._extract_query(match.group(2) if match else "")
        
        if not query:
            return {"status": "error", "reason": "No location specified"}
        
        url = f"https://www.google.com/maps/search/{requests.utils.quote(query)}"
        webbrowser.open(url)
        return {"status": "success", "action": "maps", "query": query, "url": url}
    
    def _handle_open_app(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle application opening requests"""
        match = self.intent_patterns["open_app"].search(text)
        app = (match.group(2).lower() if match else "notepad").strip()
        
        app_commands = {
            "notepad": "notepad",
            "calculator": "calc",
            "cmd": "cmd",
            "command window": "cmd",
            "powershell": "powershell",
            "terminal": "powershell",
            "chrome": "chrome",
            "edge": "msedge",
            "youtube": "https://www.youtube.com",
            "spotify": "spotify",
            "vscode": "code",
            "word": "winword"
        }
        
        if app in app_commands:
            command = app_commands[app]
            if command.startswith("http"):
                webbrowser.open(command)
                return {"status": "success", "action": "open_website", "app": app, "url": command}
            else:
                subprocess.Popen(command, shell=True)
                return {"status": "success", "action": "open_app", "app": app, "command": command}
        
        return {"status": "error", "reason": f"Unknown application: {app}"}
    
    def _handle_open_website(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle website opening requests"""
        match = self.intent_patterns["open_website"].search(text)
        url = match.group(2) if match else ""
        
        if not url:
            return {"status": "error", "reason": "No website specified"}
        
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        
        webbrowser.open(url)
        return {"status": "success", "action": "open_website", "url": url}
    
    def _handle_open_path(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle file path opening requests"""
        match = self.intent_patterns["open_path"].search(text)
        path = match.group(2) if match else ""
        
        if not path or not os.path.exists(path):
            return {"status": "error", "reason": f"Path not found: {path}"}
        
        if os.name == 'nt':  # Windows
            subprocess.Popen(f'explorer.exe "{path}"', shell=True)
        else:  # Unix-like
            subprocess.Popen(['xdg-open', path])
        
        return {"status": "success", "action": "open_path", "path": path}
    
    def _handle_open_special(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle special folder opening requests"""
        match = self.intent_patterns["open_special"].search(text)
        folder = (match.group(2).lower().replace(" ", "") if match else "")
        
        special_folders = {
            "mypc": "shell:MyComputerFolder",
            "thispc": "shell:MyComputerFolder",
            "videos": os.path.join(os.path.expanduser("~"), "Videos"),
            "photos": os.path.join(os.path.expanduser("~"), "Pictures"),
            "pictures": os.path.join(os.path.expanduser("~"), "Pictures"),
            "documents": os.path.join(os.path.expanduser("~"), "Documents"),
            "downloads": os.path.join(os.path.expanduser("~"), "Downloads"),
            "desktop": os.path.join(os.path.expanduser("~"), "Desktop"),
            "music": os.path.join(os.path.expanduser("~"), "Music"),
        }
        
        if folder not in special_folders:
            return {"status": "error", "reason": f"Unknown special folder: {folder}"}
        
        path = special_folders[folder]
        
        if os.name == 'nt':  # Windows
            if path.startswith("shell:"):
                subprocess.Popen(f'explorer.exe {path}', shell=True)
            else:
                subprocess.Popen(f'explorer.exe "{path}"', shell=True)
        else:  # Unix-like
            subprocess.Popen(['xdg-open', path])
        
        return {"status": "success", "action": "open_special", "folder": folder, "path": path}
    
    def _handle_email(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle email composition requests"""
        # Extract email details from text
        to_match = re.search(r"\bto\s+(\S+)", text, re.I)
        subject_match = re.search(r"\bsubject\s+(.+?)(?=\s+body\b|$)", text, re.I)
        body_match = re.search(r"\bbody\s+(.+)$", text, re.I)
        
        to = to_match.group(1) if to_match else ""
        subject = subject_match.group(1).strip() if subject_match else ""
        body = body_match.group(1).strip() if body_match else ""
        
        mailto = f"mailto:{to}?subject={requests.utils.quote(subject)}&body={requests.utils.quote(body)}"
        webbrowser.open(mailto)
        
        return {"status": "success", "action": "email", "to": to, "subject": subject, "body": body}
    
    def _handle_send_slack(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle Slack notification requests"""
        if not self.config.slack_webhook_url:
            return {"status": "error", "reason": "Slack webhook not configured"}
        
        payload = {
            "text": f"[{analysis.emotion}/{analysis.sentiment}] {text}"
        }
        
        try:
            response = requests.post(
                self.config.slack_webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            return {"status": "success" if response.ok else "error", "action": "slack", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "action": "slack", "error": str(e)}
    
    def _handle_create_note(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle note creation requests"""
        # Extract note content
        note_match = re.search(r"(?:note|remember|jot|write down|save this)\b(.*)$", text, re.I)
        note_content = note_match.group(1).strip() if note_match and note_match.group(1).strip() else text
        
        # Create notes directory
        notes_dir = Path("notes")
        notes_dir.mkdir(exist_ok=True)
        
        # Create note file
        timestamp = int(time.time())
        note_file = notes_dir / f"note_{timestamp}.txt"
        
        with open(note_file, "w", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}]\n")
            f.write(f"[emotion={analysis.emotion}, sentiment={analysis.sentiment}]\n")
            f.write(f"{note_content}\n")
        
        return {"status": "success", "action": "create_note", "file": str(note_file), "content": note_content}
    
    def _handle_system_command(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle system command execution"""
        match = re.search(r"\bsystem\b(.*)$", text, re.I)
        command = match.group(1).strip() if match else ""
        
        if not command:
            return {"status": "error", "reason": "No command specified"}
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            return {
                "status": "success",
                "action": "system_command",
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "action": "system_command", "error": "Command timed out"}
        except Exception as e:
            return {"status": "error", "action": "system_command", "error": str(e)}
    
    def _handle_run_webhook(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle custom webhook execution"""
        if not self.config.custom_automation_webhook:
            return {"status": "error", "reason": "Custom webhook not configured"}
        
        payload = {
            "text": text,
            "emotion": analysis.emotion,
            "sentiment": analysis.sentiment,
            "language": analysis.language
        }
        
        try:
            response = requests.post(self.config.custom_automation_webhook, json=payload, timeout=10)
            return {"status": "success" if response.ok else "error", "action": "webhook", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "action": "webhook", "error": str(e)}
    
    def _handle_time(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle time requests"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return {"status": "success", "action": "time", "time": current_time}
    
    def _handle_date(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle date requests"""
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        return {"status": "success", "action": "date", "date": current_date}
    
    def _handle_stop(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle stop/exit requests"""
        return {"status": "success", "action": "stop", "message": "Goodbye!"}
    
    def _handle_help(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        """Handle help requests"""
        help_text = """
        I can help you with:
        - Playing music on Spotify or YouTube
        - Searching the web or Wikipedia
        - Opening applications and websites
        - Creating notes and reminders
        - Getting current time and date
        - And much more!
        
        Just speak naturally and I'll understand your intent.
        """
        return {"status": "success", "action": "help", "message": help_text}
    
    def _extract_query(self, text: str) -> str:
        """Extract and clean query from text"""
        if not text:
            return ""
        return text.strip().strip('"').strip(""").strip("'")
    
    # ==================== TEXT-TO-SPEECH ====================
    
    def speak_response(self, text: str, language: str = None) -> bool:
        """
        Convert text to speech and play it
        
        Returns:
            True if successful, False otherwise
        """
        if not text.strip():
            return False
        
        language = language or self.current_language
        
        # Try gTTS first (online)
        if HAS_GTTS:
            try:
                tts = gTTS(text=text, lang=language, slow=False)
                audio_file = "temp_speech.mp3"
                tts.save(audio_file)
                
                # Play audio
                if os.name == 'nt':  # Windows
                    os.system(f'start {audio_file}')
                elif os.name == 'posix':  # Unix-like
                    os.system(f'mpg321 {audio_file} 2>/dev/null || mplayer {audio_file} 2>/dev/null || play {audio_file} 2>/dev/null')
                
                # Clean up
                time.sleep(1)
                try:
                    os.remove(audio_file)
                except:
                    pass
                
                self.logger.info(f"TTS (gTTS): {text[:50]}...")
                return True
            except Exception as e:
                self.logger.warning(f"gTTS failed: {e}")
        
        # Fallback to pyttsx3 (offline)
        if HAS_PYTTSX3:
            try:
                engine = pyttsx3.init()
                
                # Set voice based on language
                voices = engine.getProperty('voices')
                for voice in voices:
                    if language in voice.id.lower() or language in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                
                engine.say(text)
                engine.runAndWait()
                
                self.logger.info(f"TTS (pyttsx3): {text[:50]}...")
                return True
            except Exception as e:
                self.logger.warning(f"pyttsx3 failed: {e}")
        
        # Fallback to system TTS
        try:
            if os.name == 'nt':  # Windows
                os.system(f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"')
            elif os.name == 'posix':  # Unix-like
                os.system(f'espeak "{text}" 2>/dev/null')
            
            self.logger.info(f"TTS (system): {text[:50]}...")
            return True
        except Exception as e:
            self.logger.error(f"All TTS methods failed: {e}")
            return False
    
    # ==================== MAIN PROCESSING LOOP ====================
    
    def process_input(self, text: str, execute_actions: bool = True) -> Dict[str, Any]:
        """
        Main processing function that handles the complete pipeline
        
        Returns:
            Dictionary with processing results
        """
        if not text.strip():
            return {"status": "error", "reason": "Empty input"}
        
        # Analyze sentiment and emotion
        analysis = self.analyze_sentiment(text)
        
        # Detect intent
        intent = self.interpret_command(text)
        
        # Generate response
        response = self.craft_response(text, analysis, intent)
        
        # Execute action if requested
        action_result = {"status": "skipped", "reason": "no intent detected"}
        if intent and execute_actions:
            action_result = self.execute_action(intent, text, analysis)
        
        return {
            "status": "success",
            "text": text,
            "analysis": analysis.__dict__,
            "intent": intent,
            "response": response,
            "action_result": action_result
        }
    
    def run_continuous(self, execute_actions: bool = True):
        """
        Run the continuous listening and processing loop
        """
        self.logger.info("Starting Project ZORA continuous mode...")
        self.logger.info("Say 'stop', 'exit', or 'quit' to end the session")
        
        self.is_listening = True
        
        try:
            while self.is_listening:
                try:
                    # Listen for speech
                    text, language, confidence = self.listen_speech()
                    
                    if not text:
                        continue
                    
                    # Process the input
                    result = self.process_input(text, execute_actions)
                    
                    # Display results
                    print(f"\n🎤 You said: {text}")
                    print(f"🌍 Language: {result['analysis']['language']}")
                    print(f"😊 Emotion: {result['analysis']['emotion']} ({result['analysis']['emotion_score']:.3f})")
                    print(f"💭 Sentiment: {result['analysis']['sentiment']} ({result['analysis']['sentiment_score']:.3f})")
                    print(f"🎯 Intent: {result['intent']}")
                    print(f"🤖 Response: {result['response']}")
                    
                    if result['action_result']['status'] != 'skipped':
                        print(f"⚡ Action: {result['action_result']}")
                    
                    # Speak the response
                    self.speak_response(result['response'], result['analysis']['language'])
                    
                    # Check for stop commands
                    if result['intent'] == 'stop':
                        self.is_listening = False
                        print("👋 Goodbye!")
                        break
                    
                except KeyboardInterrupt:
                    print("\n👋 Interrupted by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    print(f"❌ Error: {e}")
                    continue
                    
        finally:
            self.is_listening = False
            self.logger.info("Project ZORA stopped")
    
    def run_single(self, text: str, execute_actions: bool = True) -> Dict[str, Any]:
        """
        Process a single text input
        
        Returns:
            Processing results
        """
        return self.process_input(text, execute_actions)


def main():
    """Main entry point for Project ZORA"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Project ZORA - Unified Voice Assistant")
    parser.add_argument("--text", "-t", type=str, help="Single text input to process")
    parser.add_argument("--no-actions", action="store_true", help="Disable action execution")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    if args.log_level:
        config.log_level = args.log_level
    
    # Initialize Project ZORA
    zora = ProjectZORA(config)
    
    if args.text:
        # Single text processing
        result = zora.run_single(args.text, execute_actions=not args.no_actions)
        
        print(f"Input: {result['text']}")
        print(f"Language: {result['analysis']['language']}")
        print(f"Emotion: {result['analysis']['emotion']} ({result['analysis']['emotion_score']:.3f})")
        print(f"Sentiment: {result['analysis']['sentiment']} ({result['analysis']['sentiment_score']:.3f})")
        print(f"Intent: {result['intent']}")
        print(f"Response: {result['response']}")
        
        if result['action_result']['status'] != 'skipped':
            print(f"Action: {result['action_result']}")
    else:
        # Continuous mode
        zora.run_continuous(execute_actions=not args.no_actions)


if __name__ == "__main__":
    main()