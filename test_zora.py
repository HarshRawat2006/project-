#!/usr/bin/env python3
"""
Project ZORA - Installation and Functionality Test Script
=========================================================

This script tests the core components of Project ZORA to ensure
everything is working correctly.
"""

import sys
import os
import importlib
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    required_modules = [
        ('speech_recognition', 'SpeechRecognition'),
        ('langdetect', 'langdetect'),
        ('requests', 'requests'),
        ('dotenv', 'python-dotenv')
    ]
    
    optional_modules = [
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('textblob', 'textblob'),
        ('gtts', 'gtts'),
        ('pyttsx3', 'pyttsx3'),
        ('pytube', 'pytube'),
        ('duckduckgo_search', 'duckduckgo-search'),
        ('spotipy', 'spotipy')
    ]
    
    results = {'required': [], 'optional': []}
    
    # Test required modules
    for module, package in required_modules:
        try:
            importlib.import_module(module)
            results['required'].append((module, package, True, None))
            print(f"  ✅ {module}")
        except ImportError as e:
            results['required'].append((module, package, False, str(e)))
            print(f"  ❌ {module} - {e}")
    
    # Test optional modules
    for module, package in optional_modules:
        try:
            importlib.import_module(module)
            results['optional'].append((module, package, True, None))
            print(f"  ✅ {module} (optional)")
        except ImportError as e:
            results['optional'].append((module, package, False, str(e)))
            print(f"  ⚠️  {module} (optional) - {e}")
    
    return results

def test_microphone():
    """Test microphone access"""
    print("\n🎤 Testing microphone access...")
    
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        print("  📊 Available microphones:")
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"    {i}: {name}")
        
        print("  🔧 Testing microphone initialization...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("  ✅ Microphone access successful")
        return True
        
    except Exception as e:
        print(f"  ❌ Microphone test failed: {e}")
        return False

def test_tts():
    """Test text-to-speech functionality"""
    print("\n🔊 Testing text-to-speech...")
    
    # Test pyttsx3 (offline)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        print("  ✅ pyttsx3 (offline TTS) available")
        
        # Test voice synthesis (without actually playing)
        engine.say("Test")
        print("  ✅ pyttsx3 voice synthesis test passed")
        
    except Exception as e:
        print(f"  ⚠️  pyttsx3 test failed: {e}")
    
    # Test gTTS (online)
    try:
        from gtts import gTTS
        
        # Create a test TTS object (don't save)
        tts = gTTS(text="Test", lang='en')
        print("  ✅ gTTS (online TTS) available")
        
    except Exception as e:
        print(f"  ⚠️  gTTS test failed: {e}")

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("\n🧠 Testing sentiment analysis...")
    
    test_text = "I'm really happy and excited about this new project!"
    
    # Test transformers
    try:
        from transformers import pipeline
        
        print("  📥 Loading sentiment model (this may take a moment)...")
        sentiment_pipeline = pipeline("sentiment-analysis")
        
        result = sentiment_pipeline(test_text)
        print(f"  ✅ Transformers sentiment: {result}")
        
    except Exception as e:
        print(f"  ⚠️  Transformers test failed: {e}")
    
    # Test TextBlob fallback
    try:
        from textblob import TextBlob
        
        blob = TextBlob(test_text)
        sentiment = blob.sentiment
        print(f"  ✅ TextBlob sentiment: polarity={sentiment.polarity:.2f}, subjectivity={sentiment.subjectivity:.2f}")
        
    except Exception as e:
        print(f"  ⚠️  TextBlob test failed: {e}")

def test_zora_components():
    """Test Project ZORA components"""
    print("\n🤖 Testing Project ZORA components...")
    
    try:
        # Import the main module
        from project_zora_unified import (
            LanguageDetector, SentimentAnalyzer, IntentClassifier,
            AutomationEngine, TextToSpeech, ResponseGenerator
        )
        
        # Test language detection
        detector = LanguageDetector()
        lang = detector.detect_language("Hello, how are you?")
        print(f"  ✅ Language detection: '{lang}'")
        
        # Test sentiment analyzer
        analyzer = SentimentAnalyzer()
        analysis = analyzer.analyze("I'm feeling great today!")
        print(f"  ✅ Sentiment analysis: {analysis.emotion}/{analysis.sentiment}")
        
        # Test intent classifier
        classifier = IntentClassifier()
        intent, params = classifier.classify_intent("play music on spotify")
        print(f"  ✅ Intent classification: {intent} with params {params}")
        
        # Test automation engine
        engine = AutomationEngine()
        print("  ✅ Automation engine initialized")
        
        # Test TTS
        tts = TextToSpeech()
        print("  ✅ Text-to-speech initialized")
        
        # Test response generator
        generator = ResponseGenerator()
        response = generator.get_wake_response()
        print(f"  ✅ Response generator: '{response}'")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ZORA components test failed: {e}")
        return False

def generate_report(import_results, mic_success, zora_success):
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("📋 PROJECT ZORA INSTALLATION REPORT")
    print("="*60)
    
    # Required modules status
    print("\n🔧 REQUIRED MODULES:")
    required_ok = True
    for module, package, success, error in import_results['required']:
        status = "✅ OK" if success else "❌ MISSING"
        print(f"  {module:<20} {status}")
        if not success:
            required_ok = False
            print(f"    Install with: pip install {package}")
    
    # Optional modules status
    print("\n⚡ OPTIONAL MODULES:")
    optional_count = 0
    for module, package, success, error in import_results['optional']:
        status = "✅ OK" if success else "⚠️  Missing"
        print(f"  {module:<20} {status}")
        if success:
            optional_count += 1
        elif not success:
            print(f"    Install with: pip install {package}")
    
    # Hardware status
    print(f"\n🎤 MICROPHONE ACCESS:     {'✅ OK' if mic_success else '❌ FAILED'}")
    
    # Core functionality
    print(f"🤖 ZORA COMPONENTS:      {'✅ OK' if zora_success else '❌ FAILED'}")
    
    # Overall assessment
    print(f"\n📊 SUMMARY:")
    print(f"  Required modules:      {'✅ All OK' if required_ok else '❌ Missing dependencies'}")
    print(f"  Optional modules:      {optional_count}/8 available")
    print(f"  Hardware access:       {'✅ Ready' if mic_success else '❌ Check microphone'}")
    print(f"  Core functionality:    {'✅ Ready' if zora_success else '❌ Check installation'}")
    
    if required_ok and mic_success and zora_success:
        print(f"\n🎉 PROJECT ZORA IS READY TO USE!")
        print(f"   Run: python project_zora_unified.py")
    else:
        print(f"\n⚠️  SETUP INCOMPLETE - Please fix the issues above")
        
        if not required_ok:
            print(f"   Install missing modules: pip install -r requirements.txt")
        if not mic_success:
            print(f"   Check microphone permissions and hardware")
        if not zora_success:
            print(f"   Verify project_zora_unified.py is in the current directory")

def main():
    """Run all tests and generate report"""
    print("🚀 Project ZORA Installation Test")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python Version: {sys.version}")
    print(f"📂 Working Directory: {os.getcwd()}")
    
    # Run tests
    import_results = test_imports()
    mic_success = test_microphone()
    
    # Test TTS (informational only)
    test_tts()
    
    # Test sentiment analysis (informational only)
    test_sentiment_analysis()
    
    # Test ZORA components
    zora_success = test_zora_components()
    
    # Generate final report
    generate_report(import_results, mic_success, zora_success)

if __name__ == "__main__":
    main()