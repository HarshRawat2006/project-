# Project ZORA - Unified Voice Assistant 🤖

A comprehensive, production-ready voice assistant that combines speech-to-text, sentiment analysis, natural language interpretation, automation, and text-to-speech capabilities.

## ✨ Features

### 🎤 Speech-to-Text (STT)
- Continuous speech recognition using Google Speech API
- Automatic language detection (supports 12+ languages)
- Wake word activation ("zora" by default)
- Noise cancellation and ambient adjustment

### 🧠 Sentiment & Emotion Analysis
- Real-time emotion detection (joy, sadness, anger, fear, surprise, neutral)
- Sentiment analysis (positive, negative, neutral)
- Uses advanced transformer models (HuggingFace) with TextBlob fallback
- Contextual response adjustment based on user emotion

### 🎯 Natural Language Interpretation
- Intent classification with regex patterns
- Parameter extraction from natural speech
- Support for complex commands and queries
- Handles ambiguous requests gracefully

### 🔧 System Automation & App Control
- Open applications (Spotify, Chrome, VS Code, etc.)
- File and folder operations
- Web searches and website navigation
- YouTube and Spotify integration
- Note creation with emotional context
- System information queries

### 🗣️ Text-to-Speech (TTS)
- Multilingual speech synthesis
- Emotional tone adjustment
- Online (gTTS) and offline (pyttsx3) support
- Automatic language matching

### 🔄 Continuous Operation
- Threaded architecture for responsiveness
- Queue-based command processing
- Session statistics and logging
- Graceful error handling and recovery

## 🚀 Quick Start

### Prerequisites

**System Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev espeak espeak-data libespeak1 libespeak-dev

# macOS
brew install portaudio espeak

# Windows
# Download and install eSpeak from: http://espeak.sourceforge.net/
# PyAudio wheels are usually available for Windows
```

### Installation

1. **Clone or download the project files**
2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables (optional):**
Create a `.env` file in the project directory:
```env
# Optional: Custom wake word (default: "zora")
WAKE_WORD=jarvis

# Optional: Custom AI models
EMOTION_MODEL=j-hartmann/emotion-english-distilroberta-base
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest

# Optional: Spotify integration
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret

# Optional: OpenAI integration (for future features)
OPENAI_API_KEY=your_openai_api_key
```

### Usage

**Voice Mode (Default):**
```bash
python project_zora_unified.py
```

**Text Mode (for testing):**
```bash
python project_zora_unified.py --text
```

## 🎯 Supported Commands

### 🎵 Media & Entertainment
- "Play [song] on Spotify"
- "Play [video] on YouTube" 
- "Search for [content] on YouTube"

### 🌐 Web & Search
- "Google [query]"
- "Search for [information]"
- "Open [website.com]"
- "Define [word]"
- "Wikipedia [topic]"

### 💻 Applications & System
- "Open [notepad/calculator/chrome/vscode/etc.]"
- "What time is it?"
- "Open [desktop/documents/downloads] folder"
- "Create note [content]"

### 🌤️ Information
- "What's the weather?"
- "Weather in [city]"
- "Help" - Show available commands

### 🛑 Control
- "Stop/Quit/Exit/Goodbye" - Shut down assistant

## 🔧 Configuration

### Wake Word Customization
Change the wake word by setting the `WAKE_WORD` environment variable:
```env
WAKE_WORD=computer
```

### Language Support
The assistant automatically detects and responds in the following languages:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)
- Hindi (hi)
- Arabic (ar)

### Spotify Integration
1. Create a Spotify app at https://developer.spotify.com/
2. Add your credentials to the `.env` file
3. Set redirect URI to `http://localhost:8888/callback`

## 📁 Project Structure

```
project_zora_unified.py    # Main unified assistant file
requirements.txt           # Python dependencies
README.md                 # This file
.env                      # Environment variables (create this)
notes/                    # Created automatically for notes
zora.log                  # Application logs
```

## 🔍 Architecture

### Core Components

1. **SpeechToText**: Handles microphone input and speech recognition
2. **SentimentAnalyzer**: Analyzes emotion and sentiment
3. **IntentClassifier**: Determines user intent from text
4. **AutomationEngine**: Executes system commands and automations
5. **TextToSpeech**: Converts responses to speech
6. **ResponseGenerator**: Creates contextual responses
7. **ProjectZORA**: Main orchestrator class

### Threading Model
- Main thread: Wake word detection
- Command thread: Process commands from queue
- Non-blocking audio playback

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'pyaudio'"**
```bash
# Linux
sudo apt-get install portaudio19-dev python3-dev
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Windows
pip install pyaudio
# If that fails, download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
```

**"Speech recognition not working"**
- Check microphone permissions
- Test with: `python -c "import speech_recognition as sr; print('OK')"`
- Ensure internet connection for Google Speech API

**"TTS not working"**
- For offline TTS: Install espeak system package
- For online TTS: Check internet connection
- Test with: `python -c "import pyttsx3; pyttsx3.speak('test')"`

**"Transformers models downloading slowly"**
- Models are downloaded on first use (~500MB total)
- Use TextBlob fallback: `pip install textblob`

### Performance Optimization

**Reduce startup time:**
- Use TextBlob instead of transformers for sentiment analysis
- Use offline TTS (pyttsx3) instead of gTTS

**Reduce memory usage:**
- Set smaller transformer models in `.env`:
```env
EMOTION_MODEL=cardiffnlp/twitter-roberta-base-emotion
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
```

## 🔒 Privacy & Security

- Speech recognition uses Google's API (requires internet)
- No audio data is stored locally
- Conversation logs are stored in `zora.log` (can be disabled)
- Notes are stored locally in `notes/` directory
- No data is sent to third parties except for configured integrations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## 🙏 Acknowledgments

- Google Speech Recognition API
- HuggingFace Transformers
- OpenAI (for inspiration)
- All the open-source libraries that make this possible

---

**Made with ❤️ by the Project ZORA Team**

For support or questions, please check the troubleshooting section or create an issue in the repository.