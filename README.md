# 🤖 PROJECT ZORA

**Unified Voice Assistant with Sentiment Analysis**

A production-ready, modular voice assistant that listens to your speech, analyzes your emotions, understands your intent, and executes commands with context-aware responses.

---

## 🌟 Features

### 🎤 Speech-to-Text (STT)
- Continuous listening with ambient noise adjustment
- Automatic language detection (supports 15+ languages)
- Powered by Google Speech Recognition API

### 💭 Sentiment & Emotion Analysis
- Real-time emotion detection (anger, joy, sadness, fear, surprise, disgust, neutral)
- Sentiment analysis (positive, negative, neutral)
- Uses state-of-the-art transformer models from HuggingFace

### 🎯 Natural Language Understanding
- Intent recognition with 20+ command patterns
- Smart parameter extraction
- Handles multi-language commands

### ⚙️ Automation & App Control
- **Media Control**: Play songs on Spotify, YouTube videos with autoplay
- **Web Navigation**: Google search, open websites, Wikipedia, Maps
- **Application Control**: Open/close apps (Chrome, Spotify, VSCode, etc.)
- **System Commands**: Get time/date, open folders, manage files
- **Information Queries**: Definitions, directions, web searches

### 🗣️ Intelligent Response Generation
- Context-aware responses based on sentiment
- Emotion-adapted tone (empathetic, cheerful, serious)
- Natural conversation flow

### 🔊 Text-to-Speech (TTS)
- Multilingual support (responds in the same language as input)
- Online TTS (gTTS) and offline TTS (pyttsx3)
- Natural-sounding voice output

---

## 📋 Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Internet Connection**: Required for speech recognition and online TTS
- **Microphone**: For voice input

---

## 🚀 Installation

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd project-zora
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### Platform-Specific Notes:

**Windows:**
- PyAudio may require a pre-built wheel. Download from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
- Install: `pip install PyAudio-0.2.13-cpXX-cpXX-win_amd64.whl`

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install python3-pyaudio portaudio19-dev
sudo apt-get install espeak espeak-ng  # For offline TTS
pip install pyaudio
```

### 4. Download ML Models (First Run)

On first run, the script will automatically download required transformer models (~500MB). This may take a few minutes.

---

## 🎮 Usage

### Basic Usage (Continuous Mode)

```bash
python project_zora.py
```

The assistant will continuously listen and respond to your commands.

### Process One Command and Exit

```bash
python project_zora.py --once
```

### Use Offline TTS (pyttsx3)

```bash
python project_zora.py --offline-tts
```

### Test with Text Input (Debug Mode)

```bash
python project_zora.py --test "open chrome and search for python tutorials"
```

---

## 💬 Example Commands

### 🎵 Media & Entertainment
- "Play Shape of You on Spotify"
- "Play lofi music on YouTube"
- "Open Spotify"
- "Play music"

### 🌐 Web Search & Navigation
- "Search for Python tutorials"
- "Google the weather"
- "Open YouTube"
- "Go to github.com"

### 📱 Application Control
- "Open Chrome"
- "Launch VSCode"
- "Open Calculator"
- "Start Notepad"

### 🕐 System Information
- "What time is it?"
- "What's the date today?"

### 📁 File Management
- "Open my downloads folder"
- "Show my documents"
- "Open desktop"

### 📚 Information Queries
- "Wikipedia quantum computing"
- "Define artificial intelligence"
- "Maps directions to Central Park"

### 🚪 Control Commands
- "Exit" / "Quit" / "Stop" / "Goodbye"
- "Help" / "What can you do?"

---

## ⚙️ Configuration

### Environment Variables (Optional)

Create a `.env` file in the project root:

```env
# Sentiment Models (optional - uses defaults if not specified)
EMOTION_MODEL=j-hartmann/emotion-english-distilroberta-base
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest

# Spotify Integration (optional)
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
SPOTIFY_DEVICE_ID=your_device_id
```

### Spotify Setup (Optional)

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create an app and get your Client ID and Client Secret
3. Add `http://localhost:8888/callback` as a Redirect URI
4. Add credentials to `.env` file

---

## 🏗️ Architecture

### Modular Design

```
project_zora.py
├── SpeechToTextEngine      # Handles voice input
├── SentimentAnalyzer       # Emotion & sentiment detection
├── IntentRecognizer        # Command parsing & intent extraction
├── AutomationEngine        # Command execution & app control
├── ResponseGenerator       # Context-aware response creation
├── TextToSpeechEngine      # Voice output
└── VoiceAssistant          # Main orchestrator
```

### Data Flow

```
User Speech
    ↓
[Speech-to-Text] → SpeechInput
    ↓
[Sentiment Analysis] → SentimentAnalysis
    ↓
[Intent Recognition] → CommandIntent
    ↓
[Automation Execution] → ActionResult
    ↓
[Response Generation] → ResponseText
    ↓
[Text-to-Speech] → Audio Output
```

---

## 🧪 Development

### Code Structure

- **Clean, modular functions** with clear separation of concerns
- **Detailed documentation** with docstrings for all classes and methods
- **Type hints** for better IDE support and code clarity
- **Error handling** with graceful fallbacks
- **Logging** for debugging and monitoring

### Adding New Commands

1. Add a new pattern to `IntentRecognizer.intent_patterns`
2. Create a handler method in `AutomationEngine` (e.g., `_handle_my_command`)
3. Map the intent to the handler in `AutomationEngine.execute()`

Example:

```python
# In IntentRecognizer.__init__
self.intent_patterns["my_command"] = re.compile(r"\bmy pattern\b", re.I)

# In AutomationEngine
def _handle_my_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
    # Your implementation
    return {"status": "success", "message": "Command executed!"}

# Map in AutomationEngine.execute()
handlers = {
    # ... existing handlers
    "my_command": self._handle_my_command,
}
```

---

## 🐛 Troubleshooting

### Issue: Microphone not detected

**Solution:**
- Check microphone permissions in system settings
- Test with: `python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"`

### Issue: PyAudio installation fails

**Solution:**
- **Windows**: Download pre-built wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
- **macOS**: `brew install portaudio` then `pip install pyaudio`
- **Linux**: `sudo apt-get install python3-pyaudio portaudio19-dev`

### Issue: Models downloading too slow

**Solution:**
- Use a VPN or proxy
- Manually download models from HuggingFace and specify local path in `.env`

### Issue: Speech recognition not working

**Solution:**
- Check internet connection (Google Speech API requires internet)
- Reduce ambient noise
- Speak clearly and not too fast
- Adjust `energy_threshold` in `SpeechToTextEngine.__init__`

---

## 📝 Supported Languages

ZORA automatically detects and responds in the following languages:

- 🇺🇸 English (en)
- 🇮🇳 Hindi (hi)
- 🇪🇸 Spanish (es)
- 🇫🇷 French (fr)
- 🇩🇪 German (de)
- 🇮🇹 Italian (it)
- 🇵🇹 Portuguese (pt)
- 🇷🇺 Russian (ru)
- 🇯🇵 Japanese (ja)
- 🇰🇷 Korean (ko)
- 🇨🇳 Chinese (zh-cn)
- 🇸🇦 Arabic (ar)

---

## 🔒 Privacy & Security

- **No data collection**: All processing happens locally except for speech recognition (Google API)
- **No persistent storage**: Audio files are temporary and can be auto-deleted
- **Optional features**: Spotify, YouTube integrations are optional

---

## 📜 License

MIT License - Feel free to use, modify, and distribute.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Test thoroughly
5. Submit a pull request

---

## 🙏 Acknowledgments

- **HuggingFace** for transformer models
- **Google** for Speech Recognition API
- **Spotify** for music playback API
- All open-source contributors

---

## 📧 Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Built with ❤️ by the Project ZORA Team**

*Making human-AI interaction more natural and emotionally intelligent.*
