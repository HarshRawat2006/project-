# Project ZORA - Unified Voice Assistant with Sentiment Analysis

A comprehensive, production-ready voice assistant that combines speech recognition, sentiment analysis, natural language processing, automation, and text-to-speech capabilities into one unified system.

## 🌟 Features

### Core Capabilities
- **Speech-to-Text (STT)**: Continuous listening with automatic language detection
- **Sentiment Analysis**: Real-time emotion and sentiment detection using AI models
- **Natural Language Processing**: Intent recognition and command interpretation
- **Automation & App Control**: System-level automation and application control
- **Text-to-Speech (TTS)**: Multilingual speech synthesis with language consistency
- **Continuous Operation**: Always-on listening and response loop

### Supported Languages
- English (en)
- Hindi (hi)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)

### Voice Commands Examples
- **Media**: "Play music on Spotify", "Open YouTube and search cats", "Play lofi music"
- **Search**: "Google weather forecast", "Search for Python tutorials", "Open Wikipedia"
- **Applications**: "Open Chrome", "Launch VS Code", "Open calculator"
- **System**: "What time is it?", "Create a note", "Open my documents"
- **Communication**: "Send email to john@example.com", "Notify the team on Slack"

## 🚀 Quick Start

### Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd project-zora
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configurations
   ```

### Basic Usage

#### Continuous Mode (Recommended)
```bash
python project_zora.py
```
This starts the continuous listening mode. Speak naturally and the assistant will respond.

#### Single Command Mode
```bash
python project_zora.py --text "Play music on Spotify"
```

#### Disable Actions (Testing)
```bash
python project_zora.py --no-actions
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Model Configurations
EMOTION_MODEL=j-hartmann/emotion-english-distilroberta-base
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
WHISPER_MODEL=small

# API Keys (Optional)
OPENAI_API_KEY=your_openai_api_key_here
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
CUSTOM_AUTOMATION_WEBHOOK=your_webhook_url_here

# Spotify Integration (Optional)
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
SPOTIFY_OAUTH_TOKEN=your_spotify_token
SPOTIFY_DEVICE_ID=your_spotify_device_id

# Logging
LOG_LEVEL=INFO
LOG_FILE=project_zora.log
```

### Model Options

#### Emotion Models
- `j-hartmann/emotion-english-distilroberta-base` (default)
- `j-hartmann/emotion-english-distilroberta-base`
- `cardiffnlp/twitter-roberta-base-emotion`

#### Sentiment Models
- `cardiffnlp/twitter-roberta-base-sentiment-latest` (default)
- `cardiffnlp/twitter-roberta-base-sentiment`
- `nlptown/bert-base-multilingual-uncased-sentiment`

#### Whisper Models
- `tiny` - Fastest, least accurate
- `base` - Good balance
- `small` - Better accuracy (default)
- `medium` - High accuracy
- `large` - Best accuracy, slowest

## 🏗️ Architecture

### Core Components

1. **Speech Recognition Module**
   - Primary: `speech_recognition` with Google API
   - Fallback: `faster-whisper` for offline processing
   - Language detection and confidence scoring

2. **Sentiment Analysis Module**
   - Emotion detection using HuggingFace transformers
   - Sentiment classification (positive/negative/neutral)
   - Fallback keyword-based analysis

3. **Natural Language Processing Module**
   - Intent recognition using regex patterns
   - Command parsing and parameter extraction
   - Context-aware response generation

4. **Automation Module**
   - Application launching and control
   - Web browser automation
   - System command execution
   - File and folder operations

5. **Text-to-Speech Module**
   - Primary: Google Text-to-Speech (gTTS)
   - Fallback: pyttsx3 for offline operation
   - Language consistency and voice selection

### Data Flow

```
Audio Input → Speech Recognition → Language Detection
     ↓
Text Processing → Sentiment Analysis → Intent Recognition
     ↓
Response Generation → Action Execution → Text-to-Speech
     ↓
Audio Output
```

## 📁 Project Structure

```
project_zora/
├── project_zora.py          # Main application file
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .env.example            # Environment variables template
├── notes/                  # Generated notes directory
├── logs/                   # Log files (created automatically)
└── temp/                   # Temporary files (created automatically)
```

## 🔧 Advanced Usage

### Custom Intent Patterns

Add custom intent patterns in the `setup_intent_patterns()` method:

```python
self.intent_patterns["custom_intent"] = re.compile(r"your_pattern_here", re.I)
```

### Custom Automation Handlers

Add custom handlers in the `setup_automation_handlers()` method:

```python
self.automation_handlers["custom_intent"] = self._handle_custom_intent
```

### API Integration

The system supports various API integrations:

- **OpenAI GPT**: For advanced language understanding
- **Spotify**: For music playback and control
- **Slack**: For team notifications
- **Custom Webhooks**: For automation triggers

## 🐛 Troubleshooting

### Common Issues

1. **Microphone not detected**:
   - Check microphone permissions
   - Install PyAudio: `pip install pyaudio`
   - On Linux: `sudo apt-get install portaudio19-dev`

2. **Speech recognition errors**:
   - Check internet connection (Google API requires internet)
   - Try offline mode with Whisper
   - Adjust microphone sensitivity

3. **TTS not working**:
   - Install audio players: `mpg321` (Linux) or `mplayer`
   - Check gTTS installation
   - Try pyttsx3 fallback

4. **Model loading errors**:
   - Ensure sufficient disk space (models are large)
   - Check internet connection for model downloads
   - Verify transformers installation

### Debug Mode

Run with debug logging:
```bash
python project_zora.py --log-level DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for transformer models
- Google for speech recognition API
- OpenAI for language models
- The open-source community for various libraries

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `project_zora.log`
3. Open an issue on GitHub
4. Contact the development team

---

**Project ZORA** - Making voice interaction intelligent, emotional, and productive! 🚀