# 🚀 QUICK START GUIDE - PROJECT ZORA

Get up and running with ZORA in 5 minutes!

---

## ⚡ Quick Installation

### Step 1: Install Python Dependencies

```bash
pip install speech_recognition gtts transformers torch langdetect requests python-dotenv
```

**Note:** If you encounter PyAudio installation issues, see platform-specific notes below.

### Step 2: Run ZORA

```bash
python project_zora.py
```

**First run will download ML models (~500MB). This is a one-time process.**

---

## 🎤 Try These Commands

Once ZORA is listening, try saying:

```
"What time is it?"
"Open Chrome"
"Play Lofi on YouTube"
"Search for Python tutorials"
"What's the date today?"
"Open my downloads folder"
"Wikipedia Python programming"
"Exit"
```

---

## 🛠️ Platform-Specific Setup

### Windows

```bash
# If PyAudio fails, download the wheel:
# Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
# Download the .whl file matching your Python version
# Install: pip install PyAudio-0.2.13-cp310-cp310-win_amd64.whl
```

### macOS

```bash
# Install portaudio first
brew install portaudio

# Then install PyAudio
pip install pyaudio
```

### Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pyaudio portaudio19-dev espeak espeak-ng

# Then install Python packages
pip install pyaudio pyttsx3
```

---

## 🎯 Command Line Options

### Continuous Mode (Default)
```bash
python project_zora.py
```
Runs continuously until you say "exit"

### Single Command Mode
```bash
python project_zora.py --once
```
Processes one command and exits

### Offline TTS Mode
```bash
python project_zora.py --offline-tts
```
Uses pyttsx3 for offline text-to-speech

### Test Mode (No Microphone)
```bash
python project_zora.py --test "open chrome"
```
Tests with text input instead of speech

---

## 🔧 Troubleshooting

### "No module named 'pyaudio'"

**Solution:** Install PyAudio using platform-specific instructions above.

### "Microphone not found"

**Solution:** 
1. Check microphone is plugged in
2. Give microphone permissions to Terminal/Command Prompt
3. Test: `python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"`

### "Could not understand audio"

**Solution:**
1. Check internet connection (required for Google Speech API)
2. Speak clearly and reduce background noise
3. Ensure microphone is not muted

### Models downloading slowly

**Solution:** 
1. Be patient - first download can take 5-10 minutes
2. Check internet connection
3. Models are cached after first download

---

## 📁 Project Structure

```
project-zora/
├── project_zora.py      # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # Full documentation
├── QUICKSTART.md       # This file
├── .env.example        # Environment variables template
└── audio_output/       # Generated audio files (created automatically)
```

---

## 🎨 Customization

### Change Wake Word

Edit `project_zora.py`:

```python
WAKE_WORD = "zora"  # Change to your preferred wake word
```

### Change Assistant Name

Edit `project_zora.py`:

```python
ASSISTANT_NAME = "ZORA"  # Change to your preferred name
```

### Add Spotify Integration

1. Copy `.env.example` to `.env`
2. Add your Spotify credentials
3. Restart ZORA

---

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore all available commands
- Configure optional integrations (Spotify, etc.)
- Customize for your needs

---

## 💡 Pro Tips

1. **Speak naturally** - ZORA understands conversational language
2. **Check sentiment** - ZORA adapts responses based on your emotional tone
3. **Use specific commands** - "Play Bohemian Rhapsody on Spotify" works better than "play music"
4. **Multi-language** - ZORA automatically detects and responds in your language

---

## 🆘 Need Help?

- Check the [README.md](README.md) for detailed documentation
- Review troubleshooting section above
- Open an issue on GitHub

---

**Happy automating! 🎉**
