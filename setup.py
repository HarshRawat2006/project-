#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project ZORA Setup Script
=========================

This script helps set up Project ZORA with all necessary dependencies
and configurations.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_system_dependencies():
    """Install system-level dependencies"""
    system = platform.system().lower()
    
    if system == "linux":
        commands = [
            ("sudo apt-get update", "Updating package list"),
            ("sudo apt-get install -y portaudio19-dev python3-pyaudio espeak mpg321", "Installing audio dependencies"),
            ("sudo apt-get install -y ffmpeg", "Installing FFmpeg"),
        ]
    elif system == "darwin":  # macOS
        commands = [
            ("brew install portaudio espeak mpg321", "Installing audio dependencies"),
            ("brew install ffmpeg", "Installing FFmpeg"),
        ]
    elif system == "windows":
        print("ℹ️  Windows detected - please install PyAudio manually if needed")
        return True
    else:
        print(f"⚠️  Unknown system: {system} - please install dependencies manually")
        return True
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️  {description} failed - you may need to install manually")
    
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def create_directories():
    """Create necessary directories"""
    directories = ["notes", "logs", "temp", "models"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    if not Path(".env").exists():
        if Path(".env.example").exists():
            run_command("cp .env.example .env", "Creating .env file from template")
            print("📝 Please edit .env file with your API keys and configurations")
        else:
            print("⚠️  .env.example not found - please create .env manually")
    else:
        print("✅ .env file already exists")

def test_installation():
    """Test if the installation works"""
    print("🧪 Testing installation...")
    try:
        # Test basic imports
        import speech_recognition
        print("✅ speech_recognition imported successfully")
        
        import requests
        print("✅ requests imported successfully")
        
        # Test microphone access
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("✅ Microphone access working")
        
        print("🎉 Installation test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Project ZORA Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install system dependencies
    print("\n📦 Installing system dependencies...")
    install_system_dependencies()
    
    # Install Python dependencies
    print("\n🐍 Installing Python dependencies...")
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create .env file
    print("\n⚙️  Setting up configuration...")
    create_env_file()
    
    # Test installation
    print("\n🧪 Testing installation...")
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit .env file with your API keys (optional)")
        print("2. Run: python project_zora.py")
        print("3. Start speaking to your voice assistant!")
    else:
        print("\n❌ Setup completed with errors")
        print("Please check the error messages above and fix any issues")

if __name__ == "__main__":
    main()