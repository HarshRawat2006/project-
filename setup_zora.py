#!/usr/bin/env python3
"""
Project ZORA - Setup and Installation Helper
===========================================

This script helps set up Project ZORA with guided installation
and configuration.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🤖" + "="*58 + "🤖")
    print("🛠️                PROJECT ZORA SETUP                   🛠️")
    print("🤖" + "="*58 + "🤖")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   Project ZORA requires Python 3.7 or higher")
        print("   Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_system_dependencies():
    """Check and guide installation of system dependencies"""
    print("\n🔧 Checking system dependencies...")
    
    system = platform.system().lower()
    
    if system == "linux":
        print("📋 Linux detected. Required packages:")
        print("   sudo apt-get update")
        print("   sudo apt-get install portaudio19-dev python3-dev")
        print("   sudo apt-get install espeak espeak-data libespeak1 libespeak-dev")
        
    elif system == "darwin":  # macOS
        print("🍎 macOS detected. Required packages:")
        print("   brew install portaudio espeak")
        
    elif system == "windows":
        print("🪟 Windows detected. Additional setup:")
        print("   - Download eSpeak from: http://espeak.sourceforge.net/")
        print("   - PyAudio should install automatically")
        
    else:
        print(f"⚠️  Unknown system: {system}")
        print("   You may need to install portaudio and espeak manually")
    
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        print("   Make sure you're in the Project ZORA directory")
        return False
    
    try:
        # Upgrade pip first
        print("📈 Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install requirements
        print("📥 Installing requirements...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("❌ Failed to install dependencies")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ pip not found. Please ensure Python is properly installed")
        return False

def create_config_file():
    """Create configuration file from template"""
    print("\n⚙️  Setting up configuration...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_example.exists():
        print("⚠️  .env.example not found, creating basic .env file")
        
        # Create basic .env file
        basic_config = """# Project ZORA Configuration
WAKE_WORD=zora
LOG_LEVEL=INFO
ENABLE_CONVERSATION_LOG=true
"""
        env_file.write_text(basic_config)
        print("✅ Basic .env file created")
        return True
    
    # Copy example to .env
    try:
        content = env_example.read_text()
        env_file.write_text(content)
        print("✅ .env file created from template")
        print("   Edit .env to customize settings (optional)")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def run_installation_test():
    """Run the installation test"""
    print("\n🧪 Running installation test...")
    
    if not Path("test_zora.py").exists():
        print("⚠️  test_zora.py not found, skipping test")
        return True
    
    try:
        result = subprocess.run([sys.executable, "test_zora.py"], 
                              capture_output=True, text=True, timeout=60)
        
        print("📊 Test Results:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Test Warnings/Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (this is normal for first run)")
        return True
    except Exception as e:
        print(f"❌ Test failed to run: {e}")
        return False

def setup_optional_integrations():
    """Guide setup of optional integrations"""
    print("\n🔗 Optional Integrations Setup")
    print("="*35)
    
    print("\n1. 🎵 Spotify Integration:")
    print("   • Go to: https://developer.spotify.com/dashboard/applications")
    print("   • Create a new app")
    print("   • Add redirect URI: http://localhost:8888/callback")
    print("   • Copy Client ID and Secret to .env file")
    
    print("\n2. 🤖 OpenAI Integration (Future Feature):")
    print("   • Go to: https://platform.openai.com/api-keys")
    print("   • Create an API key")
    print("   • Add to .env file as OPENAI_API_KEY")
    
    print("\n3. 📢 Slack Integration:")
    print("   • Create a Slack webhook URL")
    print("   • Add to .env file as SLACK_WEBHOOK_URL")
    
    print("\nAll integrations are optional and can be set up later!")

def print_next_steps():
    """Print next steps after setup"""
    print("\n🎉 Setup Complete!")
    print("="*20)
    
    print("\n📋 Next Steps:")
    print("1. 🧪 Test installation:")
    print("   python test_zora.py")
    
    print("\n2. 🎬 Try the demo:")
    print("   python demo_zora.py")
    
    print("\n3. 🚀 Start ZORA (voice mode):")
    print("   python project_zora_unified.py")
    
    print("\n4. 💬 Start ZORA (text mode):")
    print("   python project_zora_unified.py --text")
    
    print("\n📚 Documentation:")
    print("   • Read README.md for detailed information")
    print("   • Edit .env for custom configuration")
    print("   • Check zora.log for runtime logs")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check system dependencies
    check_system_dependencies()
    
    # Ask user if they want to continue
    print("\n❓ The setup will now install Python dependencies.")
    print("   This may take a few minutes and requires internet connection.")
    
    proceed = input("\nContinue with installation? (y/n): ").lower().strip()
    if proceed not in ['y', 'yes', '']:
        print("👋 Setup cancelled")
        return 0
    
    # Install dependencies
    if not install_python_dependencies():
        print("\n❌ Setup failed during dependency installation")
        print("   Please check the error messages above and try again")
        return 1
    
    # Create config file
    create_config_file()
    
    # Run test
    print("\n🧪 Would you like to run the installation test?")
    test_choice = input("This will verify everything is working (y/n): ").lower().strip()
    
    if test_choice in ['y', 'yes', '']:
        test_success = run_installation_test()
        if not test_success:
            print("⚠️  Some tests failed, but ZORA may still work")
    
    # Optional integrations
    print("\n🔗 Would you like to see optional integration setup?")
    integration_choice = input("(Spotify, OpenAI, etc.) (y/n): ").lower().strip()
    
    if integration_choice in ['y', 'yes', '']:
        setup_optional_integrations()
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())