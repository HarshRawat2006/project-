#!/usr/bin/env python3
"""
Project ZORA - Interactive Demo Script
=====================================

This script demonstrates Project ZORA's capabilities without requiring
full voice recognition setup. Great for testing and showcasing features.
"""

import sys
import time
from datetime import datetime

def print_banner():
    """Print the demo banner"""
    print("🤖" + "="*58 + "🤖")
    print("🎯                PROJECT ZORA DEMO                    🎯")
    print("🤖" + "="*58 + "🤖")
    print()
    print("This demo showcases ZORA's text processing capabilities")
    print("without requiring microphone setup or voice recognition.")
    print()

def simulate_typing(text, delay=0.03):
    """Simulate typing effect for better demo experience"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def demo_commands():
    """Return a list of demo commands to test"""
    return [
        # Media commands
        "play shape of you on spotify",
        "play cats on youtube",
        "search for python tutorials on youtube",
        
        # Web and search
        "google artificial intelligence",
        "search for weather forecast",
        "open github.com",
        "define machine learning",
        "wikipedia quantum computing",
        
        # Applications
        "open notepad",
        "open calculator", 
        "open chrome",
        
        # System info
        "what time is it",
        "system info",
        
        # File operations
        "create note remember to buy groceries",
        "open documents folder",
        
        # Emotional responses
        "I'm feeling really sad today",
        "This is absolutely amazing!",
        "I'm so frustrated with this problem",
        
        # Help
        "help",
        "what can you do"
    ]

def run_demo():
    """Run the interactive demo"""
    try:
        # Import ZORA components
        from project_zora_unified import ProjectZORA
        
        print("🚀 Initializing Project ZORA...")
        zora = ProjectZORA()
        print("✅ ZORA initialized successfully!")
        print()
        
        # Demo mode selection
        print("Choose demo mode:")
        print("1. 🎬 Automated Demo (showcase all features)")
        print("2. 💬 Interactive Mode (type your own commands)")
        print("3. 🧪 Test Specific Features")
        print()
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            run_automated_demo(zora)
        elif choice == "2":
            run_interactive_mode(zora)
        elif choice == "3":
            run_feature_tests(zora)
        else:
            print("Invalid choice. Running automated demo...")
            run_automated_demo(zora)
            
    except ImportError as e:
        print(f"❌ Could not import Project ZORA: {e}")
        print("Make sure project_zora_unified.py is in the current directory")
        print("and all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def run_automated_demo(zora):
    """Run automated demonstration of all features"""
    print("\n🎬 AUTOMATED DEMO - Showcasing ZORA's Capabilities")
    print("="*55)
    
    demo_cmds = demo_commands()
    
    for i, command in enumerate(demo_cmds, 1):
        print(f"\n[{i}/{len(demo_cmds)}] Testing: ", end="")
        simulate_typing(f'"{command}"')
        
        try:
            # Process command
            response = zora.process_text_command(command)
            
            print("🤖 ZORA:", end=" ")
            simulate_typing(response, delay=0.02)
            
            # Brief pause between commands
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error processing command: {e}")
    
    print("\n🎉 Automated demo completed!")
    print("Try interactive mode to test your own commands.")

def run_interactive_mode(zora):
    """Run interactive mode for custom commands"""
    print("\n💬 INTERACTIVE MODE")
    print("="*20)
    print("Type your commands below. Type 'quit' to exit.")
    print("Examples: 'play music', 'what time is it', 'open chrome'")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'stop']:
                print("👋 Goodbye!")
                break
            
            # Process command
            print("🤖 ZORA: ", end="")
            response = zora.process_text_command(user_input)
            simulate_typing(response, delay=0.02)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_feature_tests(zora):
    """Run specific feature tests"""
    print("\n🧪 FEATURE TESTING MODE")
    print("="*25)
    
    features = {
        "1": ("Sentiment Analysis", test_sentiment_analysis),
        "2": ("Intent Classification", test_intent_classification),
        "3": ("Language Detection", test_language_detection),
        "4": ("Automation Commands", test_automation_commands),
        "5": ("Response Generation", test_response_generation)
    }
    
    print("Available tests:")
    for key, (name, _) in features.items():
        print(f"{key}. {name}")
    print()
    
    choice = input("Enter test number (1-5): ").strip()
    
    if choice in features:
        name, test_func = features[choice]
        print(f"\n🔬 Running {name} Test...")
        print("-" * 40)
        test_func(zora)
    else:
        print("Invalid choice. Running all tests...")
        for name, test_func in features.values():
            print(f"\n🔬 {name} Test:")
            print("-" * 30)
            test_func(zora)

def test_sentiment_analysis(zora):
    """Test sentiment analysis with various inputs"""
    test_texts = [
        "I'm absolutely thrilled about this new project!",
        "This is the worst day ever, everything is going wrong.",
        "The weather is okay today, nothing special.",
        "I'm scared about the upcoming presentation.",
        "What an amazing surprise! I can't believe it!"
    ]
    
    for text in test_texts:
        print(f"\nInput: '{text}'")
        
        # Analyze sentiment
        analysis = zora.sentiment_analyzer.analyze(text)
        print(f"Emotion: {analysis.emotion} (confidence: {analysis.emotion_score:.2f})")
        print(f"Sentiment: {analysis.sentiment} (confidence: {analysis.sentiment_score:.2f})")

def test_intent_classification(zora):
    """Test intent classification"""
    test_commands = [
        "play despacito on spotify",
        "search for funny cats on youtube", 
        "open google chrome",
        "what time is it",
        "create note buy milk and bread",
        "google machine learning tutorials"
    ]
    
    for command in test_commands:
        print(f"\nCommand: '{command}'")
        intent, params = zora.intent_classifier.classify_intent(command)
        print(f"Intent: {intent}")
        print(f"Parameters: {params}")

def test_language_detection(zora):
    """Test language detection"""
    test_phrases = [
        ("Hello, how are you today?", "English"),
        ("Hola, ¿cómo estás hoy?", "Spanish"),
        ("Bonjour, comment allez-vous?", "French"),
        ("Guten Tag, wie geht es Ihnen?", "German"),
        ("नमस्ते, आप कैसे हैं?", "Hindi")
    ]
    
    for phrase, expected_lang in test_phrases:
        print(f"\nPhrase: '{phrase}'")
        detected = zora.stt.language_detector.detect_language(phrase) if zora.stt else "N/A"
        print(f"Expected: {expected_lang}")
        print(f"Detected: {detected}")

def test_automation_commands(zora):
    """Test automation command execution"""
    test_commands = [
        "open notepad",
        "google python programming", 
        "what time is it",
        "create note test automation"
    ]
    
    for command in test_commands:
        print(f"\nCommand: '{command}'")
        try:
            response = zora.process_text_command(command)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")

def test_response_generation(zora):
    """Test response generation with different emotions"""
    scenarios = [
        ("I'm really happy today!", "positive"),
        ("I'm feeling quite sad", "negative"),
        ("Just a normal day", "neutral")
    ]
    
    for text, expected_sentiment in scenarios:
        print(f"\nInput: '{text}' (Expected: {expected_sentiment})")
        
        # Analyze and generate response
        analysis = zora.sentiment_analyzer.analyze(text)
        
        # Create a mock command result
        from project_zora_unified import CommandResult
        mock_result = CommandResult(
            success=True,
            action="test",
            details={},
            response_text="I understand your request."
        )
        
        response = zora.response_generator.generate_response(mock_result, analysis)
        print(f"Generated Response: {response}")

def main():
    """Main demo function"""
    print_banner()
    
    # Check if ZORA is available
    try:
        import project_zora_unified
        print("✅ Project ZORA found and ready for demo!")
    except ImportError:
        print("❌ Project ZORA not found!")
        print("Make sure project_zora_unified.py is in the current directory.")
        return
    
    print("\nDemo Options:")
    print("• This demo works without microphone or voice recognition")
    print("• All features are tested using text input")
    print("• Perfect for development and testing")
    print()
    
    # Ask if user wants to continue
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        # Auto mode for CI/testing
        from project_zora_unified import ProjectZORA
        zora = ProjectZORA()
        run_automated_demo(zora)
    else:
        # Interactive mode
        proceed = input("Ready to start the demo? (y/n): ").lower().strip()
        if proceed in ['y', 'yes', '']:
            run_demo()
        else:
            print("👋 Demo cancelled. Run again when ready!")

if __name__ == "__main__":
    main()