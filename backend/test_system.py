#!/usr/bin/env python3
"""
Test script for Modern TTS/STT Backend
Tests all major functionality
"""

import asyncio
import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_TEXT = "สวัสดีครับ นี่คือการทดสอบระบบ TTS และ STT"
TEST_LANGUAGE = "th"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_languages():
    """Test languages endpoint"""
    print("\n🌍 Testing languages endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/languages")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Languages: {data}")
            return True
        else:
            print(f"❌ Languages failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Languages error: {e}")
        return False

def test_tts():
    """Test TTS endpoint"""
    print("\n🎤 Testing TTS endpoint...")
    try:
        data = {
            "text": TEST_TEXT,
            "language": TEST_LANGUAGE
        }
        
        response = requests.post(f"{BASE_URL}/tts", data=data)
        if response.status_code == 200:
            # Save audio file
            output_file = "test_tts_output.wav"
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"✅ TTS successful: {output_file}")
            print(f"   File size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ TTS failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False

def test_speak():
    """Test speak endpoint"""
    print("\n🗣️ Testing speak endpoint...")
    try:
        data = {
            "text": TEST_TEXT
        }
        
        response = requests.post(f"{BASE_URL}/speak", data=data)
        if response.status_code == 200:
            # Save audio file
            output_file = "test_speak_output.wav"
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"✅ Speak successful: {output_file}")
            print(f"   File size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Speak failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Speak error: {e}")
        return False

def test_chat():
    """Test chat endpoint"""
    print("\n💬 Testing chat endpoint...")
    try:
        data = {
            "user_message": TEST_TEXT
        }
        
        response = requests.post(f"{BASE_URL}/chat", data=data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat successful:")
            print(f"   AI Response: {data.get('ai_response', 'N/A')}")
            print(f"   Audio URL: {data.get('tts_audio_url', 'N/A')}")
            print(f"   Suggestions: {data.get('suggestions', [])}")
            return True
        else:
            print(f"❌ Chat failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return False

def test_stt_with_file(audio_file):
    """Test STT endpoint with audio file"""
    print(f"\n🎵 Testing STT endpoint with {audio_file}...")
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    try:
        with open(audio_file, "rb") as f:
            files = {"audio_file": f}
            response = requests.post(f"{BASE_URL}/stt", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ STT successful:")
            print(f"   Text: {data.get('text', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 'N/A')}")
            print(f"   Emotion: {data.get('emotion', 'N/A')}")
            print(f"   Engine: {data.get('engine', 'N/A')}")
            return True
        else:
            print(f"❌ STT failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ STT error: {e}")
        return False

def create_test_audio():
    """Create a simple test audio file"""
    print("\n🎵 Creating test audio file...")
    try:
        import numpy as np
        import soundfile as sf
        
        # Create a simple beep sound
        sample_rate = 22050
        duration = 2.0  # 2 seconds
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Save as WAV
        output_file = "test_audio.wav"
        sf.write(output_file, audio, sample_rate)
        print(f"✅ Test audio created: {output_file}")
        return output_file
        
    except ImportError:
        print("⚠️ soundfile not available, using existing audio file")
        # Look for existing audio files
        audio_files = list(Path(".").glob("*.wav")) + list(Path(".").glob("*.mp3"))
        if audio_files:
            return str(audio_files[0])
        else:
            print("❌ No audio files found for testing")
            return None
    except Exception as e:
        print(f"❌ Failed to create test audio: {e}")
        return None

def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    test_files = [
        "test_tts_output.wav",
        "test_speak_output.wav",
        "test_audio.wav"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"✅ Removed: {file}")
            except Exception as e:
                print(f"⚠️ Failed to remove {file}: {e}")

def main():
    """Main test function"""
    print("🚀 Starting Modern TTS/STT Backend Tests")
    print("=" * 50)
    
    # Check if backend is running
    if not test_health():
        print("\n❌ Backend is not running. Please start the backend first.")
        print("   Run: start_backend.bat")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Languages", test_languages),
        ("TTS", test_tts),
        ("Speak", test_speak),
        ("Chat", test_chat),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Test STT if we have audio
    test_audio = create_test_audio()
    if test_audio:
        try:
            result = test_stt_with_file(test_audio)
            results.append(("STT", result))
        except Exception as e:
            print(f"❌ STT test crashed: {e}")
            results.append(("STT", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
    
    # Cleanup
    cleanup_test_files()
    
    print("\n✨ Testing completed!")

if __name__ == "__main__":
    main()
