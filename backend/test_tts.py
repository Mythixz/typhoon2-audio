#!/usr/bin/env python3
"""
Test TTS endpoint directly
"""

import requests
import json

def test_tts():
    """Test TTS endpoint"""
    url = "http://localhost:8000/tts"
    
    # Test data
    data = {
        "text": "สวัสดีครับ",
        "language": "th"
    }
    
    try:
        print(f"🔍 Testing TTS endpoint: {url}")
        print(f"📝 Text: {data['text']}")
        print(f"🌍 Language: {data['language']}")
        
        # Send request
        response = requests.post(url, data=data)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ TTS successful!")
            print(f"📁 Content-Type: {response.headers.get('content-type')}")
            print(f"📏 Content-Length: {response.headers.get('content-length')}")
            
            # Save audio file
            with open("test_tts_response.wav", "wb") as f:
                f.write(response.content)
            print("💾 Audio saved as: test_tts_response.wav")
            
        else:
            print("❌ TTS failed!")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"💥 Error: {e}")

def test_speak():
    """Test speak endpoint"""
    url = "http://localhost:8000/speak"
    
    # Test data
    data = {
        "text": "สวัสดีครับ"
    }
    
    try:
        print(f"\n🔍 Testing speak endpoint: {url}")
        print(f"📝 Text: {data['text']}")
        
        # Send request
        response = requests.post(url, data=data)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ Speak successful!")
            print(f"📁 Content-Type: {response.headers.get('content-type')}")
            print(f"📏 Content-Length: {response.headers.get('content-length')}")
            
            # Save audio file
            with open("test_speak_response.wav", "wb") as f:
                f.write(response.content)
            print("💾 Audio saved as: test_speak_response.wav")
            
        else:
            print("❌ Speak failed!")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"💥 Error: {e}")

def test_chat():
    """Test chat endpoint"""
    url = "http://localhost:8000/chat"
    
    # Test data
    data = {
        "user_message": "สวัสดีครับ"
    }
    
    try:
        print(f"\n🔍 Testing chat endpoint: {url}")
        print(f"📝 Message: {data['user_message']}")
        
        # Send request
        response = requests.post(url, data=data)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ Chat successful!")
            print(f"📁 Content-Type: {response.headers.get('content-type')}")
            print(f"📏 Content-Length: {response.headers.get('content-length')}")
            
            # Save audio file
            with open("test_chat_response.wav", "wb") as f:
                f.write(response.content)
            print("💾 Audio saved as: test_chat_response.wav")
            
        else:
            print("❌ Chat failed!")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    print("🚀 Testing TTS/STT Endpoints Directly")
    print("=" * 50)
    
    test_tts()
    test_speak()
    test_chat()
    
    print("\n" + "=" * 50)
    print("✨ Testing completed!")
