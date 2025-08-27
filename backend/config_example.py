"""
Configuration example for Modern TTS/STT Backend
Copy this file to config.py and modify as needed
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the Modern TTS/STT API"""
    
    def __init__(self):
        # API Configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
        # TTS Configuration
        self.DEFAULT_TTS_LANGUAGE = os.getenv("DEFAULT_TTS_LANGUAGE", "th")
        self.TTS_ENGINE_PRIORITY = os.getenv("TTS_ENGINE_PRIORITY", "gtts,edge-tts,pyttsx3").split(",")
        
        # STT Configuration
        self.DEFAULT_STT_LANGUAGE = os.getenv("DEFAULT_STT_LANGUAGE", "th")
        self.STT_ENGINE_PRIORITY = os.getenv("STT_ENGINE_PRIORITY", "whisper,speech_recognition").split(",")
        
        # Audio Configuration
        self.DEFAULT_SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", "22050"))
        self.MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", "50"))
        
        # CORS Configuration
        self.ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Model Configuration
        self.WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
        
        # Print configuration on startup
        self._print_config()
    
    def _print_config(self):
        """Print current configuration for debugging"""
        print("🔧 Configuration loaded:")
        print(f"   API: {self.API_HOST}:{self.API_PORT}")
        print(f"   Debug: {self.DEBUG_MODE}")
        print(f"   TTS Engine Priority: {self.TTS_ENGINE_PRIORITY}")
        print(f"   STT Engine Priority: {self.STT_ENGINE_PRIORITY}")
        print(f"   Whisper Model: {self.WHISPER_MODEL_SIZE}")
        print(f"   Sample Rate: {self.DEFAULT_SAMPLE_RATE} Hz")
        print(f"   Max Audio Size: {self.MAX_AUDIO_SIZE_MB} MB")
    
    def get_feature_status(self) -> Dict[str, Any]:
        """Get current feature status"""
        return {
            "tts_engines": {
                "gtts": "enabled",
                "edge-tts": "enabled",
                "pyttsx3": "enabled",
            },
            "stt_engines": {
                "whisper": "enabled",
                "speech_recognition": "enabled",
            },
            "audio_formats": ["wav", "mp3", "m4a", "flac", "ogg", "aac"],
            "languages": "multi-language support"
        }
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.DEBUG_MODE
    
    def get_cors_origins(self) -> list:
        """Get CORS origins list"""
        if "*" in self.ALLOW_ORIGINS:
            return ["*"]
        return [origin.strip() for origin in self.ALLOW_ORIGINS if origin.strip()]

# Example usage:
if __name__ == "__main__":
    config = Config()
    print("\nFeature Status:")
    print(json.dumps(config.get_feature_status(), indent=2))
