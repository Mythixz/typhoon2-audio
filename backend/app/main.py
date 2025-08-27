"""
FastAPI application for lightweight TTS/STT testing
Uses gTTS + pyttsx3 for TTS and Google Cloud Speech-to-Text + SpeechRecognition for STT
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
import io
import logging
from typing import Optional
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lightweight TTS/STT imports
try:
    from gtts import gTTS
    import pyttsx3
    import speech_recognition as sr
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("Some TTS/STT libraries not available. Install with: pip install gtts pyttsx3 SpeechRecognition")

app = FastAPI(
    title="Lightweight TTS/STT API",
    description="Fast TTS/STT API using lightweight models for testing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize lightweight TTS engines
tts_engines = {}

def init_tts_engines():
    """Initialize lightweight TTS engines"""
    global tts_engines
    
    if not TTS_AVAILABLE:
        return False
    
    try:
        # Initialize pyttsx3 (offline TTS)
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        tts_engines['pyttsx3'] = engine
        
        logger.info("✅ pyttsx3 TTS engine initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize pyttsx3: {e}")
        return False

def synthesize_text(text: str, language: str = "en") -> tuple:
    """
    Synthesize text to speech using lightweight engines
    
    Args:
        text: Text to synthesize
        language: Language code
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    if not TTS_AVAILABLE:
        return None, None
    
    try:
        # Try gTTS first (online, better quality)
        if language in ['th', 'en']:
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                tmp_file_path = tmp_file.name
            
            # Load audio and convert to numpy array
            audio_data, sample_rate = sf.read(tmp_file_path)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            logger.info(f"✅ gTTS synthesis successful: {len(audio_data)} samples, {sample_rate} Hz")
            return audio_data, sample_rate
            
    except Exception as e:
        logger.warning(f"gTTS failed: {e}, trying pyttsx3")
    
    try:
        # Fallback to pyttsx3 (offline)
        if 'pyttsx3' in tts_engines:
            engine = tts_engines['pyttsx3']
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                engine.save_to_file(text, tmp_file.name)
                engine.runAndWait()
                tmp_file_path = tmp_file.name
            
            # Load audio
            audio_data, sample_rate = sf.read(tmp_file_path)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            logger.info(f"✅ pyttsx3 synthesis successful: {len(audio_data)} samples, {sample_rate} Hz")
            return audio_data, sample_rate
            
    except Exception as e:
        logger.error(f"All TTS engines failed: {e}")
        return None, None

def transcribe_audio_lightweight(audio_data: np.ndarray, sample_rate: int) -> str:
    """
    Transcribe audio using lightweight STT
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of audio
    
    Returns:
        Transcribed text
    """
    if not TTS_AVAILABLE:
        return None
    
    try:
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            tmp_file_path = tmp_file.name
        
        # Use SpeechRecognition
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(tmp_file_path) as source:
            audio = recognizer.record(source)
        
        # Try Google Speech Recognition (online)
        try:
            text = recognizer.recognize_google(audio, language='th-TH')
            logger.info("✅ Google Speech Recognition successful")
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
            text = None
        except sr.RequestError as e:
            logger.warning(f"Google Speech Recognition service error: {e}")
            text = None
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return text
        
    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        return None

def is_tts_ready() -> bool:
    """Check if TTS engines are ready"""
    return TTS_AVAILABLE and len(tts_engines) > 0

@app.on_event("startup")
async def startup_event():
    """Initialize TTS/STT engines on startup"""
    logger.info("Starting Lightweight TTS/STT API...")
    
    # Initialize TTS engines
    if init_tts_engines():
        logger.info("✅ TTS/STT engines are ready!")
    else:
        logger.warning("⚠️ TTS/STT engines failed to initialize")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Lightweight TTS/STT API is running! 🚀",
        "status": "healthy",
        "tts_ready": is_tts_ready()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "tts_ready": is_tts_ready(),
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.post("/chat")
async def chat_endpoint(user_message: str = Form(...)):
    """
    Chat endpoint that converts text to speech
    
    Args:
        user_message: Text message from user
    
    Returns:
        Audio file as streaming response
    """
    logger.info(f"Chat request received: '{user_message[:50]}...'")
    
    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Convert text to speech
        audio_data, sample_rate = synthesize_text(user_message, language="th")
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech")
        
        # Convert to WAV format
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
        audio_buffer.seek(0)
        
        logger.info(f"Speech synthesis successful for: '{user_message[:30]}...'")
        
        # Return audio as streaming response
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=chat_response.wav"}
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/stt")
async def stt_endpoint(audio_file: UploadFile = File(...)):
    """
    Speech-to-Text endpoint using Google Cloud Speech-to-Text API (FREE)
    
    Args:
        audio_file: Audio file to transcribe
    
    Returns:
        JSON with transcribed text and metadata
    """
    logger.info(f"STT request received: {audio_file.filename}")
    
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="Audio file is required")
    
    try:
        # Read audio file content
        content = await audio_file.read()
        
        # Try Google Cloud Speech-to-Text API first (FREE tier)
        try:
            from google.cloud import speech
            import io
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Initialize Google Cloud Speech client
                client = speech.SpeechClient()
                
                # Read the audio file
                with open(tmp_file_path, "rb") as audio_file_obj:
                    content_audio = audio_file_obj.read()
                
                # Configure audio and recognition settings
                audio = speech.RecognitionAudio(content=content_audio)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,  # Default sample rate
                    language_code="th-TH",  # Thai language
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=False,
                    enable_word_confidence=True,
                )
                
                # Perform the transcription
                response = client.recognize(config=config, audio=audio)
                
                if response.results:
                    transcribed_text = response.results[0].alternatives[0].transcript
                    confidence = response.results[0].alternatives[0].confidence
                    
                    logger.info(f"✅ STT successful with Google Cloud Speech: {confidence:.2f}")
                    
                    return {
                        "text": transcribed_text,
                        "confidence": confidence,
                        "emotion": "neutral",
                        "emotion_confidence": 0.6,
                        "engine": "google_cloud_speech"
                    }
                else:
                    logger.warning("Google Cloud Speech returned no results")
                    return {
                        "text": "ขออภัยครับ ไม่สามารถแปลงเสียงเป็นข้อความได้ กรุณาลองใหม่อีกครั้ง",
                        "confidence": 0.0,
                        "emotion": "neutral",
                        "emotion_confidence": 0.0,
                        "engine": "fallback"
                    }
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            logger.warning(f"Google Cloud Speech failed: {e}, trying fallback methods")
            
            # Fallback to wave module
            try:
                import wave
                import numpy as np
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                try:
                    with wave.open(tmp_file_path, 'rb') as wav_file:
                        # Get audio parameters
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        
                        # Read audio data
                        audio_data = wav_file.readframes(frames)
                        
                        # Convert to numpy array
                        if sample_width == 2:  # 16-bit
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        elif sample_width == 4:  # 32-bit
                            audio_array = np.frombuffer(audio_data, dtype=np.int32)
                        else:  # 8-bit
                            audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                        
                        # Convert to float32 and normalize
                        audio_array = audio_array.astype(np.float32) / (2**(sample_width*8-1))
                        
                        # If stereo, convert to mono
                        if channels == 2:
                            audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                        
                        logger.info(f"✅ Audio loaded with wave: {len(audio_array)} samples, {sample_rate} Hz, {channels} channels")
                        
                        # Use transcribe_audio_lightweight function
                        transcribed_text = transcribe_audio_lightweight(audio_array, sample_rate)
                        
                        if transcribed_text:
                            logger.info("✅ STT successful with wave")
                            return {
                                "text": transcribed_text,
                                "confidence": 0.8,
                                "emotion": "neutral",
                                "emotion_confidence": 0.6,
                                "engine": "wave"
                            }
                        else:
                            logger.warning("Transcription returned None, using fallback")
                            return {
                                "text": "ขออภัยครับ ไม่สามารถแปลงเสียงเป็นข้อความได้ กรุณาลองใหม่อีกครั้ง",
                                "confidence": 0.0,
                                "emotion": "neutral",
                                "emotion_confidence": 0.0,
                                "engine": "fallback"
                            }
                            
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                        
            except Exception as e:
                logger.warning(f"wave failed: {e}, trying SpeechRecognition directly")
                
                # Final fallback to SpeechRecognition directly
                try:
                    recognizer = sr.Recognizer()
                    
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(content)
                        tmp_file_path = tmp_file.name
                    
                    try:
                        with sr.AudioFile(tmp_file_path) as source:
                            audio = recognizer.record(source)
                        
                        # Try Google Speech Recognition
                        try:
                            text = recognizer.recognize_google(audio, language='th-TH')
                            logger.info("✅ STT successful with Google")
                            
                            return {
                                "text": text,
                                "confidence": 0.8,
                                "emotion": "neutral",
                                "emotion_confidence": 0.6,
                                "engine": "google_speech"
                            }
                            
                        except sr.UnknownValueError:
                            logger.warning("Google Speech Recognition could not understand audio")
                            return {
                                "text": "ขออภัยครับ ไม่สามารถแปลงเสียงเป็นข้อความได้ กรุณาลองใหม่อีกครั้ง",
                                "confidence": 0.0,
                                "emotion": "neutral",
                                "emotion_confidence": 0.0,
                                "engine": "fallback"
                            }
                            
                        except sr.RequestError as e:
                            logger.warning(f"Google Speech Recognition service error: {e}")
                            return {
                                "text": "ขออภัยครับ ไม่สามารถเชื่อมต่อกับ Google Speech Recognition ได้ กรุณาลองใหม่อีกครั้ง",
                                "confidence": 0.0,
                                "emotion": "neutral",
                                "emotion_confidence": 0.0,
                                "engine": "fallback"
                            }
                            
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                            
                except Exception as e:
                    logger.error(f"SpeechRecognition failed: {e}")
                    return {
                        "text": "ขออภัยครับ SpeechRecognition ไม่พร้อมใช้งาน กรุณาติดตั้ง SpeechRecognition ก่อน",
                        "confidence": 0.0,
                        "emotion": "neutral",
                        "emotion_confidence": 0.0,
                        "engine": "fallback"
                    }
        
    except Exception as e:
        logger.error(f"STT endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/tts")
async def tts_endpoint(text: str = Form(...), language: str = Form("th")):
    """
    Text-to-Speech endpoint
    
    Args:
        text: Text to convert to speech
        language: Language code (default: th)
    
    Returns:
        Audio file as streaming response
    """
    logger.info(f"TTS request received: '{text[:50]}...' (language: {language})")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Convert text to speech
        audio_data, sample_rate = synthesize_text(text, language=language)
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech")
        
        # Convert to WAV format
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
        audio_buffer.seek(0)
        
        logger.info(f"TTS synthesis successful for: '{text[:30]}...'")
        
        # Return audio as streaming response
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_response.wav"}
        )
        
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/speak")
async def speak_endpoint(text: str = Form(...)):
    """Simple speak endpoint"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        logger.info(f"Speak request: '{text[:50]}...'")
        
        # Use TTS endpoint
        return await tts_endpoint(text, "th")
        
    except Exception as e:
        logger.error(f"Speak error: {e}")
        raise HTTPException(status_code=500, detail=f"Speak failed: {str(e)}")

@app.get("/languages")
async def get_languages():
    """Get available languages for TTS"""
    return {
        "success": True,
        "languages": [
            {"code": "th", "name": "Thai", "native": "ไทย"},
            {"code": "en", "name": "English", "native": "English"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 