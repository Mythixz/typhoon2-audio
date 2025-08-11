from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import os
import wave
import numpy as np
import soundfile as sf

from . import typhoon_tts

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, os.pardir))
STATIC_AUDIO_DIR = os.path.join(PROJECT_ROOT, "static", "audio")
os.makedirs(STATIC_AUDIO_DIR, exist_ok=True)

app = FastAPI(title="AI Call Center Backend (POC)")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=STATIC_AUDIO_DIR), name="audio")


class ChatRequest(BaseModel):
    user_message: str


class ChatResponse(BaseModel):
    ai_response: str
    suggestions: List[str]
    tts_audio_url: str


def get_suggestions(message: str) -> List[str]:
    text = (message or "").strip()
    lower_text = text.lower()

    if "โปรโมชั่น" in text or "promotion" in lower_text:
        return [
            "ดูโปรโมชั่นปัจจุบัน",
            "สอบถามโปรโมชั่นพิเศษ",
            "ยกเลิกโปรโมชั่น",
            "ข้อมูลเงื่อนไขโปรโมชั่น",
        ]
    if "บัตร" in text or "เครดิต" in text or "credit" in lower_text:
        return [
            "ตรวจสอบวงเงินคงเหลือ",
            "เช็คสถานะบัตร",
            "ขอเพิ่มวงเงิน",
            "ขอปิดบัตร",
        ]
    if "โอน" in text or "transfer" in lower_text:
        return [
            "โอนเงินระหว่างบัญชี",
            "โอนต่างธนาคาร",
            "กำหนดรายการโปรด",
            "ตรวจสอบค่าธรรมเนียม",
        ]
    return [
        "ตรวจสอบยอดคงเหลือ",
        "เช็คสถานะคำขอ",
        "ติดต่อเจ้าหน้าที่",
        "สอบถามข้อมูลผลิตภัณฑ์",
    ]


def generate_tone_wav(file_path: str, duration_seconds: float = 1.0, framerate: int = 16000, freq: float = 440.0, amplitude: float = 0.2):
    t = np.linspace(0, duration_seconds, int(duration_seconds * framerate), endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(file_path, samples, framerate, format="WAV")


def generate_tts_audio(text: str) -> str:
    filename = f"tts_{uuid.uuid4().hex}.wav"
    file_path = os.path.join(STATIC_AUDIO_DIR, filename)

    # Try Typhoon2-Audio TTS if enabled
    try:
        result = typhoon_tts.synthesize(text)
        if result is not None:
            wav, sr = result
            sf.write(file_path, wav.T, sr, format="WAV")
            return f"/audio/{filename}"
    except Exception as e:
        print(f"[backend] Typhoon TTS failed, falling back: {e}")

    # Fallback to an audible tone
    try:
        generate_tone_wav(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ไม่สามารถสร้างไฟล์เสียงได้: {e}")
    return f"/audio/{filename}"


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    ai_text = "สวัสดีครับ/ค่ะ ยินดีต้อนรับสู่ศูนย์บริการ AI Call Center"
    suggestions = get_suggestions(payload.user_message)
    tts_url = generate_tts_audio(ai_text)
    return ChatResponse(ai_response=ai_text, suggestions=suggestions, tts_audio_url=tts_url)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "typhoon_tts": typhoon_tts.is_enabled()}


# Feedback endpoint for Human-in-the-Loop simulation
class FeedbackRequest(BaseModel):
    original_message: str
    corrected_message: str


@app.post("/feedback")
async def feedback_endpoint(payload: FeedbackRequest) -> Dict[str, Any]:
    return {
        "status": "received",
        "message": "ขอบคุณสำหรับข้อมูล เราจะนำไปปรับปรุงระบบ",
    } 