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
import time
import random

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
    # lightweight add-ons for hackathon
    candidates: List[str] = []
    kb: List[Dict[str, str]] = []


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

    # Fallback to gTTS (CPU, Thai supported)
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="th")
        mp3_tmp = file_path.replace(".wav", ".mp3")
        tts.save(mp3_tmp)
        # Convert mp3 -> wav via soundfile not supported. We will just serve mp3 directly.
        return f"/audio/{os.path.basename(mp3_tmp)}"
    except Exception as e:
        print(f"[backend] gTTS failed, falling back to tone: {e}")

    # Fallback to an audible tone
    try:
        generate_tone_wav(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ไม่สามารถสร้างไฟล์เสียงได้: {e}")
    return f"/audio/{filename}"


# --- Hackathon-lite helpers ---

def generate_candidates(user_message: str) -> List[str]:
    msg = (user_message or "").strip()
    base = [
        "สวัสดีค่ะ ดิฉันยินดีให้ความช่วยเหลือค่ะ ต้องการติดต่อเรื่องใดคะ",
        "รบกวนขอรายละเอียดเพิ่มเติม เช่น เลขที่ลูกค้า/หมายเลขอ้างอิง เพื่อช่วยตรวจสอบได้เร็วขึ้นค่ะ",
        "ขอบคุณค่ะ ขอตรวจสอบข้อมูลสักครู่ กรุณาถือสายรอเล็กน้อยนะคะ",
    ]
    if "โปร" in msg or "promotion" in msg.lower():
        base.insert(1, "ขอแจ้งโปรโมชันปัจจุบัน พร้อมเงื่อนไขคร่าวๆ ให้ทราบนะคะ ต้องการสมัครเลยไหมคะ")
    if "บัตร" in msg or "เครดิต" in msg:
        base.insert(1, "ต้องการเช็คสถานะบัตร/วงเงิน/เพิ่มวงเงิน ใช่ไหมคะ")
    return base[:4]


def get_kb_snippets(user_message: str) -> List[Dict[str, str]]:
    # stub KB for demo
    return [
        {"title": "คู่มือการยืนยันตัวตน", "snippet": "เตรียมบัตร ปชช. และข้อมูลวันเกิด เพื่อยืนยันก่อนดำเนินการ."},
        {"title": "โปรโมชันปัจจุบัน", "snippet": "แพ็กเสริมอินเทอร์เน็ต 10GB/99บาท ต่ออายุอัตโนมัติยกเลิกได้ทุกเมื่อ."},
    ]


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    ai_text = "สวัสดีครับ/ค่ะ ยินดีต้อนรับสู่ศูนย์บริการ AI Call Center"
    suggestions = get_suggestions(payload.user_message)
    tts_url = generate_tts_audio(ai_text)
    # add candidates + kb for agent assist
    candidates = generate_candidates(payload.user_message)
    kb = get_kb_snippets(payload.user_message)
    return ChatResponse(ai_response=ai_text, suggestions=suggestions, tts_audio_url=tts_url, candidates=candidates, kb=kb)


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


# --- New: simple /speak endpoint for TTS any text ---
class SpeakRequest(BaseModel):
    text: str


class SpeakResponse(BaseModel):
    tts_audio_url: str


@app.post("/speak", response_model=SpeakResponse)
async def speak_endpoint(payload: SpeakRequest) -> SpeakResponse:
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="missing text")
    url = generate_tts_audio(text)
    return SpeakResponse(tts_audio_url=url)


# --- Minimal OTP API (mock provider; replaceable with AIS OTP) ---
OTP_EXPIRY_SECONDS = 180
OTP_STORE: Dict[str, Dict[str, Any]] = {}


class OtpSendRequest(BaseModel):
    phone: str


class OtpSendResponse(BaseModel):
    request_id: str


class OtpVerifyRequest(BaseModel):
    request_id: str
    code: str


class OtpVerifyResponse(BaseModel):
    verified: bool


@app.post("/otp/send", response_model=OtpSendResponse)
async def otp_send(payload: OtpSendRequest) -> OtpSendResponse:
    phone = (payload.phone or "").strip()
    if not phone:
        raise HTTPException(status_code=400, detail="missing phone")
    request_id = uuid.uuid4().hex
    code = f"{random.randint(0, 999999):06d}"
    OTP_STORE[request_id] = {
        "phone": phone,
        "code": code,
        "ts": time.time(),
    }
    print(f"[otp] request_id={request_id} phone={phone} code={code}")
    # TODO: integrate AIS OTP here if credentials available
    return OtpSendResponse(request_id=request_id)


@app.post("/otp/verify", response_model=OtpVerifyResponse)
async def otp_verify(payload: OtpVerifyRequest) -> OtpVerifyResponse:
    req = OTP_STORE.get(payload.request_id)
    if not req:
        return OtpVerifyResponse(verified=False)
    if time.time() - req["ts"] > OTP_EXPIRY_SECONDS:
        OTP_STORE.pop(payload.request_id, None)
        return OtpVerifyResponse(verified=False)
    ok = payload.code.strip() == req["code"]
    if ok:
        OTP_STORE.pop(payload.request_id, None)
    return OtpVerifyResponse(verified=ok) 