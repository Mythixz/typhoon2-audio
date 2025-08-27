# 🚀 Quick Start Guide

## เริ่มต้นใช้งาน Modern TTS/STT Backend ใน 5 นาที

### 📋 **Prerequisites**
- Python 3.8+ ✅
- Windows 10/11 ✅
- Internet connection ✅

### ⚡ **Quick Start (Windows)**

#### 1. **เปิด Command Prompt ใน backend folder**
```cmd
cd typhoon2-audio\backend
```

#### 2. **รัน startup script**
```cmd
start_backend.bat
```

#### 3. **รอให้ระบบพร้อม**
```
✅ Audio processor initialized
✅ TTS engine initialized  
✅ STT engine initialized
🎉 All engines are ready!
```

#### 4. **ทดสอบระบบ**
เปิด browser ไปที่: http://localhost:8000

### 🧪 **Test the System**

#### **Option 1: ใช้ Test Script (แนะนำ)**
```cmd
python test_system.py
```

#### **Option 2: ทดสอบด้วย Browser**
1. ไปที่ http://localhost:8000
2. ดู health status
3. ทดสอบ endpoints

#### **Option 3: ทดสอบด้วย curl**
```cmd
# Health check
curl http://localhost:8000/health

# TTS test
curl -X POST "http://localhost:8000/tts" -d "text=สวัสดีครับ&language=th"

# Chat test  
curl -X POST "http://localhost:8000/chat" -d "user_message=สวัสดีครับ"
```

### 🔧 **Troubleshooting**

#### **ปัญหา: "No module named 'xxx'"**
```cmd
# เปิด virtual environment
venv\Scripts\activate

# ติดตั้ง dependencies
pip install -r requirements.txt
```

#### **ปัญหา: Port 8000 already in use**
```cmd
# เปลี่ยน port
uvicorn app.main:app --port 8001
```

#### **ปัญหา: TTS/STT engines failed**
```cmd
# ติดตั้ง engines แยก
pip install gtts edge-tts pyttsx3
pip install openai-whisper SpeechRecognition
```

### 📱 **Frontend Integration**

#### 1. **เปิด Terminal ใหม่**
#### 2. **ไปที่ frontend folder**
```cmd
cd typhoon2-audio\frontend
```

#### 3. **รัน frontend**
```cmd
npm run dev
```

#### 4. **ทดสอบระบบ**
เปิด browser ไปที่: http://localhost:3000

### 🎯 **What's Working Now**

✅ **TTS (Text-to-Speech)**
- แปลงข้อความเป็นเสียง
- รองรับภาษาไทย
- หลาย engine พร้อม fallback

✅ **STT (Speech-to-Text)**  
- แปลงเสียงเป็นข้อความ
- รองรับภาษาไทย
- Whisper + Google Speech Recognition

✅ **Audio Processing**
- รองรับหลาย format (WAV, MP3, M4A, FLAC, OGG, AAC)
- Audio enhancement
- Noise reduction

✅ **API Endpoints**
- `/tts` - แปลงข้อความเป็นเสียง
- `/stt` - แปลงเสียงเป็นข้อความ  
- `/chat` - แชทกับ AI
- `/speak` - พูดข้อความ

### 🚀 **Next Steps**

1. **ทดสอบ TTS/STT** - ใช้ test_system.py
2. **ทดสอบ Frontend** - เปิด http://localhost:3000
3. **ทดสอบการทำงานร่วมกัน** - ใช้ frontend เรียก backend
4. **ปรับแต่ง configuration** - แก้ไข config.py

### 📞 **Need Help?**

- **Backend Issues**: ดู logs ใน terminal
- **Frontend Issues**: ดู console ใน browser
- **Integration Issues**: ตรวจสอบ ports และ URLs

---

**🎉 ยินดีด้วย! ตอนนี้คุณมีระบบ TTS/STT ที่ใช้งานได้จริงแล้ว**
