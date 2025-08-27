# Modern TTS/STT Backend

ระบบ Backend ที่เสถียรและทันสมัยสำหรับ AI Call Center พร้อมระบบ TTS/STT ที่ใช้งานได้จริง

## 🚀 **Features**

### ✅ **ฟีเจอร์ที่มีครบถ้วน:**

1. **Modern TTS Engine** - หลาย engine พร้อม fallback
2. **Advanced STT Engine** - Whisper + Google Speech Recognition
3. **Robust Audio Processing** - รองรับหลาย format
4. **FastAPI Framework** - เร็วและเสถียร
5. **Multiple Language Support** - ไทย, อังกฤษ, และภาษาอื่นๆ

### 🎯 **TTS Engines:**
- **gTTS** (Google TTS) - คุณภาพสูง, ออนไลน์
- **Edge TTS** (Microsoft) - คุณภาพสูง, ออนไลน์  
- **pyttsx3** - ออฟไลน์, รองรับภาษาอังกฤษ

### 🎤 **STT Engines:**
- **Whisper** (OpenAI) - คุณภาพสูง, ออฟไลน์
- **Google Speech Recognition** - ออนไลน์, รองรับภาษาไทย
- **Edge STT** - Microsoft, ออนไลน์

## 🏗️ **Architecture**

```
Frontend (Next.js) → FastAPI Backend → Multiple TTS/STT Engines
                    ↓
              Audio Processing & Enhancement
```

## 📦 **Installation**

### 1. **Clone Repository**
```bash
git clone <your-repo>
cd typhoon2-audio/backend
```

### 2. **Create Virtual Environment**
```bash
python -m venv venv
venv\Scripts\activate     # Windows
# หรือ
source venv/bin/activate  # Linux/Mac
```

### 3. **Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. **Create Directories**
```bash
mkdir temp
mkdir temp\audio
mkdir static
mkdir static\audio
```

## 🚀 **Running the Backend**

### **Windows (Recommended):**
```bash
start_backend.bat
```

### **Manual Start:**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### **Production Mode:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📡 **API Endpoints**

### **Core Endpoints:**
- `GET /` - Health check
- `GET /health` - สถานะระบบ
- `POST /tts` - แปลงข้อความเป็นเสียง
- `POST /stt` - แปลงเสียงเป็นข้อความ
- `POST /chat` - แชทกับ AI (TTS response)
- `POST /speak` - พูดข้อความ

### **Utility Endpoints:**
- `GET /languages` - ภาษาที่รองรับ

## 🧪 **Testing**

### **Test TTS:**
```bash
curl -X POST "http://localhost:8000/tts" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=สวัสดีครับ&language=th"
```

### **Test STT:**
```bash
curl -X POST "http://localhost:8000/stt" \
  -F "audio_file=@test_audio.wav"
```

### **Test Chat:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "user_message=สวัสดีครับ"
```

## 🔍 **Monitoring & Debugging**

### **Logs:**
ระบบจะแสดง logs แบบ real-time ระหว่างการทำงาน

### **Health Check:**
```bash
curl http://localhost:8000/health
```

### **Engine Status:**
ระบบจะแสดงสถานะของ TTS/STT engines ทั้งหมด

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **TTS Engine Failed:**
   ```bash
   # Check if engines are available
   pip install gtts edge-tts pyttsx3
   ```

2. **STT Engine Failed:**
   ```bash
   # Install Whisper
   pip install openai-whisper
   
   # Install Speech Recognition
   pip install SpeechRecognition
   ```

3. **Audio Processing Failed:**
   ```bash
   # Install audio dependencies
   pip install soundfile librosa pydub
   ```

4. **Port Already in Use:**
   ```bash
   # Change port
   uvicorn app.main:app --port 8001
   ```

## 🔮 **Future Enhancements**

### **Planned Features:**
- [ ] **Real-time Streaming** - WebSocket support
- [ ] **Advanced Emotion Detection** - AI-based emotion analysis
- [ ] **Voice Cloning** - Custom voice synthesis
- [ ] **Batch Processing** - Multiple audio files
- [ ] **Cloud Integration** - AWS, Azure, GCP

### **Integration Possibilities:**
- **AIS OTP** - Real SMS integration
- **Line Bot** - Messaging platform
- **Slack** - Team collaboration
- **Zendesk** - Customer service platform

## 📚 **Dependencies**

### **Core:**
- FastAPI 0.104.1+
- Uvicorn 0.24.0+
- Python 3.8+

### **Audio Processing:**
- soundfile 0.12.1+
- numpy 1.24.3+
- librosa 0.10.1+
- pydub 0.25.1+

### **TTS Engines:**
- gtts 2.4.0+
- edge-tts 6.1.9+
- pyttsx3 2.90+

### **STT Engines:**
- openai-whisper 20231117+
- SpeechRecognition 3.10.0+

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 **License**

MIT License - see LICENSE file for details

## 🆘 **Support**

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

---

**Built with ❤️ for Modern AI Audio Processing** 