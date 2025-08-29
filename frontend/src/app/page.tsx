"use client";

import React, { useMemo, useRef, useState } from "react";
import ChatMessage from "@/components/ChatMessage";
import SuggestionButtons from "@/components/SuggestionButtons";
import HITLModal from "@/components/HITLModal";
import SpeechToText from "@/components/SpeechToText";
import TwoWayCall from "@/components/TwoWayCall";
import { 
  postChat, 
  postFeedback, 
  postSpeak, 
  postOtpSend, 
  postOtpVerify, 
  postEnhancedChat,
  type ChatResponse 
} from "@/lib/api";
import Image from "next/image";

type TabType = 'chat' | 'speech' | 'call' | 'enhanced';

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<TabType>('chat');
  const [messages, setMessages] = useState<Array<{ role: "user" | "ai"; text: string; audioUrl?: string }>>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([
    "สวัสดีครับ มีอะไรให้ช่วยเหลือไหม?",
    "ต้องการทราบข้อมูลเกี่ยวกับบริการอะไรบ้าง?",
    "มีปัญหาอะไรที่ต้องการความช่วยเหลือไหม?",
    "ต้องการทดสอบระบบอะไรเป็นพิเศษไหม?"
  ]);
  const [candidates, setCandidates] = useState<string[]>([
    "ขอบคุณสำหรับการต้อนรับครับ ผมต้องการทดสอบระบบแชทกับ AI",
    "สวัสดีครับ ผมสนใจในบริการของ AI Call Center ครับ",
    "ต้องการทราบข้อมูลเพิ่มเติมเกี่ยวกับระบบครับ"
  ]);
  const [kb, setKb] = useState<Array<{ title: string; snippet: string }>>([
    {
      title: "AI Call Center System",
      snippet: "ระบบศูนย์บริการ AI แบบครบวงจรสำหรับผู้พิการทางการได้ยิน"
    },
    {
      title: "บริการหลัก",
      snippet: "แชทพื้นฐาน, แปลงเสียง, สนทนาสองทาง, แชทขั้นสูง"
    },
    {
      title: "เทคโนโลยีที่ใช้",
      snippet: "AI, Speech Recognition, Text-to-Speech, Emotion Detection"
    }
  ]);
  const [showHitl, setShowHitl] = useState(false);
  const [lastUserMessage, setLastUserMessage] = useState("");
  const [otpPhone, setOtpPhone] = useState("");
  const [otpReqId, setOtpReqId] = useState<string | null>(null);
  const [otpCode, setOtpCode] = useState("");
  const [otpVerified, setOtpVerified] = useState(false);
  const [transcriptionText, setTranscriptionText] = useState("");
  const [detectedEmotion, setDetectedEmotion] = useState("");
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  async function handleSend() {
    if (!canSend) return;
    const text = input.trim();
    setLastUserMessage(text);
    setInput("");
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", text }]);
    
    // Demo mode - simulate AI response
    setTimeout(() => {
      let aiResponse = "";
      if (text.toLowerCase().includes("สวัสดี") || text.toLowerCase().includes("hello")) {
        aiResponse = "สวัสดีครับ! ยินดีต้อนรับสู่ AI Call Center System ผมเป็น AI Assistant ที่พร้อมให้บริการคุณครับ มีอะไรให้ช่วยเหลือไหม?";
      } else if (text.toLowerCase().includes("บริการ") || text.toLowerCase().includes("service")) {
        aiResponse = "เรามีบริการหลัก 4 ประเภทครับ:\n1. แชทพื้นฐาน - สนทนากับ AI\n2. แปลงเสียง - แปลงเสียงเป็นข้อความ\n3. สนทนาสองทาง - สนทนากับ AI แบบสองทาง\n4. แชทขั้นสูง - พร้อมตรวจจับอารมณ์";
      } else if (text.toLowerCase().includes("ทดสอบ") || text.toLowerCase().includes("test")) {
        aiResponse = "ยินดีครับ! นี่คือการทดสอบระบบแชทกับ AI ระบบทำงานได้ปกติครับ คุณสามารถทดสอบฟีเจอร์อื่นๆ ได้เช่นกัน";
      } else {
        aiResponse = "ขอบคุณสำหรับข้อความครับ ผมเข้าใจแล้ว และพร้อมให้บริการคุณต่อไปครับ มีอะไรให้ช่วยเหลือเพิ่มเติมไหม?";
      }
      
      setMessages((prev) => [
        ...prev,
        { role: "ai", text: aiResponse },
      ]);
      setLoading(false);
    }, 1000);
  }

  async function handleEnhancedChat() {
    if (!input.trim() || loading) return;
    const text = input.trim();
    setLastUserMessage(text);
    setInput("");
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", text }]);
    
    // Demo mode - simulate enhanced AI response with emotion detection
    setTimeout(() => {
      let aiResponse = "";
      let emotion = "";
      
      if (text.toLowerCase().includes("สวัสดี") || text.toLowerCase().includes("hello")) {
        aiResponse = "สวัสดีครับ! ยินดีต้อนรับสู่ระบบแชทขั้นสูงครับ ผมสามารถตรวจจับอารมณ์และตอบสนองตามบริบทได้ครับ";
        emotion = "ยินดี";
      } else if (text.toLowerCase().includes("เศร้า") || text.toLowerCase().includes("sad")) {
        aiResponse = "ผมเข้าใจความรู้สึกของคุณครับ อย่าเพิ่งท้อใจนะครับ มีอะไรให้ผมช่วยเหลือไหม? ผมพร้อมเป็นกำลังใจให้คุณครับ";
        emotion = "เห็นใจ";
      } else if (text.toLowerCase().includes("โกรธ") || text.toLowerCase().includes("angry")) {
        aiResponse = "ผมเข้าใจว่าคุณอาจจะไม่พอใจอะไรบางอย่างครับ ลองหายใจลึกๆ และบอกผมว่าเกิดอะไรขึ้นครับ";
        emotion = "เข้าใจ";
      } else if (text.toLowerCase().includes("ดีใจ") || text.toLowerCase().includes("happy")) {
        aiResponse = "ดีใจด้วยครับที่คุณมีความสุข! ความสุขของคุณทำให้ผมมีความสุขด้วยครับ มีอะไรดีๆ มาบอกเล่าไหมครับ?";
        emotion = "ยินดี";
      } else {
        aiResponse = "ขอบคุณสำหรับข้อความครับ ผมได้วิเคราะห์อารมณ์ของคุณแล้ว และพร้อมให้บริการต่อไปครับ";
        emotion = "เป็นกลาง";
      }
      
      setMessages((prev) => [
        ...prev,
        { role: "ai", text: aiResponse },
      ]);
      setDetectedEmotion(emotion);
      setLoading(false);
    }, 1500);
  }

  async function handleSpeak(text: string) {
    const t = text.trim();
    if (!t) return;
    
    // Demo mode - simulate TTS
    setMessages((prev) => [...prev, { role: "ai", text: `🔊 ระบบจำลอง: "${t}" (ในเวอร์ชันจริงจะมีการแปลงข้อความเป็นเสียง)` }]);
  }

  async function handleOtpSend() {
    const phone = otpPhone.trim();
    if (!phone) return;
    
    // Demo mode - simulate OTP sending
    const demoRequestId = `demo_${Date.now()}`;
    setOtpReqId(demoRequestId);
    alert(`🔐 ระบบจำลอง: OTP ได้ถูกส่งไปยังเบอร์ ${phone} แล้ว\nรหัส OTP: 123456 (ในเวอร์ชันจริงจะส่ง SMS จริง)`);
  }

  async function handleOtpVerify() {
    if (!otpReqId) return;
    const code = otpCode.trim();
    
    // Demo mode - simulate OTP verification
    if (code === "123456") {
      setOtpVerified(true);
      alert("✅ ระบบจำลอง: OTP ถูกต้อง! ยืนยันตัวตนสำเร็จแล้ว (ในเวอร์ชันจริงจะเชื่อมต่อกับ AIS OTP API)");
    } else {
      setOtpVerified(false);
      alert("❌ ระบบจำลอง: OTP ไม่ถูกต้อง กรุณาลองใหม่ (รหัสที่ถูกต้องคือ 123456)");
    }
  }

  function handleTranscriptionComplete(text: string, emotion?: string) {
    setTranscriptionText(text);
    setDetectedEmotion(emotion || '');
    setInput(text); // Auto-fill the input field
  }

  // Demo mode - simulate speech transcription
  function handleDemoSpeechTranscription() {
    const demoTexts = [
      "สวัสดีครับ ผมต้องการทดสอบระบบแปลงเสียงเป็นข้อความครับ",
      "ระบบนี้ทำงานได้ดีมากครับ ผมประทับใจมาก",
      "ต้องการทราบข้อมูลเพิ่มเติมเกี่ยวกับบริการครับ",
      "ขอบคุณสำหรับการให้บริการครับ"
    ];
    const randomText = demoTexts[Math.floor(Math.random() * demoTexts.length)];
    const emotions = ["ยินดี", "ประทับใจ", "สนใจ", "ขอบคุณ"];
    const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];
    
    setTranscriptionText(randomText);
    setDetectedEmotion(randomEmotion);
    setInput(randomText);
  }

  function handleCallEnd() {
    // Reset call-related state if needed
    setTranscriptionText("");
    setDetectedEmotion("");
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'chat':
        return (
          <div className="space-y-8 animate-slide-up">
            {/* Chat Header */}
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-[#00A651] to-[#0066CC] rounded-full mb-4">
                <span className="text-2xl">💬</span>
              </div>
                             <h2 className="text-3xl font-bold text-[#1A1A1A] mb-2 font-anuphan">แชทกับ AI</h2>
               <p className="text-[#666] text-lg font-anuphan-medium">สนทนากับ AI Call Center แบบเรียลไทม์</p>
            </div>

            <div className="h-[60vh] overflow-y-auto rounded-3xl p-8 bg-gradient-to-br from-white/95 to-white/80 backdrop-blur-md border border-white/40 shadow-2xl">
                             {messages.length === 0 ? (
                 <div className="text-center py-16">
                   <h3 className="text-2xl font-bold text-[#1A1A1A] mb-2 font-anuphan">ยินดีต้อนรับสู่ AI Call Center</h3>
                   <p className="text-[#666] text-lg font-anuphan-medium">เริ่มต้นการสนทนาด้วยการพิมพ์ข้อความด้านล่าง</p>
                 </div>
               ) : (
                <>
                  {messages.map((m, idx) => (
                    <div key={idx} className="animate-fade-in" style={{animationDelay: `${idx * 0.1}s`}}>
                      <ChatMessage role={m.role} text={m.text} audioUrl={m.audioUrl} autoPlay={!loading && m.role === "ai"} />
                    </div>
                  ))}
                  <SuggestionButtons suggestions={suggestions} onChoose={(t) => setInput(t)} />
                </>
              )}

              {candidates?.length ? (
                <div className="mt-8 animate-scale-in">
                                   <h3 className="text-xl font-bold text-[#1A1A1A] mb-6 flex items-center gap-3">
                   <span className="text-gradient">คำตอบแนะนำ</span>
                 </h3>
                  <div className="grid grid-cols-1 gap-6">
                    {candidates.map((c, i) => (
                      <div key={`cand-${i}`} className="card-jump card-jump-primary p-6 hover-lift animate-fade-in" style={{animationDelay: `${i * 0.1}s`}}>
                        <p className="text-[#1A1A1A] whitespace-pre-wrap text-base mb-4 leading-relaxed">{c}</p>
                        <div className="flex gap-4">
                          <button className="btn-jump-outline text-sm px-6 py-3" onClick={() => setInput(c)} type="button">
                            ✨ ใช้ข้อความนี้
                          </button>
                          <button className="btn-jump-secondary text-sm px-6 py-3" onClick={() => handleSpeak(c)} type="button">
                            🔊 พูดข้อความนี้
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {kb?.length ? (
                <div className="mt-8 animate-scale-in">
                                   <h3 className="text-xl font-bold text-[#1A1A1A] mb-6 flex items-center gap-3">
                   <span className="text-gradient">ความรู้ที่เกี่ยวข้อง</span>
                 </h3>
                  <ul className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {kb.map((k, i) => (
                      <li key={`kb-${i}`} className="card-jump card-jump-secondary p-6 hover-lift animate-fade-in" style={{animationDelay: `${i * 0.1}s`}}>
                        <p className="text-[#1A1A1A] text-base font-semibold mb-3">{k.title}</p>
                        <p className="text-[#666] text-sm leading-relaxed">{k.snippet}</p>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>

            {/* Enhanced Input Section */}
            <div className="bg-gradient-to-r from-[#00A651]/5 to-[#0066CC]/5 rounded-3xl p-6 border border-white/30">
              <div className="flex flex-col lg:flex-row items-stretch gap-4">
                <div className="flex-1 relative">
                  <input
                    className="input-jump w-full text-lg py-4 pl-12"
                    placeholder="พิมพ์ข้อความที่นี่..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                  />
                                                       <div className="absolute inset-y-0 left-4 flex items-center">
                    <span className="text-[#1A1A1A] text-xl">✍</span>
                  </div>
                </div>
                <div className="flex gap-3">
                                     <button
                     className="disabled:opacity-50 disabled:cursor-not-allowed text-lg px-8 py-4 text-white font-medium rounded-xl transition-all duration-500 shadow-2xl hover:shadow-[0_0_50px_rgba(186,218,85,0.4)] transform hover:-translate-y-2 hover:scale-110"
                     style={{
                       background: 'linear-gradient(135deg, rgb(156, 191, 27) 0%, rgb(136, 171, 7) 100%)',
                       boxShadow: '0 0 30px rgba(186, 218, 85, 0.24), 0 8px 32px rgba(186, 218, 85, 0.24)',
                       filter: 'drop-shadow(0 0 20px rgba(186, 218, 85, 0.24))'
                     }}
                     disabled={!canSend}
                     onClick={handleSend}
                   >
                                         {loading ? (
                       <div className="flex items-center gap-2">
                         <div className="spinner-jump"></div>
                         <span>กำลังส่ง...</span>
                       </div>
                     ) : (
                       "ส่งข้อความ"
                     )}
                  </button>
                                     <button
                     className="btn-jump-accent text-lg px-6 py-4"
                     onClick={() => setShowHitl(true)}
                     type="button"
                   >
                     แก้ไข
                   </button>
                   <button
                     className="btn-jump-secondary text-lg px-6 py-4"
                     onClick={() => handleSpeak(input)}
                     type="button"
                   >
                     พูด
                   </button>
                </div>
              </div>
            </div>
          </div>
        );

      case 'speech':
        return (
          <div className="space-y-8 animate-slide-up">
            {/* Speech Header */}
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-[#0066CC] to-[#00A651] rounded-full mb-6">
                <span className="text-3xl">🎤</span>
              </div>
                             <h2 className="text-4xl font-bold text-[#1A1A1A] mb-3 font-anuphan">แปลงเสียงเป็นข้อความ</h2>
               <p className="text-[#666] text-xl max-w-2xl mx-auto leading-relaxed font-anuphan-medium">
                 บันทึกเสียงพูดของคุณและแปลงเป็นข้อความด้วย AI พร้อมการตรวจจับอารมณ์แบบเรียลไทม์
               </p>
            </div>

                        <div className="card-jump card-jump-primary p-10 animate-scale-in">
              <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-[#1A1A1A] mb-3">เริ่มต้นการบันทึกเสียง</h3>
                <p className="text-[#666] text-lg">คลิกปุ่มด้านล่างเพื่อเริ่มบันทึกเสียงของคุณ หรือทดสอบระบบด้วยข้อมูลจำลอง</p>
              </div>
              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <SpeechToText onTranscriptionComplete={handleTranscriptionComplete} />
                <button
                  onClick={handleDemoSpeechTranscription}
                  className="btn-jump-accent text-lg px-8 py-4 transform hover:scale-105 transition-all duration-300"
                >
                  🎯 ทดสอบด้วยข้อมูลจำลอง
                </button>
              </div>
            </div>

            {transcriptionText && (
              <div className="card-jump card-jump-secondary p-10 animate-fade-in">
                                 <div className="text-center mb-8">
                   <h3 className="text-3xl font-bold text-[#1A1A1A] mb-3">ผลลัพธ์การแปลงเสียง</h3>
                   <p className="text-[#666] text-lg">AI ได้แปลงเสียงของคุณเป็นข้อความแล้ว</p>
                 </div>
                
                <div className="space-y-6 max-w-3xl mx-auto">
                                     <div className="bg-gradient-to-r from-white/90 to-white/80 p-8 rounded-2xl border border-[#0066CC]/30 shadow-lg">
                     <div className="flex items-center gap-3 mb-4">
                       <h4 className="text-xl font-semibold text-[#1A1A1A]">ข้อความที่แปลงได้</h4>
                     </div>
                     <p className="text-[#1A1A1A] text-xl leading-relaxed">{transcriptionText}</p>
                   </div>
                  
                  {detectedEmotion && (
                                         <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#0066CC]/30 shadow-lg">
                       <div className="flex items-center gap-3 mb-3">
                         <h4 className="text-lg font-semibold text-[#1A1A1A]">อารมณ์ที่ตรวจพบ</h4>
                       </div>
                       <div className="inline-flex items-center gap-2 bg-gradient-to-r from-[#0066CC]/20 to-[#00A651]/20 px-4 py-2 rounded-full border border-[#0066CC]/30">
                         <span className="text-[#1A1A1A] font-semibold">{detectedEmotion}</span>
                       </div>
                     </div>
                  )}
                  
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                                                              <button
                       onClick={() => setActiveTab('chat')}
                       className="text-lg px-8 py-4 text-white font-medium rounded-xl transition-all duration-500 shadow-2xl hover:shadow-[0_0_50px_rgba(186,218,85,0.4)] transform hover:-translate-y-2 hover:scale-110"
                       style={{
                         background: 'linear-gradient(135deg, rgb(156, 191, 27) 0%, rgb(136, 171, 7) 100%)',
                         boxShadow: '0 0 30px rgba(186, 218, 85, 0.24), 0 8px 32px rgba(186, 218, 85, 0.24)',
                         filter: 'drop-shadow(0 0 20px rgba(186, 218, 85, 0.24))'
                       }}
                     >
                        ใช้ข้อความนี้ในการแชท
                      </button>
                     <button
                       onClick={() => handleSpeak(transcriptionText)}
                       className="btn-jump-secondary text-lg px-8 py-4 transform hover:scale-105 transition-all duration-300"
                     >
                       ฟังเสียงที่แปลงได้
                     </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        );

      case 'call':
        return (
          <div className="space-y-8 animate-slide-up">
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-[#00A651] to-[#0066CC] rounded-full mb-6">
                <span className="text-3xl">📞</span>
              </div>
              <h2 className="text-4xl font-bold text-[#1A1A1A] mb-3 font-anuphan">การสนทนาสองทาง</h2>
              <p className="text-[#666] text-xl max-w-2xl mx-auto leading-relaxed font-anuphan-medium">
                ทดสอบการสนทนากับ AI Call Center แบบสองทาง พร้อมการแปลงเสียงและตรวจจับอารมณ์
              </p>
            </div>

            <div className="card-jump card-jump-primary p-10 animate-scale-in">
              <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-[#1A1A1A] mb-3">ระบบสนทนาสองทาง</h3>
                <p className="text-[#666] text-lg">ทดสอบการสนทนากับ AI แบบสองทาง พร้อมการแปลงเสียงและตรวจจับอารมณ์</p>
              </div>
              
              <div className="space-y-6">
                <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#00A651]/30 shadow-lg">
                  <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4">สถานะการเชื่อมต่อ</h4>
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-3 h-3 bg-[#00A651] rounded-full animate-pulse"></div>
                    <span className="text-[#1A1A1A] font-medium">พร้อมใช้งาน</span>
                  </div>
                  <p className="text-[#666] text-sm">ระบบพร้อมสำหรับการสนทนาสองทางกับ AI</p>
                </div>

                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <button
                    onClick={() => alert("🎯 ระบบจำลอง: เริ่มการสนทนาสองทาง\nในเวอร์ชันจริงจะมีการเชื่อมต่อกับ AI Call Center จริง")}
                    className="btn-jump-primary text-lg px-8 py-4 transform hover:scale-105 transition-all duration-300"
                  >
                    📞 เริ่มการสนทนา
                  </button>
                  <button
                    onClick={() => alert("🎯 ระบบจำลอง: หยุดการสนทนา\nในเวอร์ชันจริงจะมีการปิดการเชื่อมต่อกับ AI Call Center")}
                    className="btn-jump-secondary text-lg px-8 py-4 transform hover:scale-105 transition-all duration-300"
                  >
                    ⏹️ หยุดการสนทนา
                  </button>
                </div>

                <div className="bg-gradient-to-r from-[#00A651]/10 to-[#0066CC]/10 rounded-2xl p-6 border border-[#00A651]/20">
                  <p className="text-sm text-[#666] text-center">
                    <strong>หมายเหตุ:</strong> ระบบนี้เป็นเดโม่สำหรับการทดสอบ UI และ UX — ในเวอร์ชันจริงจะมีการเชื่อมต่อกับ AI Call Center จริง
                  </p>
                </div>
              </div>
            </div>
          </div>
        );

      case 'enhanced':
        return (
          <div className="space-y-6">
            <div className="card-jump card-jump-secondary p-8">
              <h2 className="text-2xl font-bold text-[#1A1A1A] mb-6 flex items-center gap-3">
                <span className="text-[#0066CC] text-3xl">🚀</span>
                แชทขั้นสูง
              </h2>
              <p className="text-[#666] mb-6">แชทที่มีการตรวจจับอารมณ์และตอบสนองตามบริบท</p>
              
              <div className="h-[60vh] overflow-y-auto rounded-2xl p-6 bg-white/80 backdrop-blur-sm border border-[#0066CC]/20 shadow-lg">
                {messages.map((m, idx) => (
                  <ChatMessage key={idx} role={m.role} text={m.text} audioUrl={m.audioUrl} autoPlay={!loading && m.role === "ai"} />
                ))}
              </div>

              <div className="flex items-center gap-3 mt-6">
                <input
                  className="input-jump flex-1"
                  placeholder="พิมพ์ข้อความที่นี่ (จะมีการตรวจจับอารมณ์)"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                />
                <button
                  className="btn-jump-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                  disabled={!input.trim() || loading}
                  onClick={handleEnhancedChat}
                >
                  {loading ? (
                    <div className="flex items-center gap-2">
                      <div className="spinner-jump"></div>
                      <span>กำลังประมวลผล...</span>
                    </div>
                  ) : (
                    "ส่งข้อความขั้นสูง"
                  )}
                </button>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#E8F5E8] via-[#F8F9FA] to-[#E6F3FF]">
      {/* Enhanced Hero Section with JUMP THAILAND Theme */}
      <div 
        className="relative py-20 overflow-hidden"
        style={{
          backgroundImage: `url('https://static.wixstatic.com/media/6e391d_4b6ebe4961af4f7ea64c1909080884f9~mv2.jpg/v1/fill/w_980,h_937,al_t,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/6e391d_4b6ebe4961af4f7ea64c1909080884f9~mv2.jpg')`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundAttachment: 'fixed',
          backgroundRepeat: 'no-repeat'
        }}
      >
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0" style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%231A1A1A' fill-opacity='0.4'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          }}></div>
        </div>
        
        {/* Floating Elements */}
        <div className="absolute top-10 left-10 w-20 h-20 bg-white/10 rounded-full blur-xl animate-float"></div>
        <div className="absolute top-20 right-20 w-32 h-32 bg-white/10 rounded-full blur-xl animate-float" style={{animationDelay: '2s'}}></div>
        <div className="absolute bottom-20 left-1/4 w-24 h-24 bg-white/10 rounded-full blur-xl animate-float" style={{animationDelay: '4s'}}></div>
        
        <div className="max-w-7xl mx-auto px-4 relative z-10">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Content */}
            <div className="text-center lg:text-left">
              <div className="inline-flex items-center gap-2 bg-white/20 backdrop-blur-sm rounded-full px-6 py-2 mb-8 border border-white/30">
                <span className="w-2 h-2 bg-[#FFD700] rounded-full animate-pulse"></span>
                <span className="text-sm font-medium text-[#1A1A1A]">AI Innovation Platform - DEMO MODE</span>
              </div>
              
              <h1 className="text-5xl lg:text-6xl xl:text-7xl font-bold mb-6 leading-tight font-anuphan">
                <span className="block text-[#1A1A1A]">AI Call Center</span>
                <span className="block text-[#FFD700]">System</span>
              </h1>
              
              <p className="text-xl lg:text-2xl text-[#1A1A1A] mb-8 max-w-2xl lg:max-w-none leading-relaxed font-anuphan-medium">
                ระบบศูนย์บริการ AI แบบครบวงจรสำหรับผู้พิการทางการได้ยิน
                <br className="hidden sm:block" />
                <span className="text-[#FFD700] font-semibold">ขับเคลื่อนอนาคตด้วยนวัตกรรมในยุคปัญญาประดิษฐ์</span>
              </p>
              
                             <div className="flex flex-wrap items-center justify-center lg:justify-start gap-4 mb-8">
                 <div className="badge-jump badge-jump-accent text-sm px-4 py-2 text-[#1A1A1A]">
                   POC v2.0
                 </div>
                 <div className="badge-jump badge-jump-accent text-sm px-4 py-2 text-[#1A1A1A]">
                   JUMP THAILAND Theme
                 </div>
                 <div className="badge-jump badge-jump-accent text-sm px-4 py-2 text-[#1A1A1A]">
                   AI-Powered
                 </div>
               </div>
              
              <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4">
                                                  <button 
                   className="text-lg px-8 py-4 text-white font-medium rounded-xl transition-all duration-500 shadow-2xl hover:shadow-[0_0_50px_rgba(186,218,85,0.4)] transform hover:-translate-y-2 hover:scale-110"
                   style={{
                     background: 'linear-gradient(135deg, rgb(156, 191, 27) 0%, rgb(136, 171, 7) 100%)',
                     boxShadow: '0 0 30px rgba(186, 218, 85, 0.24), 0 8px 32px rgba(186, 218, 85, 0.24)',
                     filter: 'drop-shadow(0 0 20px rgba(186, 218, 85, 0.24))'
                   }}
                 >
                    เริ่มต้นใช้งาน
                  </button>
                 <button className="border-2 border-[#1A1A1A]/30 text-[#1A1A1A] hover:bg-[#1A1A1A]/10 px-8 py-4 rounded-xl transition-all duration-300 backdrop-blur-sm">
                   เรียนรู้เพิ่มเติม
                 </button>
              </div>
            </div>
            
            {/* Right Image */}
            <div className="relative flex justify-center lg:justify-end">
              <div className="relative">
                                 {/* Main Image Placeholder */}
                                   <div className="relative w-80 h-80 lg:w-96 lg:h-96 bg-gradient-to-br from-white/20 to-white/10 rounded-3xl shadow-2xl border border-white/30 backdrop-blur-sm flex items-center justify-center">
                    <Image
                      src="/logo.png"
                      alt="SIANG JAI Logo"
                      width={200}
                      height={200}
                      className="object-contain"
                    />
                  </div>
                
                {/* Floating Stats */}
                <div className="absolute -top-4 -left-4 bg-white/95 backdrop-blur-sm rounded-2xl p-4 shadow-xl border border-white/30">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-[#00A651]">99.9%</div>
                    <div className="text-sm text-[#666]">ความแม่นยำ</div>
                  </div>
                </div>
                
                <div className="absolute -bottom-4 -right-4 bg-white/95 backdrop-blur-sm rounded-2xl p-4 shadow-xl border border-white/30">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-[#0066CC]">24/7</div>
                    <div className="text-sm text-[#666]">บริการ</div>
                  </div>
                </div>
                
                {/* Decorative Elements */}
                <div className="absolute top-1/2 -left-8 w-16 h-16 bg-[#FFD700]/20 rounded-full blur-lg animate-pulse"></div>
                <div className="absolute bottom-1/2 -right-8 w-20 h-20 bg-[#00A651]/20 rounded-full blur-lg animate-pulse" style={{animationDelay: '1s'}}></div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Wave Separator */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg className="w-full h-16 text-[#1A1A1A]" viewBox="0 0 1200 120" preserveAspectRatio="none">
            <path d="M0,0V46.29c47.79,22.2,103.59,32.17,158,28,70.36-5.37,136.33-33.31,206.8-37.5C438.64,32.43,512.34,53.67,583,72.05c69.27,18,138.3,24.88,209.4,13.08,36.15-6,69.85-17.84,104.45-29.34C989.49,25,1113-14.29,1200,52.47V0Z" opacity=".25" fill="currentColor"></path>
            <path d="M0,0V15.81C13,36.92,27.64,56.86,47.69,72.05,99.41,111.27,165,111,224.58,91.58c31.15-10.15,60.09-26.07,89.67-39.8,40.92-19,84.73-46,130.83-49.67,36.26-2.85,70.9,9.42,98.6,31.56,31.77,25.39,62.32,62,103.63,73,40.44,10.71,81.35-6.69,119.13-24.28s75.16-39,116.92-43.05c59.73-5.85,113.28,22.88,168.9,38.84,30.2,8.66,59,6.17,87.09-7.5,22.43-10.89,48-26.93,65.6-42.72C1006.38,59.18,1031.56,61.29,1058,60.81c27.9-.5,56.19-9.69,82.84-21.56,31.84-14.27,65.27-35.85,92.8-54.59C1200,0,1200,0,1200,0Z" opacity=".5" fill="currentColor"></path>
            <path d="M0,0V5.63C149.93,59,314.09,71.32,475.83,42.57c43-7.64,84.23-20.12,127.61-26.46,59-8.63,112.48,12.24,165.56,35.4C827.93,77.22,886,95.24,951.2,90c86.53-7,172.46-45.71,248.8-84.81V0Z" fill="currentColor"></path>
          </svg>
        </div>
      </div>

             <div className="max-w-7xl mx-auto px-4 py-12">
         {/* Enhanced Tab Navigation with Hero Theme */}
         <div className="relative mb-16">
           <div className="absolute inset-0 bg-gradient-to-r from-[#00A651]/10 to-[#0066CC]/10 rounded-3xl blur-3xl"></div>
           <div className="relative bg-white/95 backdrop-blur-md rounded-3xl p-6 shadow-2xl border border-white/40">
             <div className="text-center mb-6">
               <h2 className="text-3xl font-bold text-[#1A1A1A] mb-2 font-anuphan">บริการหลักของเรา</h2>
               <p className="text-[#666] text-lg font-anuphan-medium">เลือกบริการที่ต้องการใช้งาน</p>
             </div>
             <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
               {[
                 { id: 'chat', label: 'แชทพื้นฐาน', icon: '💬', color: 'from-[#00A651] to-[#00A651]', desc: 'แชทพื้นฐานกับ AI' },
                 { id: 'speech', label: 'แปลงเสียง', icon: '🎤', color: 'from-[#0066CC] to-[#0066CC]', desc: 'แปลงเสียงเป็นข้อความ' },
                 { id: 'call', label: 'สนทนาสองทาง', icon: '📞', color: 'from-[#00A651] to-[#0066CC]', desc: 'สนทนากับ AI แบบสองทาง' },
                 { id: 'enhanced', label: 'แชทขั้นสูง', icon: '🚀', color: 'from-[#0066CC] to-[#00A651]', desc: 'แชทขั้นสูงพร้อมตรวจจับอารมณ์' }
               ].map((tab) => (
                 <button
                   key={tab.id}
                   onClick={() => setActiveTab(tab.id as TabType)}
                   className={`relative p-6 rounded-2xl text-sm font-medium transition-all duration-500 group overflow-hidden ${
                     activeTab === tab.id
                       ? `bg-gradient-to-r ${tab.color} text-[#1A1A1A] shadow-xl transform scale-105`
                       : 'bg-white/80 text-[#666] hover:text-[#1A1A1A] hover:bg-white hover:scale-105 hover:shadow-lg'
                   }`}
                 >
                   <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                   <div className="relative flex flex-col items-center space-y-3">
                     <div className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl transition-all duration-300 ${
                       activeTab === tab.id 
                         ? 'bg-[#1A1A1A]/20 backdrop-blur-sm' 
                         : 'bg-gradient-to-br from-[#00A651]/10 to-[#0066CC]/10'
                     }`}>
                       <span className="group-hover:scale-110 transition-transform duration-300">{tab.icon}</span>
                     </div>
                     <div className="text-center">
                       <span className="font-semibold text-base block">{tab.label}</span>
                       <span className={`text-xs opacity-75 mt-1 block ${activeTab === tab.id ? 'text-[#1A1A1A]' : 'text-[#666]'}`}>
                         {tab.desc}
                       </span>
                     </div>
                   </div>
                 </button>
               ))}
             </div>
           </div>
         </div>

                 {/* Enhanced Tab Content with Hero Theme */}
         <div className="relative mb-16">
           <div className="absolute inset-0 bg-gradient-to-r from-[#00A651]/5 to-[#0066CC]/5 rounded-3xl blur-3xl"></div>
           <div className="relative bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl border border-white/40 overflow-hidden">
             <div className="bg-gradient-to-r from-[#00A651]/10 to-[#0066CC]/10 p-6 border-b border-white/30">
               <h3 className="text-2xl font-bold text-[#1A1A1A] font-anuphan">พื้นที่ใช้งาน</h3>
               <p className="text-[#666] font-anuphan-medium">เลือกบริการและเริ่มใช้งานได้เลย</p>
             </div>
             <div className="p-8">
               {renderTabContent()}
             </div>
           </div>
         </div>

                 {/* Enhanced OTP Section with Hero Theme */}
         <div className="relative mb-16">
           <div className="absolute inset-0 bg-gradient-to-r from-[#00A651]/10 to-[#0066CC]/10 rounded-3xl blur-3xl"></div>
           <div className="relative bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl border border-white/40 overflow-hidden">
             <div className="bg-gradient-to-r from-[#00A651]/10 to-[#0066CC]/10 p-8 border-b border-white/30">
               <div className="text-center">
                 <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-[#00A651] to-[#0066CC] rounded-full mb-6 shadow-xl">
                   <span className="text-4xl">🔐</span>
                 </div>
                 <h3 className="text-3xl font-bold text-[#1A1A1A] mb-3 font-anuphan">ยืนยันตัวตน (OTP Demo)</h3>
                 <p className="text-[#666] text-lg font-anuphan-medium">ระบบยืนยันตัวตนด้วยรหัส OTP แบบปลอดภัย - โหมดเดโม</p>
               </div>
             </div>
             <div className="p-10">
               <div className="space-y-6 max-w-2xl mx-auto">
              <div className="flex flex-col md:flex-row items-stretch md:items-center gap-4">
                <div className="flex-1 relative">
                  <input 
                    className="input-jump w-full text-center text-lg py-4" 
                    placeholder="เบอร์โทรศัพท์" 
                    value={otpPhone} 
                    onChange={(e) => setOtpPhone(e.target.value)} 
                  />
                  <div className="absolute inset-y-0 left-4 flex items-center">
                    <span className="text-[#1A1A1A] text-xl">📱</span>
                  </div>
                </div>
                                 <button 
                   className="text-lg px-8 py-4 text-white font-medium rounded-xl transition-all duration-500 shadow-2xl hover:shadow-[0_0_50px_rgba(186,218,85,0.4)] transform hover:-translate-y-2 hover:scale-110" 
                   style={{
                     background: 'linear-gradient(135deg, rgb(156, 191, 27) 0%, rgb(136, 171, 7) 100%)',
                     boxShadow: '0 0 30px rgba(186, 218, 85, 0.24), 0 8px 32px rgba(186, 218, 85, 0.24)',
                     filter: 'drop-shadow(0 0 20px rgba(186, 218, 85, 0.24))'
                   }}
                   onClick={handleOtpSend}
                   type="button"
                 >
                   ส่ง OTP
                 </button>
              </div>
              
              {otpReqId ? (
                <div className="flex flex-col md:flex-row items-stretch md:items-center gap-4">
                  <div className="flex-1 relative">
                    <input 
                      className="input-jump w-full text-center text-lg py-4" 
                      placeholder="รหัส 6 หลัก" 
                      value={otpCode} 
                      onChange={(e) => setOtpCode(e.target.value)} 
                    />
                    <div className="absolute inset-y-0 left-4 flex items-center">
                      <span className="text-[#1A1A1A] text-xl">🔢</span>
                    </div>
                  </div>
                  <button 
                    className="btn-jump-secondary text-lg px-8 py-4 transform hover:scale-105 transition-all duration-300" 
                    onClick={handleOtpVerify}
                    type="button"
                  >
                    ยืนยัน
                  </button>
                                     {otpVerified ? (
                     <div className="badge-jump badge-jump-primary text-lg px-6 py-3 text-[#1A1A1A]">
                       ยืนยันแล้ว
                     </div>
                   ) : null}
                </div>
              ) : null}
              
              <div className="bg-gradient-to-r from-[#00A651]/10 to-[#0066CC]/10 rounded-2xl p-6 border border-[#00A651]/20">
                               <p className="text-sm text-[#666] text-center">
                 <strong>หมายเหตุ:</strong> เดโม่นี้แสดงรหัส OTP ในฝั่งเซิร์ฟเวอร์ log (mock) — เปลี่ยนเป็น AIS OTP ได้ทันทีเมื่อมี credentials
               </p>
              </div>
               </div>
             </div>
           </div>
         </div>

        {/* Enhanced Footer */}
        <div className="mt-20 relative">
          <div className="absolute inset-0 bg-gradient-to-r from-[#00A651]/5 to-[#0066CC]/5 rounded-3xl blur-3xl"></div>
          <div className="relative bg-gradient-to-br from-white/95 to-white/80 backdrop-blur-md rounded-3xl p-12 shadow-2xl border border-white/40">
            <div className="text-center mb-10">
              <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-[#00A651] to-[#0066CC] rounded-full mb-8">
                <span className="text-4xl">🎯</span>
              </div>
              <h4 className="text-4xl font-bold text-[#1A1A1A] mb-6">
                ภารกิจ "คิดเผื่อ"
              </h4>
              <p className="text-xl text-[#666] mb-8 max-w-3xl mx-auto leading-relaxed">
                ขับเคลื่อนอนาคตด้วยนวัตกรรมในยุคปัญญาประดิษฐ์ เพื่อเพิ่มศักยภาพให้กับผู้สูงอายุ หรือ คนพิการ
              </p>
              
                             <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
                 <div className="bg-gradient-to-br from-[#00A651]/10 to-[#00A651]/5 rounded-2xl p-6 border border-[#00A651]/20">
                   <h5 className="text-lg font-semibold text-[#1A1A1A] mb-2">AI Innovation</h5>
                   <p className="text-[#666] text-sm">เทคโนโลยีปัญญาประดิษฐ์ล้ำสมัย</p>
                 </div>
                 <div className="bg-gradient-to-br from-[#0066CC]/10 to-[#0066CC]/5 rounded-2xl p-6 border border-[#0066CC]/20">
                   <h5 className="text-lg font-semibold text-[#1A1A1A] mb-2">Inclusive Technology</h5>
                   <p className="text-[#666] text-sm">เทคโนโลยีที่เข้าถึงได้ทุกคน</p>
                 </div>
                 <div className="bg-gradient-to-br from-[#FFD700]/10 to-[#FFD700]/5 rounded-2xl p-6 border border-[#FFD700]/20">
                   <h5 className="text-lg font-semibold text-[#1A1A1A] mb-2">Future Ready</h5>
                   <p className="text-[#666] text-sm">พร้อมรับมือกับอนาคต</p>
                 </div>
               </div>
              
              <div className="flex flex-wrap items-center justify-center gap-4 text-sm">
                <span className="badge-jump badge-jump-primary text-base px-6 py-3 text-[#1A1A1A]">AIS JUMP THAILAND</span>
                <span className="badge-jump badge-jump-accent text-base px-6 py-3 text-[#1A1A1A]">AI Innovation</span>
                <span className="badge-jump badge-jump-secondary text-base px-6 py-3 text-[#1A1A1A]">Inclusive Technology</span>
              </div>
            </div>
            
            <div className="border-t border-white/30 pt-8">
              <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-[#00A651] to-[#0066CC] rounded-full"></div>
                  <span className="text-[#1A1A1A] font-medium">© 2024 AI Call Center System</span>
                </div>
                <div className="flex items-center gap-6">
                  <span className="text-[#1A1A1A] hover:text-[#00A651] transition-colors cursor-pointer">นโยบายความเป็นส่วนตัว</span>
                  <span className="text-[#1A1A1A] hover:text-[#0066CC] transition-colors cursor-pointer">เงื่อนไขการใช้งาน</span>
                  <span className="text-[#1A1A1A] hover:text-[#FFD700] transition-colors cursor-pointer">ติดต่อเรา</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <HITLModal
        open={showHitl}
        onClose={() => setShowHitl(false)}
        originalMessage={lastUserMessage}
        onSubmit={async (corrected) => {
          if (!corrected.trim()) return;
          await postFeedback(lastUserMessage, corrected.trim());
          await handleSpeak(corrected.trim());
        }}
      />

      {/* Enhanced Background Decorations */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        {/* Main Floating Elements */}
        <div className="absolute -top-40 -left-40 w-80 h-80 bg-gradient-to-br from-[#00A651]/20 to-[#00A651]/5 rounded-full blur-3xl animate-float"></div>
        <div className="absolute -bottom-40 -right-40 w-80 h-80 bg-gradient-to-br from-[#0066CC]/20 to-[#0066CC]/5 rounded-full blur-3xl animate-float" style={{animationDelay: '2s'}}></div>
        <div className="absolute top-1/2 left-1/4 w-60 h-60 bg-gradient-to-br from-[#FFD700]/20 to-[#FFD700]/5 rounded-full blur-3xl animate-float" style={{animationDelay: '4s'}}></div>
        
        {/* Additional Decorative Elements */}
        <div className="absolute top-1/4 right-1/4 w-40 h-40 bg-gradient-to-br from-[#00A651]/15 to-[#0066CC]/15 rounded-full blur-2xl animate-float" style={{animationDelay: '1s'}}></div>
        <div className="absolute bottom-1/3 left-1/3 w-32 h-32 bg-gradient-to-br from-[#FFD700]/15 to-[#00A651]/15 rounded-full blur-2xl animate-float" style={{animationDelay: '3s'}}></div>
        
        {/* Grid Pattern */}
        <div className="absolute inset-0 opacity-5">
          <div className="absolute inset-0" style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%231A1A1A' fill-opacity='0.1'%3E%3Cpath d='M0 0h100v100H0z'/%3E%3C/g%3E%3Cg fill='%231A1A1A' fill-opacity='0.1'%3E%3Cpath d='M0 0h50v50H0zM50 50h50v50H50z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          }}></div>
        </div>
        
        {/* Particle Effects */}
        <div className="absolute top-20 left-20 w-2 h-2 bg-[#00A651] rounded-full animate-pulse"></div>
        <div className="absolute top-40 right-40 w-1 h-1 bg-[#0066CC] rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></div>
        <div className="absolute bottom-40 left-40 w-1.5 h-1.5 bg-[#FFD700] rounded-full animate-pulse" style={{animationDelay: '1s'}}></div>
        <div className="absolute bottom-20 right-20 w-1 h-1 bg-[#00A651] rounded-full animate-pulse" style={{animationDelay: '1.5s'}}></div>
      </div>
    </main>
  );
}
