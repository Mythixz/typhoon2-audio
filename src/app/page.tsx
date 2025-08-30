"use client";

import React, { useMemo, useRef, useState } from "react";
import ChatMessage from "@/components/ChatMessage";
import SuggestionButtons from "@/components/SuggestionButtons";
import HITLModal from "@/components/HITLModal";
import SpeechToText from "@/components/SpeechToText";
import TwoWayCall from "@/components/TwoWayCall";
import ChatTab from "@/components/ChatTab";
import SpeechTab from "@/components/SpeechTab";
import CallTab from "@/components/CallTab";
import EnhancedTab from "@/components/EnhancedTab";
import SupervisorTab from "@/components/SupervisorTab";
import CollaborativeTab from "@/components/CollaborativeTab";
import CRMTab from "@/components/CRMTab";
import RAGTab from "@/components/RAGTab";
import { 
  postChat, 
  postFeedback, 
  postSpeak, 
  postEnhancedChat,
  type ChatResponse 
} from "@/lib/api";
import { TabType, type ChatMessage as ChatMessageType, type KnowledgeItem } from "@/types";
import Image from "next/image";
import logoImage from '@/assets/images/logo.png';

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<TabType>('chat');
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
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
  const [kb, setKb] = useState<KnowledgeItem[]>([
    {
      title: "AI Call Center System",
      snippet: "ระบบ AI Empowerment เพื่อสร้างงาน Call Center คุณภาพสูงสำหรับผู้พิการ"
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
          <ChatTab
            messages={messages}
            loading={loading}
            suggestions={suggestions}
            candidates={candidates}
            kb={kb}
            input={input}
            setInput={setInput}
            handleSend={handleSend}
            handleSpeak={handleSpeak}
            setShowHitl={setShowHitl}
          />
        );

      case 'speech':
        return (
          <SpeechTab
            transcriptionText={transcriptionText}
            detectedEmotion={detectedEmotion}
            handleTranscriptionComplete={handleTranscriptionComplete}
            handleDemoSpeechTranscription={handleDemoSpeechTranscription}
            setActiveTab={setActiveTab}
            handleSpeak={handleSpeak}
          />
        );

      case 'call':
        return <CallTab />;

      case 'enhanced':
        return (
          <EnhancedTab
            messages={messages}
            loading={loading}
            input={input}
            setInput={setInput}
            handleEnhancedChat={handleEnhancedChat}
          />
        );

      case 'supervisor':
        return <SupervisorTab />;

      case 'collaborative':
        return <CollaborativeTab />;

      case 'crm':
        return <CRMTab />;

      case 'rag':
        return <RAGTab />;

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
              
              {/* Accessibility Focus */}
              <div className="bg-gradient-to-r from-[#00A651]/20 to-[#0066CC]/20 rounded-2xl p-4 mb-6 border border-[#00A651]/30">
                <div className="text-center">
                  <h3 className="text-lg font-semibold text-[#1A1A1A] mb-2">🎯 ระบบออกแบบเพื่อผู้พิการ</h3>
                  <p className="text-[#1A1A1A] text-sm">
                    <strong>ผู้พิการทางการได้ยิน</strong> • <strong>ผู้พิการทางการสื่อสาร</strong> • <strong>ผู้พิการทางสติปัญญา</strong>
                  </p>
                </div>
              </div>
              
              <h1 className="text-5xl lg:text-6xl xl:text-7xl font-bold mb-6 leading-tight font-anuphan">
                <span className="block text-[#1A1A1A]">AI Call Center</span>
                <span className="block text-[#FFD700]">System</span>
              </h1>
              
              <p className="text-xl lg:text-2xl text-[#1A1A1A] mb-8 max-w-2xl lg:max-w-none leading-relaxed font-anuphan-medium">
              ระบบ AI Empowerment เพื่อสร้างงาน Call Center คุณภาพสูงสำหรับผู้พิการ
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
                      src={logoImage}
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
               
               {/* Accessibility Guide */}
               <div className="bg-gradient-to-r from-[#FFD700]/20 to-[#00A651]/20 rounded-xl p-4 mt-4 border border-[#FFD700]/30">
                 <h4 className="text-sm font-semibold text-[#1A1A1A] mb-2">🎯 ระบบออกแบบเพื่อผู้พิการแต่ละกลุ่ม</h4>
                 <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-xs">
                   <div className="text-[#00A651] font-medium">👂 ผู้พิการทางการได้ยิน</div>
                   <div className="text-[#0066CC] font-medium">💬 ผู้พิการทางการสื่อสาร</div>
                   <div className="text-[#FFD700] font-medium">🧠 ผู้พิการทางสติปัญญา</div>
                 </div>
               </div>
             </div>
             <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
               {[
                 { 
                   id: 'chat', 
                   label: 'แชทพื้นฐาน', 
                   icon: '💬', 
                   color: 'from-[#00A651] to-[#00A651]', 
                   desc: 'พิมพ์ข้อความโต้ตอบกับ AI',
                   benefit: 'เหมาะสำหรับผู้พิการทางการได้ยิน',
                   accessibility: '👂 ผู้พิการทางการได้ยิน • 💬 ผู้พิการทางการสื่อสาร'
                 },
                 { 
                   id: 'speech', 
                   label: 'แปลงเสียง', 
                   icon: '🎤', 
                   color: 'from-[#0066CC] to-[#0066CC]', 
                   desc: 'แปลงเสียงพูดเป็นข้อความ',
                   benefit: 'ช่วยผู้พิการทางการได้ยินเข้าใจลูกค้า',
                   accessibility: '👂 ผู้พิการทางการได้ยิน • 💬 ผู้พิการทางการสื่อสาร'
                 },
                 { 
                   id: 'call', 
                   label: 'สนทนาสองทาง', 
                   icon: '📞', 
                   color: 'from-[#00A651] to-[#0066CC]', 
                   desc: 'สนทนากับ AI แบบสองทาง',
                   benefit: 'รองรับการสื่อสารแบบครบวงจร',
                   accessibility: '👂 ผู้พิการทางการได้ยิน • 💬 ผู้พิการทางการสื่อสาร • 🧠 ผู้พิการทางสติปัญญา'
                 },
                 { 
                   id: 'enhanced', 
                   label: 'แชทขั้นสูง', 
                   icon: '🚀', 
                   color: 'from-[#0066CC] to-[#00A651]', 
                   desc: 'ตรวจจับอารมณ์และแนะนำคำตอบ',
                   benefit: 'ช่วยผู้พิการทางสติปัญญา',
                   accessibility: '🧠 ผู้พิการทางสติปัญญา • 👂 ผู้พิการทางการได้ยิน'
                 },
                 { 
                   id: 'supervisor', 
                   label: 'AI Supervisor', 
                   icon: '👨‍💼', 
                   color: 'from-[#FFD700] to-[#FF6B35]', 
                   desc: 'ควบคุมและดูแล AI Call Center',
                   benefit: 'คนพิการเป็นผู้ควบคุม AI แทนการเป็นพนักงาน',
                   accessibility: '👂 ผู้พิการทางการได้ยิน • 💬 ผู้พิการทางการสื่อสาร • 🧠 ผู้พิการทางสติปัญญา'
                 },
                 { 
                   id: 'collaborative', 
                   label: 'Collaborative Training', 
                   icon: '🤝', 
                   color: 'from-[#9C27B0] to-[#E91E63]', 
                   desc: 'เทรน AI ร่วมกันระหว่าง AIS กับผู้พิการ',
                   benefit: 'พนักงาน AIS และผู้พิการเทรน AI ร่วมกัน',
                   accessibility: '👂 ผู้พิการทางการได้ยิน • 💬 ผู้พิการทางการสื่อสาร • 🧠 ผู้พิการทางสติปัญญา'
                 },
                 { 
                   id: 'crm', 
                   label: 'CRM System', 
                   icon: '📊', 
                   color: 'from-[#FF6B6B] to-[#4ECDC4]', 
                   desc: 'ระบบจัดการลูกค้าแบบครบวงจร',
                   benefit: 'จัดการข้อมูลลูกค้าและความต้องการพิเศษ',
                   accessibility: '👂 ผู้พิการทางการได้ยิน • 💬 ผู้พิการทางการสื่อสาร • 🧠 ผู้พิการทางสติปัญญา'
                 },
                 { 
                   id: 'rag', 
                   label: 'RAG System', 
                   icon: '🧠', 
                   color: 'from-[#6C5CE7] to-[#A29BFE]', 
                   desc: 'ระบบค้นหาความรู้แบบอัจฉริยะ',
                   benefit: 'ตอบคำถามด้วยความรู้จากฐานข้อมูล',
                   accessibility: '👂 ผู้พิการทางการได้ยิน • 💬 ผู้พิการทางการสื่อสาร • 🧠 ผู้พิการทางสติปัญญา'
                 }
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
                       <span className={`text-xs opacity-60 mt-1 block ${activeTab === tab.id ? 'text-[#1A1A1A]/80' : 'text-[#666]/80]'}`}>
                         {tab.benefit}
                       </span>
                       <span className={`text-xs opacity-50 mt-1 block ${activeTab === tab.id ? 'text-[#1A1A1A]/60' : 'text-[#666]/60]'}`}>
                         {tab.accessibility}
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
              
              {/* Complete Feature Overview */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-10">
                <div className="bg-gradient-to-br from-[#00A651]/10 to-[#00A651]/5 rounded-2xl p-6 border border-[#00A651]/20">
                  <h5 className="text-lg font-semibold text-[#1A1A1A] mb-3">💬 Text-Based System</h5>
                  <p className="text-[#666] text-sm mb-2">พิมพ์ข้อความโต้ตอบกับ AI</p>
                  <p className="text-[#00A651] text-xs font-medium">เหมาะสำหรับผู้พิการทางการได้ยิน</p>
                  <div className="mt-2 text-xs text-[#666]">👂 ผู้พิการทางการได้ยิน • 💬 ผู้พิการทางการสื่อสาร</div>
                </div>
                <div className="bg-gradient-to-br from-[#0066CC]/10 to-[#0066CC]/5 rounded-2xl p-6 border border-[#0066CC]/20">
                  <h5 className="text-lg font-semibold text-[#1A1A1A] mb-3">🎤 Speech-to-Text</h5>
                  <p className="text-[#666] text-sm mb-2">แปลงเสียงลูกค้าเป็นข้อความ</p>
                  <p className="text-[#00A651] text-xs font-medium">ช่วยผู้พิการทางการได้ยินเข้าใจลูกค้า</p>
                  <div className="mt-2 text-xs text-[#666]">👂 ผู้พิการทางการได้ยิน • 💬 ผู้พิการทางการสื่อสาร</div>
                </div>
                <div className="bg-gradient-to-br from-[#FFD700]/10 to-[#FFD700]/5 rounded-2xl p-6 border border-[#FFD700]/20">
                  <h5 className="text-lg font-semibold text-[#1A1A1A] mb-3">🎵 Text-to-Speech</h5>
                  <p className="text-[#666] text-sm mb-2">แปลงข้อความเป็นเสียง</p>
                  <p className="text-[#00A651] text-xs font-medium">ช่วยผู้พิการทางการได้ยินโต้ตอบกับลูกค้า</p>
                  <div className="mt-2 text-xs text-[#666]">👂 ผู้พิการทางการได้ยิน • 💬 ผู้พิการทางการสื่อสาร</div>
                </div>
                <div className="bg-gradient-to-br from-[#00A651]/10 to-[#00A651]/5 rounded-2xl p-6 border border-[#00A651]/20">
                  <h5 className="text-lg font-semibold text-[#1A1A1A] mb-3">🤖 AI Suggestion</h5>
                  <p className="text-[#666] text-sm mb-2">AI แนะนำคำตอบที่เหมาะสม</p>
                  <p className="text-[#00A651] text-xs font-medium">ช่วยผู้พิการทางสติปัญญาที่มีปัญหาเรื่องความจำ</p>
                  <div className="mt-2 text-xs text-[#666]">🧠 ผู้พิการทางสติปัญญา • 👂 ผู้พิการทางการได้ยิน</div>
                </div>
                <div className="bg-gradient-to-br from-[#0066CC]/10 to-[#0066CC]/5 rounded-2xl p-6 border border-[#0066CC]/20">
                  <h5 className="text-lg font-semibold text-[#1A1A1A] mb-3">📚 Knowledge Management</h5>
                  <p className="text-[#666] text-sm mb-2">คลังความรู้และข้อมูลที่เกี่ยวข้อง</p>
                  <p className="text-[#00A651] text-xs font-medium">ช่วยผู้พิการทางสติปัญญาเข้าถึงข้อมูลได้รวดเร็ว</p>
                  <div className="mt-2 text-xs text-[#666]">🧠 ผู้พิการทางสติปัญญา • 👂 ผู้พิการทางการได้ยิน</div>
                </div>
                <div className="bg-gradient-to-br from-[#FFD700]/10 to-[#FFD700]/5 rounded-2xl p-6 border border-[#FFD700]/20">
                  <h5 className="text-lg font-semibold text-[#1A1A1A] mb-3">😊 Emotional Detection</h5>
                  <p className="text-[#666] text-sm mb-2">ตรวจจับอารมณ์จากเสียงและข้อความ</p>
                  <p className="text-[#00A651] text-xs font-medium">เหมาะสำหรับผู้พิการทุกกลุ่ม</p>
                  <div className="mt-2 text-xs text-[#666]">👂 ผู้พิการทางการได้ยิน • 💬 ผู้พิการทางการสื่อสาร • 🧠 ผู้พิการทางสติปัญญา</div>
                </div>
              </div>
              
              {/* Accessibility Summary */}
              <div className="bg-gradient-to-r from-[#00A651]/10 to-[#0066CC]/10 rounded-2xl p-6 border border-[#00A651]/30 mb-8">
                <h4 className="text-lg font-semibold text-[#1A1A1A] mb-4 text-center">🎯 ระบบออกแบบเพื่อผู้พิการแต่ละกลุ่ม</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-3xl mb-2">👂</div>
                    <h5 className="font-semibold text-[#1A1A1A] mb-2">ผู้พิการทางการได้ยิน</h5>
                    <p className="text-[#666] text-sm">ใช้การพิมพ์และแปลงข้อความเป็นเสียง เพื่อการสื่อสารที่ราบรื่น</p>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl mb-2">💬</div>
                    <h5 className="font-semibold text-[#1A1A1A] mb-2">ผู้พิการทางการสื่อสาร</h5>
                    <p className="text-[#666] text-sm">ใช้การพิมพ์และ AI แนะนำ เพื่อการตอบสนองที่เหมาะสม</p>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl mb-2">🧠</div>
                    <h5 className="font-semibold text-[#1A1A1A] mb-2">ผู้พิการทางสติปัญญา</h5>
                    <p className="text-[#666] text-sm">ใช้ AI แนะนำและจัดการความรู้ เพื่อการเข้าถึงข้อมูลที่ง่าย</p>
                  </div>
                </div>
              </div>
              
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
