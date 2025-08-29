"use client";

import React from "react";
import SpeechToText from "./SpeechToText";
import { TabType } from "@/types";

interface SpeechTabProps {
  transcriptionText: string;
  detectedEmotion: string;
  handleTranscriptionComplete: (text: string, emotion?: string) => void;
  handleDemoSpeechTranscription: () => void;
  setActiveTab: React.Dispatch<React.SetStateAction<TabType>>;
  handleSpeak: (text: string) => void;
}

export default function SpeechTab({
  transcriptionText,
  detectedEmotion,
  handleTranscriptionComplete,
  handleDemoSpeechTranscription,
  setActiveTab,
  handleSpeak
}: SpeechTabProps) {
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
        
        {/* Feature Benefits */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6 max-w-4xl mx-auto">
          <div className="bg-gradient-to-r from-[#0066CC]/10 to-[#0066CC]/5 rounded-xl p-4 border border-[#0066CC]/20">
            <div className="text-2xl mb-2">🔊</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">Speech-to-Text</h4>
            <p className="text-[#666] text-xs">แปลงเสียงลูกค้าเป็นข้อความ</p>
            <p className="text-[#00A651] text-xs font-medium">ช่วยผู้พิการทางการได้ยินเข้าใจลูกค้า</p>
          </div>
          <div className="bg-gradient-to-r from-[#00A651]/10 to-[#00A651]/5 rounded-xl p-4 border border-[#00A651]/20">
            <div className="text-2xl mb-2">😊</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">Emotion Detection</h4>
            <p className="text-[#666] text-xs">ตรวจจับอารมณ์จากเสียงพูด</p>
            <p className="text-[#00A651] text-xs font-medium">ตอบสนองได้เหมาะสมกับอารมณ์</p>
          </div>
        </div>
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
} 