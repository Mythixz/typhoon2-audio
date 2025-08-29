"use client";

import React from "react";
import ChatMessage from "./ChatMessage";
import { ChatMessage as ChatMessageType } from "@/types";

interface EnhancedTabProps {
  messages: ChatMessageType[];
  loading: boolean;
  input: string;
  setInput: (input: string) => void;
  handleEnhancedChat: () => void;
}

export default function EnhancedTab({
  messages,
  loading,
  input,
  setInput,
  handleEnhancedChat
}: EnhancedTabProps) {
  return (
    <div className="space-y-6">
      <div className="card-jump card-jump-secondary p-8">
        <h2 className="text-2xl font-bold text-[#1A1A1A] mb-6 flex items-center gap-3">
          <span className="text-[#0066CC] text-3xl">🚀</span>
          แชทขั้นสูง
        </h2>
        <p className="text-[#666] mb-6">แชทที่มีการตรวจจับอารมณ์และตอบสนองตามบริบท</p>
        
        {/* Enhanced Features */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="bg-gradient-to-r from-[#0066CC]/10 to-[#0066CC]/5 rounded-xl p-4 border border-[#0066CC]/20">
            <div className="text-2xl mb-2">😊</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">Emotion Detection</h4>
            <p className="text-[#666] text-xs">ตรวจจับอารมณ์จากข้อความ</p>
            <p className="text-[#00A651] text-xs font-medium">ตอบสนองได้เหมาะสมกับอารมณ์</p>
          </div>
          <div className="bg-gradient-to-r from-[#00A651]/10 to-[#00A651]/5 rounded-xl p-4 border border-[#00A651]/20">
            <div className="text-2xl mb-2">🧠</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">Context Awareness</h4>
            <p className="text-[#666] text-xs">เข้าใจบริบทการสนทนา</p>
            <p className="text-[#00A651] text-xs font-medium">ช่วยผู้พิการทางสติปัญญา</p>
          </div>
        </div>
            
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
} 