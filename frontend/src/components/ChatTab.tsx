"use client";

import React from "react";
import ChatMessage from "./ChatMessage";
import SuggestionButtons from "./SuggestionButtons";
import { ChatMessage as ChatMessageType, KnowledgeItem } from "@/types";

interface ChatTabProps {
  messages: ChatMessageType[];
  loading: boolean;
  suggestions: string[];
  candidates: string[];
  kb: KnowledgeItem[];
  input: string;
  setInput: (input: string) => void;
  handleSend: () => void;
  handleSpeak: (text: string) => void;
  setShowHitl: (show: boolean) => void;
}

export default function ChatTab({
  messages,
  loading,
  suggestions,
  candidates,
  kb,
  input,
  setInput,
  handleSend,
  handleSpeak,
  setShowHitl
}: ChatTabProps) {
  return (
    <div className="space-y-8 animate-slide-up">
      {/* Chat Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-[#00A651] to-[#0066CC] rounded-full mb-4">
          <span className="text-2xl">💬</span>
        </div>
        <h2 className="text-3xl font-bold text-[#1A1A1A] mb-2 font-anuphan">แชทกับ AI</h2>
        <p className="text-[#666] text-lg font-anuphan-medium">สนทนากับ AI Call Center แบบเรียลไทม์ พร้อมฟีเจอร์ครบครัน</p>
        
        {/* Feature Highlights */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-gradient-to-r from-[#00A651]/10 to-[#00A651]/5 rounded-xl p-4 border border-[#00A651]/20">
            <div className="text-2xl mb-2">✍️</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">Text-Based Chat</h4>
            <p className="text-[#666] text-xs">พิมพ์ข้อความโต้ตอบกับ AI</p>
          </div>
          <div className="bg-gradient-to-r from-[#0066CC]/10 to-[#0066CC]/5 rounded-xl p-4 border border-[#0066CC]/20">
            <div className="text-2xl mb-2">🎵</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">Text-to-Speech</h4>
            <p className="text-[#666] text-xs">แปลงข้อความเป็นเสียง</p>
          </div>
          <div className="bg-gradient-to-r from-[#FFD700]/10 to-[#FFD700]/5 rounded-xl p-4 border border-[#FFD700]/20">
            <div className="text-2xl mb-2">🤖</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">AI Suggestions</h4>
            <p className="text-[#666] text-xs">AI แนะนำคำตอบ</p>
          </div>
        </div>
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
                    <button className="btn-jump-accent text-sm px-6 py-3" onClick={() => handleSpeak(c)} type="button">
                      🎵 ฟังเสียง AI
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
              disabled={!input.trim() || loading}
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
} 