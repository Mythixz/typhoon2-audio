"use client";

import React, { useMemo, useRef, useState } from "react";
import ChatMessage from "@/components/ChatMessage";
import SuggestionButtons from "@/components/SuggestionButtons";
import HITLModal from "@/components/HITLModal";
import { postChat, postFeedback, type ChatResponse } from "@/lib/api";

export default function HomePage() {
  const [messages, setMessages] = useState<Array<{ role: "user" | "ai"; text: string; audioUrl?: string }>>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showHitl, setShowHitl] = useState(false);
  const [lastUserMessage, setLastUserMessage] = useState("");
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  async function handleSend() {
    if (!canSend) return;
    const text = input.trim();
    setLastUserMessage(text);
    setInput("");
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", text }]);
    try {
      const res: ChatResponse = await postChat(text);
      setMessages((prev) => [
        ...prev,
        { role: "ai", text: res.ai_response, audioUrl: res.tts_audio_url },
      ]);
      setSuggestions(res.suggestions || []);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: "ai", text: `เกิดข้อผิดพลาด: ${e?.message ?? "ไม่ทราบสาเหตุ"}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-2xl px-4 py-8">
        <h1 className="text-2xl font-bold text-gray-900">AI Supervisor Chat (POC)</h1>

        <div className="mt-6 h-[60vh] overflow-y-auto bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
          {messages.map((m, idx) => (
            <ChatMessage key={idx} role={m.role} text={m.text} audioUrl={m.audioUrl} autoPlay={!loading && m.role === "ai"} />
          ))}
          <SuggestionButtons suggestions={suggestions} onChoose={(t) => setInput(t)} />
        </div>

        <div className="mt-4 flex items-center gap-2">
          <input
            className="flex-1 border border-gray-300 rounded-md px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="พิมพ์ข้อความที่นี่"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button
            className="px-4 py-2 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={!canSend}
            onClick={handleSend}
          >ส่งข้อความ</button>
          <button
            className="px-4 py-2 rounded-md bg-amber-500 hover:bg-amber-600 text-white"
            onClick={() => setShowHitl(true)}
            type="button"
          >แก้ไขข้อความ</button>
        </div>

        <HITLModal
          open={showHitl}
          onClose={() => setShowHitl(false)}
          originalMessage={lastUserMessage}
          onSubmit={async (corrected) => {
            if (!corrected.trim()) return;
            await postFeedback(lastUserMessage, corrected.trim());
          }}
        />
      </div>
    </main>
  );
}
