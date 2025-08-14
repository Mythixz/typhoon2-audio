"use client";

import React, { useMemo, useRef, useState } from "react";
import ChatMessage from "@/components/ChatMessage";
import SuggestionButtons from "@/components/SuggestionButtons";
import HITLModal from "@/components/HITLModal";
import { postChat, postFeedback, postSpeak, postOtpSend, postOtpVerify, type ChatResponse } from "@/lib/api";

export default function HomePage() {
  const [messages, setMessages] = useState<Array<{ role: "user" | "ai"; text: string; audioUrl?: string }>>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [candidates, setCandidates] = useState<string[]>([]);
  const [kb, setKb] = useState<Array<{ title: string; snippet: string }>>([]);
  const [showHitl, setShowHitl] = useState(false);
  const [lastUserMessage, setLastUserMessage] = useState("");
  const [otpPhone, setOtpPhone] = useState("");
  const [otpReqId, setOtpReqId] = useState<string | null>(null);
  const [otpCode, setOtpCode] = useState("");
  const [otpVerified, setOtpVerified] = useState(false);
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
      setCandidates(res.candidates || []);
      setKb(res.kb || []);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: "ai", text: `เกิดข้อผิดพลาด: ${e?.message ?? "ไม่ทราบสาเหตุ"}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function handleSpeak(text: string) {
    const t = text.trim();
    if (!t) return;
    try {
      const { tts_audio_url } = await postSpeak(t);
      setMessages((prev) => [...prev, { role: "ai", text: t, audioUrl: tts_audio_url }]);
    } catch (e) {
      setMessages((prev) => [...prev, { role: "ai", text: `พูดไม่ได้: ${(e as any)?.message ?? "ไม่ทราบสาเหตุ"}` }]);
    }
  }

  async function handleOtpSend() {
    const phone = otpPhone.trim();
    if (!phone) return;
    try {
      const { request_id } = await postOtpSend(phone);
      setOtpReqId(request_id);
    } catch (e) {
      alert(`ส่ง OTP ไม่สำเร็จ: ${(e as any)?.message ?? "ไม่ทราบสาเหตุ"}`);
    }
  }

  async function handleOtpVerify() {
    if (!otpReqId) return;
    try {
      const { verified } = await postOtpVerify(otpReqId, otpCode.trim());
      setOtpVerified(verified);
      if (!verified) alert("รหัสไม่ถูกต้องหรือหมดอายุ");
    } catch (e) {
      alert(`ยืนยัน OTP ไม่สำเร็จ: ${(e as any)?.message ?? "ไม่ทราบสาเหตุ"}`);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-100 to-slate-200">
      <div className="mx-auto max-w-3xl px-4 py-10">
        <div className="rounded-2xl bg-white/70 backdrop-blur border border-slate-200 shadow-sm p-6">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-gray-900">AI Supervisor Chat</h1>
            <div className="text-xs text-gray-500">POC for Inclusive Call Center</div>
          </div>

          <div className="mt-6 h-[60vh] overflow-y-auto rounded-xl p-4 bg-slate-50 border border-slate-200">
            {messages.map((m, idx) => (
              <ChatMessage key={idx} role={m.role} text={m.text} audioUrl={m.audioUrl} autoPlay={!loading && m.role === "ai"} />
            ))}
            <SuggestionButtons suggestions={suggestions} onChoose={(t) => setInput(t)} />

            {candidates?.length ? (
              <div className="mt-5">
                <h3 className="text-sm font-semibold text-gray-800">คำตอบแนะนำ</h3>
                <div className="mt-2 grid grid-cols-1 gap-3">
                  {candidates.map((c, i) => (
                    <div key={`cand-${i}`} className="border border-gray-200 rounded-xl p-3 bg-white/80 backdrop-blur shadow-sm">
                      <p className="text-gray-900 whitespace-pre-wrap text-sm">{c}</p>
                      <div className="mt-2 flex gap-2">
                        <button className="px-2.5 py-1.5 rounded-md bg-white border border-gray-300 text-gray-800 text-sm hover:bg-gray-100" onClick={() => setInput(c)} type="button">ใช้ข้อความนี้</button>
                        <button className="px-2.5 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white text-sm" onClick={() => handleSpeak(c)} type="button">พูดข้อความนี้</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {kb?.length ? (
              <div className="mt-5">
                <h3 className="text-sm font-semibold text-gray-800">ความรู้ที่เกี่ยวข้อง</h3>
                <ul className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-3">
                  {kb.map((k, i) => (
                    <li key={`kb-${i}`} className="border border-gray-200 rounded-xl p-3 bg-white/80 backdrop-blur shadow-sm">
                      <p className="text-gray-900 text-sm font-medium">{k.title}</p>
                      <p className="text-gray-600 text-sm mt-0.5">{k.snippet}</p>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>

          <div className="mt-4 flex items-center gap-2">
            <input
              className="flex-1 border border-gray-300 rounded-xl px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 shadow-sm"
              placeholder="พิมพ์ข้อความที่นี่"
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button
              className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white disabled:opacity-50 disabled:cursor-not-allowed shadow"
              disabled={!canSend}
              onClick={handleSend}
            >ส่งข้อความ</button>
            <button
              className="px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-600 text-white shadow"
              onClick={() => setShowHitl(true)}
              type="button"
            >แก้ไขข้อความ</button>
            <button
              className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white shadow"
              onClick={() => handleSpeak(input)}
              type="button"
            >พูด</button>
          </div>

          <div className="mt-6 rounded-xl p-4 bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200">
            <h3 className="text-sm font-semibold text-emerald-900">ยืนยันตัวตน (OTP Demo)</h3>
            <div className="mt-3 flex flex-col md:flex-row items-stretch md:items-center gap-2">
              <input className="flex-1 border border-emerald-300 rounded-lg px-3 py-2 bg-white shadow-sm" placeholder="เบอร์โทรศัพท์" value={otpPhone} onChange={(e) => setOtpPhone(e.target.value)} />
              <button className="px-3 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white shadow" onClick={handleOtpSend} type="button">ส่ง OTP</button>
            </div>
            {otpReqId ? (
              <div className="mt-3 flex flex-col md:flex-row items-stretch md:items-center gap-2">
                <input className="flex-1 border border-emerald-300 rounded-lg px-3 py-2 bg-white shadow-sm" placeholder="รหัส 6 หลัก" value={otpCode} onChange={(e) => setOtpCode(e.target.value)} />
                <button className="px-3 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white shadow" onClick={handleOtpVerify} type="button">ยืนยัน</button>
                {otpVerified ? <span className="text-emerald-700 text-sm">ยืนยันแล้ว</span> : null}
              </div>
            ) : null}
            <p className="text-xs text-emerald-900/70 mt-2">หมายเหตุ: เดโม่นี้แสดงรหัส OTP ในฝั่งเซิร์ฟเวอร์ log (mock) — เปลี่ยนเป็น AIS OTP ได้ทันทีเมื่อมี credentials</p>
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

      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute -top-10 -left-10 h-60 w-60 rounded-full bg-indigo-200/20 blur-3xl" />
        <div className="absolute -bottom-10 -right-10 h-60 w-60 rounded-full bg-emerald-200/20 blur-3xl" />
      </div>
    </main>
  );
}
