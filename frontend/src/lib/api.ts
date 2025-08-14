export type ChatResponse = {
  ai_response: string;
  suggestions: string[];
  tts_audio_url: string;
  candidates?: string[];
  kb?: { title: string; snippet: string }[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export async function postChat(userMessage: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_message: userMessage }),
  });
  if (!res.ok) {
    throw new Error(`Chat request failed: ${res.status}`);
  }
  const data: ChatResponse = await res.json();
  if (data.tts_audio_url && !/^https?:\/\//.test(data.tts_audio_url)) {
    data.tts_audio_url = `${API_BASE}${data.tts_audio_url}`;
  }
  return data;
}

export async function postFeedback(original_message: string, corrected_message: string): Promise<{ status: string; message: string }>{
  const res = await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ original_message, corrected_message }),
  });
  if (!res.ok) {
    throw new Error(`Feedback request failed: ${res.status}`);
  }
  return res.json();
}

export async function postSpeak(text: string): Promise<{ tts_audio_url: string }>{
  const res = await fetch(`${API_BASE}/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    throw new Error(`Speak request failed: ${res.status}`);
  }
  const data = await res.json();
  if (data.tts_audio_url && !/^https?:\/\//.test(data.tts_audio_url)) {
    data.tts_audio_url = `${API_BASE}${data.tts_audio_url}`;
  }
  return data;
}

export async function postOtpSend(phone: string): Promise<{ request_id: string }>{
  const res = await fetch(`${API_BASE}/otp/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phone }),
  });
  if (!res.ok) throw new Error(`OTP send failed: ${res.status}`);
  return res.json();
}

export async function postOtpVerify(request_id: string, code: string): Promise<{ verified: boolean }>{
  const res = await fetch(`${API_BASE}/otp/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id, code }),
  });
  if (!res.ok) throw new Error(`OTP verify failed: ${res.status}`);
  return res.json();
} 