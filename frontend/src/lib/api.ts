export type ChatResponse = {
  ai_response: string;
  suggestions: string[];
  tts_audio_url: string;
  candidates?: string[];
  kb?: { title: string; snippet: string }[];
};

export type SpeechToTextResponse = {
  text: string;
  confidence: number;
  emotion?: string;
  emotion_confidence?: number;
};

export type TwoWayCallResponse = {
  ai_response: string;
  tts_audio_url: string;
  emotion_detected?: string;
  suggestions: string[];
};

export type AudioUploadResponse = {
  filename: string;
  url: string;
  size: number;
  message: string;
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

export async function postEnhancedChat(userMessage: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat/enhanced`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_message: userMessage }),
  });
  if (!res.ok) {
    throw new Error(`Enhanced chat request failed: ${res.status}`);
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

// --- New API functions ---

export async function postSpeechToText(audioFile: File): Promise<SpeechToTextResponse> {
  const formData = new FormData();
  formData.append('audio_file', audioFile);
  
  const res = await fetch(`${API_BASE}/stt`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`Speech-to-text request failed: ${res.status}`);
  }
  return res.json();
}

export async function postTwoWayCall(
  audioInput?: string, 
  textInput?: string, 
  conversationId?: string
): Promise<TwoWayCallResponse> {
  const res = await fetch(`${API_BASE}/call`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ 
      audio_input: audioInput, 
      text_input: textInput, 
      conversation_id: conversationId 
    }),
  });
  if (!res.ok) {
    throw new Error(`Two-way call request failed: ${res.status}`);
  }
  const data = await res.json();
  if (data.tts_audio_url && !/^https?:\/\//.test(data.tts_audio_url)) {
    data.tts_audio_url = `${API_BASE}${data.tts_audio_url}`;
  }
  return data;
}

export async function postAudioUpload(audioFile: File): Promise<AudioUploadResponse> {
  const formData = new FormData();
  formData.append('audio_file', audioFile);
  
  const res = await fetch(`${API_BASE}/upload-audio`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`Audio upload failed: ${res.status}`);
  }
  const data = await res.json();
  if (data.url && !/^https?:\/\//.test(data.url)) {
    data.url = `${API_BASE}${data.url}`;
  }
  return data;
} 