export type ChatResponse = {
  text: string;
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

// เพิ่ม CRM System
export type CRMContact = {
  id: string;
  name: string;
  phone: string;
  email?: string;
  company?: string;
  lastContact: Date;
  callHistory: CallRecord[];
  preferences: CustomerPreferences;
  status: 'active' | 'inactive' | 'vip';
};

export type CallRecord = {
  id: string;
  timestamp: Date;
  duration: number;
  type: 'incoming' | 'outgoing';
  summary: string;
  emotion: string;
  satisfaction: number;
  agentId: string;
};

export type CustomerPreferences = {
  language: 'th' | 'en';
  communicationMethod: 'voice' | 'text' | 'both';
  preferredTime: string;
  specialNeeds: string[];
};

// เพิ่ม RAG System
export type KnowledgeItem = {
  id: string;
  title: string;
  content: string;
  category: string;
  tags: string[];
  lastUpdated: Date;
  confidence: number;
  source: string;
};

export type RAGResponse = {
  answer: string;
  sources: KnowledgeItem[];
  confidence: number;
  suggestedQuestions: string[];
};

// เพิ่ม AI Training System
export type TrainingFeedback = {
  id: string;
  query: string;
  aiResponse: string;
  humanCorrection: string;
  reason: string;
  timestamp: Date;
  agentId: string;
  category: string;
};

export type PerformanceMetrics = {
  accuracy: number;
  responseTime: number;
  customerSatisfaction: number;
  callResolutionRate: number;
  knowledgeCoverage: number;
};

// เพิ่ม Voice Generation System
export type VoiceProfile = {
  id: string;
  name: string;
  gender: 'male' | 'female';
  age: number;
  accent: string;
  emotion: string;
  speed: number;
  pitch: number;
};

export type VoiceParameters = {
  profileId: string;
  customizations: {
    speed?: number;
    pitch?: number;
    emotion?: string;
    emphasis?: string[];
  };
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export async function postChat(message: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_message: message }),
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

export async function postEnhancedChat(message: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat/enhanced`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_message: message }),
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

export async function postFeedback(original: string, corrected: string): Promise<void> {
  const res = await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ original_message: original, corrected_message: corrected }),
  });
  if (!res.ok) {
    throw new Error(`Feedback request failed: ${res.status}`);
  }
  return res.json();
}

export async function postSpeak(text: string): Promise<void> {
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

export async function postOtpSend(phone: string): Promise<{ requestId: string }> {
  const res = await fetch(`${API_BASE}/otp/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phone }),
  });
  if (!res.ok) throw new Error(`OTP send failed: ${res.status}`);
  return res.json();
}

export async function postOtpVerify(requestId: string, code: string): Promise<{ verified: boolean }> {
  const res = await fetch(`${API_BASE}/otp/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id: requestId, code }),
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

// เพิ่ม CRM APIs
export async function getCRMContacts(): Promise<CRMContact[]> {
  const response = await fetch(`${API_BASE}/crm/contacts`);
  return response.json();
}

export async function createCRMContact(contact: Omit<CRMContact, 'id' | 'lastContact' | 'callHistory'>): Promise<CRMContact> {
  const response = await fetch(`${API_BASE}/crm/contacts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(contact),
  });
  return response.json();
}

export async function updateCRMContact(id: string, updates: Partial<CRMContact>): Promise<CRMContact> {
  const response = await fetch(`${API_BASE}/crm/contacts/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
  return response.json();
}

// เพิ่ม RAG APIs
export async function searchKnowledge(query: string): Promise<KnowledgeItem[]> {
  const response = await fetch(`${API_BASE}/rag/search?q=${encodeURIComponent(query)}`);
  return response.json();
}

export async function generateRAGResponse(query: string, context?: string): Promise<RAGResponse> {
  const response = await fetch(`${API_BASE}/rag/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, context }),
  });
  return response.json();
}

export async function addKnowledge(knowledge: Omit<KnowledgeItem, 'id' | 'lastUpdated'>): Promise<KnowledgeItem> {
  const response = await fetch(`${API_BASE}/rag/knowledge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(knowledge),
  });
  return response.json();
}

// เพิ่ม AI Training APIs
export async function submitTrainingFeedback(feedback: Omit<TrainingFeedback, 'id' | 'timestamp'>): Promise<void> {
  await fetch(`${API_BASE}/ai/training/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(feedback),
  });
}

export async function getPerformanceMetrics(): Promise<PerformanceMetrics> {
  const response = await fetch(`${API_BASE}/ai/performance`);
  return response.json();
}

// เพิ่ม Voice Generation APIs
export async function generateVoice(text: string, profile: VoiceProfile): Promise<{ audioUrl: string }> {
  const response = await fetch(`${API_BASE}/voice/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, profile }),
  });
  return response.json();
}

export async function customizeVoice(profileId: string, parameters: VoiceParameters['customizations']): Promise<void> {
  await fetch(`${API_BASE}/voice/customize/${profileId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(parameters),
  });
} 