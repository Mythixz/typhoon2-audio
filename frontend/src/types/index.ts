export type TabType = 'chat' | 'speech' | 'call' | 'enhanced' | 'supervisor' | 'collaborative' | 'crm' | 'rag';

export interface ChatMessage {
  role: "user" | "ai";
  text: string;
  audioUrl?: string;
}

export interface KnowledgeItem {
  title: string;
  snippet: string;
} 