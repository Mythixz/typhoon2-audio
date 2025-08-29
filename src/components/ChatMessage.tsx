import React from "react";

export type ChatMessageProps = {
  role: "user" | "ai";
  text: string;
  audioUrl?: string;
  autoPlay?: boolean;
};

export default function ChatMessage({ role, text, audioUrl, autoPlay }: ChatMessageProps) {
  const isUser = role === "user";
  const bubbleBase = "px-4 py-3 rounded-2xl max-w-[80%] shadow-sm text-base leading-relaxed";
  const bubbleClass = isUser
    ? `${bubbleBase} bg-indigo-600 text-white`
    : `${bubbleBase} bg-white text-gray-900 border border-gray-200`;

  const avatar = isUser ? (
    <div className="h-8 w-8 rounded-full bg-indigo-600 text-white flex items-center justify-center text-sm font-semibold shadow-sm">U</div>
  ) : (
    <div className="h-8 w-8 rounded-full bg-emerald-600 text-white flex items-center justify-center text-sm font-semibold shadow-sm">AI</div>
  );

  return (
    <div className={`w-full my-3 flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`flex items-end gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
        {avatar}
        <div className={bubbleClass}>
          <p className="whitespace-pre-wrap">{text}</p>
          {!isUser && audioUrl ? (
            <audio className="mt-2 w-full" src={audioUrl} controls autoPlay={autoPlay} />
          ) : null}
        </div>
      </div>
    </div>
  );
} 