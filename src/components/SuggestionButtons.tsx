"use client";

import React from "react";

interface SuggestionButtonsProps {
  suggestions: string[];
  onChoose: (suggestion: string) => void;
}

export default function SuggestionButtons({ suggestions, onChoose }: SuggestionButtonsProps) {
  if (!suggestions || suggestions.length === 0) return null;

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold text-[#1A1A1A] mb-4 flex items-center gap-2">
        <span className="text-[#00A651] text-xl">💡</span>
        คำแนะนำ
      </h3>
      <div className="flex flex-wrap gap-3">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onChoose(suggestion)}
            className="btn-jump-outline text-sm hover-lift transition-all duration-300"
          >
            <div className="flex items-center gap-2">
              <span className="text-[#00A651]">✨</span>
              {suggestion}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
} 