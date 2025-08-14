import React from "react";

export default function SuggestionButtons({
  suggestions,
  onChoose,
}: {
  suggestions: string[];
  onChoose: (text: string) => void;
}) {
  if (!suggestions?.length) return null;
  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {suggestions.map((s, idx) => (
        <button
          key={`${s}-${idx}`}
          className="px-3 py-1.5 rounded-full bg-white/70 backdrop-blur border border-gray-300 text-gray-800 hover:bg-gray-100 text-sm shadow-sm transition-colors"
          onClick={() => onChoose(s)}
          type="button"
        >
          {s}
        </button>
      ))}
    </div>
  );
} 