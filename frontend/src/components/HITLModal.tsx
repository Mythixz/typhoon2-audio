import React, { useState } from "react";

export default function HITLModal({
  open,
  onClose,
  originalMessage,
  onSubmit,
}: {
  open: boolean;
  onClose: () => void;
  originalMessage: string;
  onSubmit: (correctedMessage: string) => Promise<void> | void;
}) {
  const [text, setText] = useState("");
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-xl p-5 w-full max-w-md border border-gray-200 shadow-xl">
        <h3 className="text-lg font-semibold text-gray-900">ปรับแก้คำแนะนำ (Human-in-the-Loop)</h3>
        <p className="text-sm text-gray-600 mt-1">ข้อความเดิม: {originalMessage}</p>
        <textarea
          className="mt-3 w-full border border-gray-300 rounded-md p-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          rows={4}
          placeholder="พิมพ์ข้อความที่ถูกต้อง"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div className="mt-3 flex justify-end gap-2">
          <button className="px-3 py-1.5 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-900" onClick={onClose} type="button">ยกเลิก</button>
          <button
            className="px-3 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white"
            onClick={async () => {
              await onSubmit(text);
              setText("");
              onClose();
            }}
            type="button"
          >ส่ง</button>
        </div>
      </div>
    </div>
  );
} 