"use client";

import React, { useState } from "react";

interface HITLModalProps {
  open: boolean;
  onClose: () => void;
  originalMessage: string;
  onSubmit: (corrected: string) => Promise<void>;
}

export default function HITLModal({ open, onClose, originalMessage, onSubmit }: HITLModalProps) {
  const [corrected, setCorrected] = useState(originalMessage);
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!open) return null;

  const handleSubmit = async () => {
    if (!corrected.trim()) return;
    setIsSubmitting(true);
    try {
      await onSubmit(corrected.trim());
      onClose();
    } catch (error) {
      console.error("Error submitting correction:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="relative bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-[#00A651] to-[#0066CC] text-white p-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold flex items-center gap-3">
              <span className="text-3xl">✏️</span>
              แก้ไขข้อความ (Human-in-the-Loop)
            </h2>
            <button
              onClick={onClose}
              className="text-white/80 hover:text-white text-2xl transition-colors"
            >
              ✕
            </button>
          </div>
          <p className="text-white/90 mt-2">
            ตรวจสอบและแก้ไขข้อความที่ AI สร้างขึ้นก่อนส่งให้ลูกค้า
          </p>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Original Message */}
          <div className="bg-gradient-to-br from-[#E8F5E8] to-[#E6F3FF] border border-[#00A651]/20 rounded-xl p-4">
            <h3 className="font-semibold text-[#1A1A1A] mb-3 flex items-center gap-2">
              <span className="text-[#00A651]">📝</span>
              ข้อความต้นฉบับ
            </h3>
            <p className="text-[#666] leading-relaxed">{originalMessage}</p>
          </div>

          {/* Correction Input */}
          <div>
            <label className="block font-semibold text-[#1A1A1A] mb-3 flex items-center gap-2">
              <span className="text-[#0066CC]">✏️</span>
              แก้ไขข้อความ
            </label>
            <textarea
              value={corrected}
              onChange={(e) => setCorrected(e.target.value)}
              className="input-jump w-full h-32 resize-none"
              placeholder="พิมพ์ข้อความที่แก้ไขแล้ว..."
            />
            <div className="flex items-center justify-between mt-2 text-sm text-[#666]">
              <span>💡 สามารถแก้ไขได้ตามต้องการ</span>
              <span>{corrected.length} ตัวอักษร</span>
            </div>
          </div>

          {/* Quality Check */}
          <div className="bg-gradient-to-br from-[#FFF3CD] to-[#F8D7DA] border border-[#FFC107]/30 rounded-xl p-4">
            <h4 className="font-semibold text-[#1A1A1A] mb-3 flex items-center gap-2">
              <span className="text-[#FFC107]">🔍</span>
              ตรวจสอบคุณภาพ
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              <div className="flex items-center gap-2">
                <span className={`w-3 h-3 rounded-full ${corrected.length > 10 ? 'bg-[#28A745]' : 'bg-[#DC3545]'}`}></span>
                <span className={corrected.length > 10 ? 'text-[#28A745]' : 'text-[#DC3545]'}>
                  ความยาวข้อความ {corrected.length > 10 ? '✅' : '❌'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`w-3 h-3 rounded-full ${corrected.includes('ครับ') || corrected.includes('ค่ะ') ? 'bg-[#28A745]' : 'bg-[#FFC107]'}`}></span>
                <span className={corrected.includes('ครับ') || corrected.includes('ค่ะ') ? 'text-[#28A745]' : 'text-[#FFC107]'}>
                  ใช้คำสุภาพ {corrected.includes('ครับ') || corrected.includes('ค่ะ') ? '✅' : '⚠️'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`w-3 h-3 rounded-full ${corrected.length > 0 ? 'bg-[#28A745]' : 'bg-[#DC3545]'}`}></span>
                <span className={corrected.length > 0 ? 'text-[#28A745]' : 'text-[#DC3545]'}>
                  มีข้อความ {corrected.length > 0 ? '✅' : '❌'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`w-3 h-3 rounded-full ${corrected !== originalMessage ? 'bg-[#28A745]' : 'bg-[#6C757D]'}`}></span>
                <span className={corrected !== originalMessage ? 'text-[#28A745]' : 'text-[#6C757D]'}>
                  มีการแก้ไข {corrected !== originalMessage ? '✅' : '➖'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 flex items-center justify-end gap-3">
          <button
            onClick={onClose}
            className="btn-jump-outline border-[#6C757D] text-[#6C757D] hover:bg-[#6C757D] hover:text-white"
          >
            ยกเลิก
          </button>
          <button
            onClick={handleSubmit}
            disabled={!corrected.trim() || isSubmitting}
            className="btn-jump-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? (
              <div className="flex items-center gap-2">
                <div className="spinner-jump"></div>
                <span>กำลังส่ง...</span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <span>✅</span>
                <span>ยืนยันการแก้ไข</span>
              </div>
            )}
          </button>
        </div>
      </div>
    </div>
  );
} 