"use client";

import React from "react";

export default function RAGTab() {
  return (
    <div className="space-y-8 animate-slide-up">
      {/* RAG Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-[#6C5CE7] to-[#A29BFE] rounded-full mb-6">
          <span className="text-3xl">🧠</span>
        </div>
        <h2 className="text-4xl font-bold text-[#1A1A1A] mb-3 font-anuphan">RAG System</h2>
        <p className="text-[#666] text-xl max-w-3xl mx-auto leading-relaxed font-anuphan-medium">
          ระบบค้นหาความรู้แบบอัจฉริยะ พร้อมการตอบคำถามด้วยความรู้จากฐานข้อมูล
        </p>
      </div>

      {/* RAG Dashboard */}
      <div className="card-jump card-jump-primary p-10 animate-scale-in">
        <div className="text-center mb-8">
          <h3 className="text-2xl font-bold text-[#1A1A1A] mb-3">ฐานข้อมูลความรู้</h3>
          <p className="text-[#666] text-lg">ฐานข้อมูลความรู้ที่เกี่ยวข้องกับปัญหาที่พนักงาน AIS ต้องการค้นหา</p>
        </div>
        
        <div className="space-y-6">
          {/* Knowledge Base */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#6C5CE7]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              📚 Knowledge Base
            </h4>
            <div className="space-y-4">
              <div className="flex gap-3">
                <button className="btn-jump-primary text-sm px-4 py-2">➕ เพิ่มความรู้ใหม่</button>
                <button className="btn-jump-secondary text-sm px-4 py-2">✏️ แก้ไขความรู้</button>
                <button className="btn-jump-accent text-sm px-4 py-2">🔍 ค้นหาความรู้</button>
              </div>
              <div className="bg-[#6C5CE7]/10 rounded-xl p-4 border border-[#6C5CE7]/20">
                <p className="text-sm text-[#1A1A1A]">
                  <strong>ตัวอย่าง:</strong> คนพิการสามารถเพิ่มความรู้ใหม่ เช่น "วิธีแก้ไขปัญหาการชำระเงิน" 
                  เพื่อให้ AI ตอบลูกค้าได้ถูกต้องและครบถ้วน
                </p>
              </div>
            </div>
          </div>

          {/* Question Answering */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#A29BFE]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              💬 Question Answering
            </h4>
            <div className="space-y-4">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-gradient-to-br from-[#6C5CE7] to-[#A29BFE] rounded-full flex items-center justify-center text-white font-semibold">
                  AIS
                </div>
                <div className="flex-1 bg-[#6C5CE7]/10 rounded-xl p-3">
                  <p className="text-[#1A1A1A] text-sm mb-2">สวัสดีครับ! ผมเป็นพนักงาน AIS ที่จะช่วยเทรน AI ร่วมกับคุณครับ</p>
                  <div className="flex gap-2">
                    <button className="btn-jump-primary text-xs px-3 py-1">✅ เข้าใจแล้ว</button>
                    <button className="btn-jump-secondary text-xs px-3 py-1">❓ มีคำถาม</button>
                  </div>
                </div>
              </div>
              
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-gradient-to-br from-[#00A651] to-[#0066CC] rounded-full flex items-center justify-center text-white font-semibold">
                  👥
                </div>
                <div className="flex-1 bg-[#00A651]/10 rounded-xl p-3">
                  <p className="text-[#1A1A1A] text-sm mb-2">สวัสดีครับ! ผมเป็นผู้พิการทางการได้ยิน ต้องการช่วยเทรน AI ให้เข้าใจปัญหาของผู้พิการครับ</p>
                  <div className="flex gap-2">
                    <button className="btn-jump-primary text-xs px-3 py-1">✅ เข้าใจแล้ว</button>
                    <button className="btn-jump-secondary text-xs px-3 py-1">❓ มีคำถาม</button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Retrieval Augmentation */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#00A651]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              📊 Retrieval Augmentation
            </h4>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-[#6C5CE7] mb-2">15</div>
                  <div className="text-sm text-[#666]">เซสชันการเทรน</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-[#A29BFE] mb-2">89%</div>
                  <div className="text-sm text-[#666]">ความแม่นยำ AI</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-[#00A651] mb-2">47</div>
                  <div className="text-sm text-[#666]">ความรู้ใหม่</div>
                </div>
              </div>
              <div className="flex gap-3">
                <button className="btn-jump-primary text-sm px-4 py-2">📊 ดูรายงาน</button>
                <button className="btn-jump-secondary text-sm px-4 py-2">📝 สรุปการเทรน</button>
                <button className="btn-jump-accent text-sm px-4 py-2">🎯 ตั้งเป้าหมาย</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Benefits Summary */}
      <div className="card-jump card-jump-accent p-8 animate-fade-in">
        <h3 className="text-2xl font-bold text-[#1A1A1A] mb-6 text-center">ประโยชน์ของการเทรน AI ร่วมกัน</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl mb-3">🎯</div>
            <h4 className="font-semibold text-[#1A1A1A] mb-2">สำหรับ AIS</h4>
            <ul className="space-y-1 text-sm text-[#666]">
              <li>• AI เข้าใจผู้พิการมากขึ้น</li>
              <li>• ลดข้อผิดพลาดในการตอบ</li>
              <li>• เพิ่มความพึงพอใจลูกค้า</li>
              <li>• ประหยัดค่าแรง call center</li>
            </ul>
          </div>
          <div className="text-center">
            <div className="text-3xl mb-3">👥</div>
            <h4 className="font-semibold text-[#1A1A1A] mb-2">สำหรับผู้พิการ</h4>
            <ul className="space-y-1 text-sm text-[#666]">
              <li>• มีส่วนร่วมในการพัฒนา AI</li>
              <li>• ได้เรียนรู้งานด้านเทคโนโลยี</li>
              <li>• สร้างความเข้าใจร่วมกัน</li>
              <li>• ได้งานที่มีคุณค่า</li>
            </ul>
          </div>
          <div className="text-center">
            <div className="text-3xl mb-3">🤖</div>
            <h4 className="font-semibold text-[#1A1A1A] mb-2">สำหรับ AI</h4>
            <ul className="space-y-1 text-sm text-[#666]">
              <li>• เรียนรู้จากประสบการณ์จริง</li>
              <li>• เข้าใจมุมมองของผู้พิการ</li>
              <li>• ตอบได้เหมาะสมมากขึ้น</li>
              <li>• พัฒนาตัวเองอย่างต่อเนื่อง</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
} 