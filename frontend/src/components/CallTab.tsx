"use client";

import React from "react";

export default function CallTab() {
  return (
    <div className="space-y-8 animate-slide-up">
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-[#00A651] to-[#0066CC] rounded-full mb-6">
          <span className="text-3xl">📞</span>
        </div>
        <h2 className="text-4xl font-bold text-[#1A1A1A] mb-3 font-anuphan">การสนทนาสองทาง</h2>
        <p className="text-[#666] text-xl max-w-2xl mx-auto leading-relaxed font-anuphan-medium">
          ทดสอบการสนทนากับ AI Call Center แบบสองทาง พร้อมการแปลงเสียงและตรวจจับอารมณ์
        </p>
      </div>

      <div className="card-jump card-jump-primary p-10 animate-scale-in">
        <div className="text-center mb-8">
          <h3 className="text-2xl font-bold text-[#1A1A1A] mb-3">ระบบสนทนาสองทาง</h3>
          <p className="text-[#666] text-lg">ทดสอบการสนทนากับ AI แบบสองทาง พร้อมการแปลงเสียงและตรวจจับอารมณ์</p>
        </div>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#00A651]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4">สถานะการเชื่อมต่อ</h4>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-3 h-3 bg-[#00A651] rounded-full animate-pulse"></div>
              <span className="text-[#1A1A1A] font-medium">พร้อมใช้งาน</span>
            </div>
            <p className="text-[#666] text-sm">ระบบพร้อมสำหรับการสนทนาสองทางกับ AI</p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => alert("🎯 ระบบจำลอง: เริ่มการสนทนาสองทาง\nในเวอร์ชันจริงจะมีการเชื่อมต่อกับ AI Call Center จริง")}
              className="btn-jump-primary text-lg px-8 py-4 transform hover:scale-105 transition-all duration-300"
            >
              📞 เริ่มการสนทนา
            </button>
            <button
              onClick={() => alert("🎯 ระบบจำลอง: หยุดการสนทนา\nในเวอร์ชันจริงจะมีการปิดการเชื่อมต่อกับ AI Call Center")}
              className="btn-jump-secondary text-lg px-8 py-4 transform hover:scale-105 transition-all duration-300"
            >
              ⏹️ หยุดการสนทนา
            </button>
          </div>

          <div className="bg-gradient-to-r from-[#00A651]/10 to-[#0066CC]/10 rounded-2xl p-6 border border-[#00A651]/20">
            <p className="text-sm text-[#666] text-center">
              <strong>หมายเหตุ:</strong> ระบบนี้เป็นเดโม่สำหรับการทดสอบ UI และ UX — ในเวอร์ชันจริงจะมีการเชื่อมต่อกับ AI Call Center จริง
            </p>
          </div>
        </div>
      </div>
    </div>
  );
} 