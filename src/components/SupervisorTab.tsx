"use client";

import React from "react";

export default function SupervisorTab() {
  return (
    <div className="space-y-8 animate-slide-up">
      {/* AI Supervisor Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-[#FFD700] to-[#FF6B35] rounded-full mb-6">
          <span className="text-3xl">👨‍💼</span>
        </div>
        <h2 className="text-4xl font-bold text-[#1A1A1A] mb-3 font-anuphan">AI Supervisor</h2>
        <p className="text-[#666] text-xl max-w-3xl mx-auto leading-relaxed font-anuphan-medium">
          คนพิการทำหน้าที่เป็นผู้ควบคุมและดูแล AI Call Center แทนการเป็นพนักงานธรรมดา
        </p>
        
        {/* Key Benefits */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6 max-w-4xl mx-auto">
          <div className="bg-gradient-to-r from-[#FFD700]/10 to-[#FFD700]/5 rounded-xl p-4 border border-[#FFD700]/20">
            <div className="text-2xl mb-2">🎯</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">Human Touch</h4>
            <p className="text-[#666] text-xs">คนพิการเข้าใจคนจริงๆ</p>
            <p className="text-[#FF6B35] text-xs font-medium">ไม่ใช่ AI อย่างเดียว</p>
          </div>
          <div className="bg-gradient-to-r from-[#FF6B35]/10 to-[#FF6B35]/5 rounded-xl p-4 border border-[#FF6B35]/20">
            <div className="text-2xl mb-2">🤖</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">AI Control</h4>
            <p className="text-[#666] text-xs">ควบคุมและดูแล AI</p>
            <p className="text-[#FF6B35] text-xs font-medium">เลือกคำตอบที่เหมาะสม</p>
          </div>
          <div className="bg-gradient-to-r from-[#00A651]/10 to-[#00A651]/5 rounded-xl p-4 border border-[#00A651]/20">
            <div className="text-2xl mb-2">💰</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">High Value</h4>
            <p className="text-[#666] text-xs">งานคุณภาพสูง</p>
            <p className="text-[#00A651] text-xs font-medium">ได้เงินเยอะกว่า Admin</p>
          </div>
        </div>
      </div>

      {/* AI Supervisor Dashboard */}
      <div className="card-jump card-jump-primary p-10 animate-scale-in">
        <div className="text-center mb-8">
          <h3 className="text-2xl font-bold text-[#1A1A1A] mb-3">Dashboard ควบคุม AI</h3>
          <p className="text-[#666] text-lg">คนพิการสามารถควบคุมและดูแล AI Call Center ได้</p>
        </div>
        
        <div className="space-y-6">
          {/* AI Performance Monitoring */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#FFD700]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              📊 AI Performance Monitoring
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-[#00A651] mb-2">95%</div>
                <div className="text-sm text-[#666]">ความแม่นยำ AI</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-[#0066CC] mb-2">2.3s</div>
                <div className="text-sm text-[#666]">เวลาตอบสนอง</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-[#FFD700] mb-2">87%</div>
                <div className="text-sm text-[#666]">ความพึงพอใจ</div>
              </div>
            </div>
          </div>

          {/* AI Response Control */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#FFD700]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              🎮 AI Response Control
            </h4>
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <p className="text-[#1A1A1A] text-sm mb-2">AI ตอบ: "ขอบคุณที่ติดต่อเรา ต้องการความช่วยเหลืออะไรครับ?"</p>
                  <div className="flex gap-2">
                    <button className="btn-jump-primary text-xs px-3 py-1">✅ อนุมัติ</button>
                    <button className="btn-jump-secondary text-xs px-3 py-1">✏️ แก้ไข</button>
                    <button className="btn-jump-accent text-xs px-3 py-1">🔄 สร้างใหม่</button>
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <p className="text-[#1A1A1A] text-sm mb-2">AI ตอบ: "ขออภัยครับ ระบบมีปัญหาชั่วคราว"</p>
                  <div className="flex gap-2">
                    <button className="btn-jump-primary text-xs px-3 py-1">✅ อนุมัติ</button>
                    <button className="btn-jump-secondary text-xs px-3 py-1">✏️ แก้ไข</button>
                    <button className="btn-jump-accent text-xs px-3 py-1">🔄 สร้างใหม่</button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Voice Synthesis Control */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#FFD700]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              🎵 Voice Synthesis Control
            </h4>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-[#1A1A1A] mb-2">เลือกเสียง</label>
                  <select className="input-jump w-full">
                    <option>เสียงผู้ชาย (สุภาพ)</option>
                    <option>เสียงผู้หญิง (เป็นมิตร)</option>
                    <option>เสียงผู้ชาย (มืออาชีพ)</option>
                    <option>เสียงผู้หญิง (อบอุ่น)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-[#1A1A1A] mb-2">โทนเสียง</label>
                  <select className="input-jump w-full">
                    <option>ปกติ</option>
                    <option>ช้า</option>
                    <option>เร็ว</option>
                    <option>เน้นคำสำคัญ</option>
                  </select>
                </div>
              </div>
              <div className="flex gap-3">
                <button className="btn-jump-primary text-sm px-4 py-2">🎵 ทดสอบเสียง</button>
                <button className="btn-jump-secondary text-sm px-4 py-2">💾 บันทึกการตั้งค่า</button>
              </div>
            </div>
          </div>

          {/* Knowledge Management */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#FFD700]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              📚 Knowledge Management
            </h4>
            <div className="space-y-4">
              <div className="flex gap-3">
                <button className="btn-jump-primary text-sm px-4 py-2">➕ เพิ่มความรู้ใหม่</button>
                <button className="btn-jump-secondary text-sm px-4 py-2">✏️ แก้ไขความรู้</button>
                <button className="btn-jump-accent text-sm px-4 py-2">🔍 ค้นหาความรู้</button>
              </div>
              <div className="bg-[#FFD700]/10 rounded-xl p-4 border border-[#FFD700]/20">
                <p className="text-sm text-[#1A1A1A]">
                  <strong>ตัวอย่าง:</strong> คนพิการสามารถเพิ่มความรู้ใหม่ เช่น "วิธีแก้ไขปัญหาการชำระเงิน" 
                  เพื่อให้ AI ตอบลูกค้าได้ถูกต้องและครบถ้วน
                </p>
              </div>
            </div>
          </div>

          {/* Call Transfer Control */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#FFD700]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              📞 Call Transfer Control
            </h4>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-[#1A1A1A] mb-2">โอนสายไปยัง</label>
                  <select className="input-jump w-full">
                    <option>ผู้เชี่ยวชาญด้านเทคนิค</option>
                    <option>ผู้เชี่ยวชาญด้านการเงิน</option>
                    <option>ผู้จัดการ</option>
                    <option>AI อื่นๆ</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-[#1A1A1A] mb-2">เหตุผล</label>
                  <select className="input-jump w-full">
                    <option>ปัญหาซับซ้อน</option>
                    <option>ต้องการการยืนยัน</option>
                    <option>เกินขอบเขตอำนาจ</option>
                    <option>ลูกค้าต้องการพูดกับคนจริง</option>
                  </select>
                </div>
              </div>
              <div className="flex gap-3">
                <button className="btn-jump-primary text-sm px-4 py-2">📞 โอนสาย</button>
                <button className="btn-jump-secondary text-sm px-4 py-2">📝 บันทึกเหตุผล</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Business Benefits */}
      <div className="card-jump card-jump-secondary p-8 animate-fade-in">
        <h3 className="text-2xl font-bold text-[#1A1A1A] mb-6 text-center">ประโยชน์ทางธุรกิจ</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-[#1A1A1A]">🎯 สำหรับบริษัท</h4>
            <ul className="space-y-2 text-sm text-[#666]">
              <li>• ลดหย่อนภาษี 100:1 (พนักงาน 400 คน จ้างคนพิการ 1 คน)</li>
              <li>• ไม่ต้องจ่ายค่าปรับกรมแรงงาน</li>
              <li>• ได้ Human Touch ที่ AI อย่างเดียวทำไม่ได้</li>
              <li>• แก้ปัญหา Call Center ขาดแคลน</li>
            </ul>
          </div>
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-[#1A1A1A]">👥 สำหรับคนพิการ</h4>
            <ul className="space-y-2 text-sm text-[#666]">
              <li>• งานคุณภาพสูง (ไม่ใช่ Admin หรือ House Keeper)</li>
              <li>• ได้เงินเยอะกว่า (Tele Sales, Call Center)</li>
              <li>• ใช้ AI ช่วยแก้ปัญหาความจำและทักษะ</li>
              <li>• สามารถทำงานได้แม้พูดไม่ได้หรือจำไม่ได้</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
} 