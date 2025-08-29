"use client";

import React from "react";

export default function CRMTab() {
  return (
    <div className="space-y-8 animate-slide-up">
      {/* CRM Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-[#FF6B6B] to-[#4ECDC4] rounded-full mb-6">
          <span className="text-3xl">📊</span>
        </div>
        <h2 className="text-4xl font-bold text-[#1A1A1A] mb-3 font-anuphan">CRM System</h2>
        <p className="text-[#666] text-xl max-w-3xl mx-auto leading-relaxed font-anuphan-medium">
          ระบบจัดการลูกค้าแบบครบวงจร พร้อมการติดตามประวัติและความต้องการพิเศษ
        </p>
      </div>

      {/* CRM Dashboard */}
      <div className="card-jump card-jump-primary p-10 animate-scale-in">
        <div className="text-center mb-8">
          <h3 className="text-2xl font-bold text-[#1A1A1A] mb-3">Dashboard จัดการลูกค้า</h3>
          <p className="text-[#666] text-lg">จัดการข้อมูลลูกค้า ประวัติการโทร และความต้องการพิเศษ</p>
        </div>
        
        <div className="space-y-6">
          {/* Customer Overview */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#FF6B6B]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              👥 Customer Overview
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-[#FF6B6B] mb-2">1,247</div>
                <div className="text-sm text-[#666]">ลูกค้าทั้งหมด</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-[#4ECDC4] mb-2">89</div>
                <div className="text-sm text-[#666]">ลูกค้าใหม่</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-[#45B7D1] mb-2">156</div>
                <div className="text-sm text-[#666]">ลูกค้า VIP</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-[#96CEB4] mb-2">4.8</div>
                <div className="text-sm text-[#666]">คะแนนความพึงพอใจ</div>
              </div>
            </div>
          </div>

          {/* Customer Management */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#4ECDC4]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              🔍 Customer Management
            </h4>
            <div className="space-y-4">
              <div className="flex gap-3">
                <button className="btn-jump-primary text-sm px-4 py-2">➕ เพิ่มลูกค้าใหม่</button>
                <button className="btn-jump-secondary text-sm px-4 py-2">✏️ แก้ไขข้อมูล</button>
                <button className="btn-jump-accent text-sm px-4 py-2">🔍 ค้นหาลูกค้า</button>
                <button className="btn-jump-outline text-sm px-4 py-2">📊 สถิติ</button>
              </div>
              <div className="bg-[#4ECDC4]/10 rounded-xl p-4 border border-[#4ECDC4]/20">
                <p className="text-sm text-[#1A1A1A]">
                  <strong>ตัวอย่าง:</strong> คนพิการสามารถจัดการข้อมูลลูกค้าได้อย่างมีประสิทธิภาพ 
                  โดยใช้ AI ช่วยในการจัดหมวดหมู่และติดตามความต้องการพิเศษ
                </p>
              </div>
            </div>
          </div>

          {/* Call History */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#45B7D1]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              📞 Call History
            </h4>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-[#1A1A1A] mb-2">ค้นหาประวัติการโทร</label>
                  <div className="space-y-2">
                    <input className="input-jump w-full" placeholder="ชื่อลูกค้าหรือเบอร์โทร" />
                    <button className="btn-jump-primary text-sm px-4 py-2 w-full">🔍 ค้นหา</button>
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-[#1A1A1A] mb-2">กรองตามประเภท</label>
                  <select className="input-jump w-full">
                    <option>ทั้งหมด</option>
                    <option>โทรเข้า</option>
                    <option>โทรออก</option>
                    <option>ลูกค้า VIP</option>
                  </select>
                </div>
              </div>
              <div className="bg-[#45B7D1]/10 rounded-xl p-4 border border-[#45B7D1]/20">
                <p className="text-sm text-[#1A1A1A]">
                  <strong>Call History:</strong> ระบบจะบันทึกประวัติการโทรทุกครั้ง พร้อมสรุปการสนทนา 
                  อารมณ์ของลูกค้า และระดับความพึงพอใจ
                </p>
              </div>
            </div>
          </div>

          {/* Special Needs Management */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#96CEB4]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              ♿ Special Needs Management
            </h4>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-[#1A1A1A] mb-2">ความต้องการพิเศษ</label>
                  <div className="space-y-2">
                    <button className="btn-jump-primary text-sm px-4 py-2 w-full">👂 ผู้พิการทางการได้ยิน</button>
                    <button className="btn-jump-secondary text-sm px-4 py-2 w-full">💬 ผู้พิการทางการสื่อสาร</button>
                    <button className="btn-jump-accent text-sm px-4 py-2 w-full">🧠 ผู้พิการทางสติปัญญา</button>
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-[#1A1A1A] mb-2">การสื่อสารที่เหมาะสม</label>
                  <select className="input-jump w-full">
                    <option>เสียงพูดปกติ</option>
                    <option>ข้อความ</option>
                    <option>เสียงสังเคราะห์</option>
                    <option>วิดีโอคอล + ซับไตเติล</option>
                  </select>
                </div>
              </div>
              <div className="bg-[#96CEB4]/10 rounded-xl p-4 border border-[#96CEB4]/20">
                <p className="text-sm text-[#1A1A1A]">
                  <strong>Special Needs:</strong> ระบบจะบันทึกความต้องการพิเศษของลูกค้าแต่ละคน 
                  เพื่อให้การบริการเหมาะสมและเข้าถึงได้
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 