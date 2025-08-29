"use client";

import React from "react";

export default function CollaborativeTab() {
  return (
    <div className="space-y-8 animate-slide-up">
      {/* Collaborative Training Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-[#9C27B0] to-[#E91E63] rounded-full mb-6">
          <span className="text-3xl">🤝</span>
        </div>
        <h2 className="text-4xl font-bold text-[#1A1A1A] mb-3 font-anuphan">Collaborative Training</h2>
        <p className="text-[#666] text-xl max-w-4xl mx-auto leading-relaxed font-anuphan-medium">
          พนักงาน AIS และผู้พิการเทรน AI ร่วมกัน เพื่อให้ AI เข้าใจมุมมองของผู้พิการและตอบได้อย่างเหมาะสม
        </p>
        
        {/* Key Benefits */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6 max-w-5xl mx-auto">
          <div className="bg-gradient-to-r from-[#9C27B0]/10 to-[#9C27B0]/5 rounded-xl p-4 border border-[#9C27B0]/20">
            <div className="text-2xl mb-2">💬</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">ข้อความและภาพ</h4>
            <p className="text-[#666] text-xs">แชทข้อความและภาพประกอบ</p>
            <p className="text-[#9C27B0] text-xs font-medium">ลดอุปสรรคการสื่อสาร</p>
          </div>
          <div className="bg-gradient-to-r from-[#E91E63]/10 to-[#E91E63]/5 rounded-xl p-4 border border-[#E91E63]/20">
            <div className="text-2xl mb-2">📹</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">วิดีโอคอล + ซับไตเติล</h4>
            <p className="text-[#666] text-xs">Live Transcription อัตโนมัติ</p>
            <p className="text-[#E91E63] text-xs font-medium">เข้าใจชัดเจนขึ้น</p>
          </div>
          <div className="bg-gradient-to-r from-[#00A651]/10 to-[#00A651]/5 rounded-xl p-4 border border-[#00A651]/20">
            <div className="text-2xl mb-2">🎓</div>
            <h4 className="font-semibold text-[#1A1A1A] text-sm mb-1">การอบรมและสร้างความเข้าใจ</h4>
            <p className="text-[#666] text-xs">เรียนรู้วิธีการสื่อสารที่เหมาะสม</p>
            <p className="text-[#00A651] text-xs font-medium">ทำงานร่วมกันอย่างมีประสิทธิภาพ</p>
          </div>
        </div>
      </div>

      {/* Collaborative Training Interface */}
      <div className="card-jump card-jump-primary p-10 animate-scale-in">
        <div className="text-center mb-8">
          <h3 className="text-2xl font-bold text-[#1A1A1A] mb-3">หน้าจอการเทรน AI ร่วมกัน</h3>
          <p className="text-[#666] text-lg">พนักงาน AIS และผู้พิการสามารถเทรน AI ร่วมกันได้แบบ Real-time</p>
        </div>
        
        <div className="space-y-6">
          {/* Real-time Chat Interface */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#9C27B0]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              💬 Real-time Chat Interface
            </h4>
            <div className="space-y-4">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-gradient-to-br from-[#9C27B0] to-[#E91E63] rounded-full flex items-center justify-center text-white font-semibold">
                  AIS
                </div>
                <div className="flex-1 bg-[#9C27B0]/10 rounded-xl p-3">
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

          {/* Video Call with Subtitles */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#E91E63]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              📹 Video Call + Live Subtitles
            </h4>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-[#1A1A1A] mb-2">การตั้งค่าวิดีโอคอล</label>
                  <div className="space-y-2">
                    <button className="btn-jump-primary text-sm px-4 py-2 w-full">📹 เริ่มวิดีโอคอล</button>
                    <button className="btn-jump-secondary text-sm px-4 py-2 w-full">🎤 เปิด/ปิดไมค์</button>
                    <button className="btn-jump-accent text-sm px-4 py-2 w-full">📷 เปิด/ปิดกล้อง</button>
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-[#1A1A1A] mb-2">การตั้งค่าซับไตเติล</label>
                  <div className="space-y-2">
                    <button className="btn-jump-primary text-sm px-4 py-2 w-full">📝 เปิด Live Subtitles</button>
                    <button className="btn-jump-secondary text-sm px-4 py-2 w-full">🌐 เลือกภาษา</button>
                    <button className="btn-jump-accent text-sm px-4 py-2 w-full">⚙️ ปรับขนาดตัวอักษร</button>
                  </div>
                </div>
              </div>
              <div className="bg-[#E91E63]/10 rounded-xl p-4 border border-[#E91E63]/20">
                <p className="text-sm text-[#1A1A1A]">
                  <strong>Live Subtitles:</strong> "สวัสดีครับ ผมเป็นพนักงาน AIS ที่จะช่วยเทรน AI ร่วมกับคุณครับ 
                  วันนี้เราจะมาทำความเข้าใจว่าผู้พิการต้องการอะไรจาก AI Call Center ครับ"
                </p>
              </div>
            </div>
          </div>

          {/* AI Training Collaboration */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#00A651]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              🤖 AI Training Collaboration
            </h4>
            <div className="space-y-4">
              <div className="bg-[#00A651]/10 rounded-xl p-4 border border-[#00A651]/20">
                <h5 className="font-semibold text-[#1A1A1A] mb-2">ตัวอย่างการเทรน AI ร่วมกัน:</h5>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-[#9C27B0] rounded-full flex items-center justify-center text-white text-xs font-semibold">
                      AIS
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-[#1A1A1A] mb-1">"ลูกค้าผู้พิการถาม: 'ผมมีปัญหาการได้ยินครับ'"</p>
                      <p className="text-xs text-[#666]">AI ตอบ: "กรุณาพูดให้ชัดเจนครับ" ❌</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-[#00A651] rounded-full flex items-center justify-center text-white text-xs font-semibold">
                      👥
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-[#1A1A1A] mb-1">"AI ควรตอบว่า 'เข้าใจครับ ผมจะช่วยเหลือคุณ กรุณาบอกปัญหาที่ต้องการความช่วยเหลือครับ'" ✅</p>
                      <p className="text-xs text-[#666]">เหตุผล: อย่าบอกให้พูดให้ชัดเจน เพราะผู้พิการทางการได้ยินพูดไม่ได้</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-[#9C27B0] rounded-full flex items-center justify-center text-white text-xs font-semibold">
                      AIS
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-[#1A1A1A] mb-1">"เข้าใจแล้วครับ! เดี๋ยวผมเทรน AI ให้ตอบถูกต้องครับ"</p>
                      <p className="text-xs text-[#666]">บันทึกความรู้ใหม่: เมื่อลูกค้าผู้พิการทางการได้ยินโทรมา อย่าบอกให้พูดให้ชัดเจน</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Training Progress Tracking */}
          <div className="bg-gradient-to-r from-white/90 to-white/80 p-6 rounded-2xl border border-[#FFD700]/30 shadow-lg">
            <h4 className="text-xl font-semibold text-[#1A1A1A] mb-4 flex items-center gap-3">
              📊 Training Progress Tracking
            </h4>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-[#9C27B0] mb-2">15</div>
                  <div className="text-sm text-[#666]">เซสชันการเทรน</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-[#E91E63] mb-2">89%</div>
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

      {/* Communication Tools */}
      <div className="card-jump card-jump-secondary p-8 animate-fade-in">
        <h3 className="text-2xl font-bold text-[#1A1A1A] mb-6 text-center">เครื่องมือการสื่อสาร</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-[#1A1A1A]">💬 สำหรับผู้พิการทางการพูด</h4>
            <ul className="space-y-2 text-sm text-[#666]">
              <li>• แชทข้อความแบบ Real-time</li>
              <li>• ส่งภาพและไฟล์แนบ</li>
              <li>• ใช้ Emoji และสัญลักษณ์</li>
              <li>• ระบบ Auto-complete</li>
            </ul>
          </div>
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-[#1A1A1A]">👂 สำหรับผู้พิการทางการได้ยิน</h4>
            <ul className="space-y-2 text-sm text-[#666]">
              <li>• วิดีโอคอลพร้อมซับไตเติล</li>
              <li>• Live Transcription อัตโนมัติ</li>
              <li>• แสดงภาพประกอบคำอธิบาย</li>
              <li>• ระบบแจ้งเตือนด้วยแสง</li>
            </ul>
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