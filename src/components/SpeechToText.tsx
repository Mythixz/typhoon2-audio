"use client";

import React, { useState, useRef } from 'react';
import { postSpeechToText, type SpeechToTextResponse } from '@/lib/api';

interface SpeechToTextProps {
  onTranscriptionComplete: (text: string, emotion?: string) => void;
}

export default function SpeechToText({ onTranscriptionComplete }: SpeechToTextProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcription, setTranscription] = useState<string>('');
  const [emotion, setEmotion] = useState<string>('');
  const [confidence, setConfidence] = useState<number>(0);
  const [error, setError] = useState<string>('');
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      setError('');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await processAudio(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      setError('ไม่สามารถเข้าถึงไมโครโฟนได้');
      console.error('Error accessing microphone:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    try {
      // Convert blob to file
      const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
      
      // Send to backend
      const response: SpeechToTextResponse = await postSpeechToText(audioFile);
      
      setTranscription(response.text);
      setEmotion(response.emotion || '');
      setConfidence(response.confidence);
      
      // Call callback
      onTranscriptionComplete(response.text, response.emotion);
      
    } catch (err) {
      setError('เกิดข้อผิดพลาดในการแปลงเสียงเป็นข้อความ');
      console.error('Error processing audio:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const getEmotionColor = (emotion: string) => {
    switch (emotion) {
      case 'happy': return 'text-[#28A745] bg-[#28A745]/20 border-[#28A745]/30';
      case 'sad': return 'text-[#0066CC] bg-[#0066CC]/20 border-[#0066CC]/30';
      case 'angry': return 'text-[#DC3545] bg-[#DC3545]/20 border-[#DC3545]/30';
      case 'anxious': return 'text-[#FFC107] bg-[#FFC107]/20 border-[#FFC107]/30';
      case 'neutral': return 'text-[#6C757D] bg-[#6C757D]/20 border-[#6C757D]/30';
      default: return 'text-[#6C757D] bg-[#6C757D]/20 border-[#6C757D]/30';
    }
  };

  const getEmotionIcon = (emotion: string) => {
    switch (emotion) {
      case 'happy': return '😊';
      case 'sad': return '😢';
      case 'angry': return '😠';
      case 'anxious': return '😰';
      case 'neutral': return '😐';
      default: return '😐';
    }
  };

  const getEmotionText = (emotion: string) => {
    switch (emotion) {
      case 'happy': return 'ดีใจ';
      case 'sad': return 'เศร้า';
      case 'angry': return 'โกรธ';
      case 'anxious': return 'กังวล';
      case 'neutral': return 'ปกติ';
      default: return 'ไม่ทราบ';
    }
  };

  return (
    <div className="space-y-6">
      {/* Recording Controls */}
      <div className="flex items-center gap-6">
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isProcessing}
          className={`px-8 py-4 rounded-2xl font-semibold text-lg transition-all duration-300 transform hover:scale-105 ${
            isRecording
              ? 'bg-[#DC3545] hover:bg-[#C82333] text-white shadow-lg hover:shadow-xl'
              : 'bg-[#00A651] hover:bg-[#008F45] text-white shadow-lg hover:shadow-xl'
          } disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none`}
        >
          {isRecording ? (
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 bg-white rounded-full animate-pulse"></div>
              หยุดบันทึก
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <span className="text-2xl">🎤</span>
              เริ่มบันทึกเสียง
            </div>
          )}
        </button>
        
        {isProcessing && (
          <div className="flex items-center gap-3 text-[#0066CC] bg-[#0066CC]/10 px-4 py-3 rounded-xl border border-[#0066CC]/20">
            <div className="spinner-jump"></div>
            <span className="font-medium">กำลังประมวลผล...</span>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-[#DC3545]/10 border border-[#DC3545]/30 rounded-xl p-4 text-[#DC3545]">
          <div className="flex items-center gap-2">
            <span className="text-xl">⚠️</span>
            <span className="font-medium">{error}</span>
          </div>
        </div>
      )}

      {/* Results Display */}
      {transcription && (
        <div className="space-y-4">
          {/* Transcription Result */}
          <div className="bg-gradient-to-br from-white/90 to-white/70 backdrop-blur-sm border border-white/30 rounded-2xl p-6 shadow-xl">
            <h4 className="text-lg font-semibold text-[#1A1A1A] mb-4 flex items-center gap-2">
              <span className="text-[#00A651] text-xl">📝</span>
              ข้อความที่แปลงได้
            </h4>
            <p className="text-[#1A1A1A] text-lg leading-relaxed mb-4">{transcription}</p>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-[#666]">
                <span className="text-[#00A651]">🎯</span>
                <span className="font-medium">ความแม่นยำ:</span>
                <span className="text-[#00A651] font-bold">{(confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="w-px h-6 bg-gray-300"></div>
              <div className="text-sm text-[#666]">
                ใช้เวลา: {isProcessing ? '...' : '< 3 วินาที'}
              </div>
            </div>
          </div>

          {/* Emotion Detection */}
          {emotion && (
            <div className="bg-gradient-to-br from-white/90 to-white/70 backdrop-blur-sm border border-white/30 rounded-2xl p-6 shadow-xl">
              <h4 className="text-lg font-semibold text-[#1A1A1A] mb-4 flex items-center gap-2">
                <span className="text-[#0066CC] text-xl">🧠</span>
                อารมณ์ที่ตรวจพบ
              </h4>
              <div className="flex items-center gap-4">
                <span className="text-4xl">{getEmotionIcon(emotion)}</span>
                <div className="flex-1">
                  <div className={`badge-jump ${getEmotionColor(emotion)} text-base px-4 py-2`}>
                    {getEmotionText(emotion)}
                  </div>
                  <p className="text-[#666] text-sm mt-2">
                    AI ตรวจพบอารมณ์จากน้ำเสียงและคำศัพท์ที่ใช้
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tips Section */}
      <div className="bg-gradient-to-br from-[#E8F5E8] to-[#E6F3FF] border border-[#00A651]/20 rounded-2xl p-6">
        <div className="flex items-start gap-3">
          <span className="text-[#00A651] text-2xl">💡</span>
          <div>
            <h5 className="font-semibold text-[#1A1A1A] mb-2">เคล็ดลับสำหรับผลลัพธ์ที่ดีที่สุด</h5>
            <ul className="text-[#666] space-y-1 text-sm">
              <li>• พูดช้าๆ และชัดเจน</li>
              <li>• ใช้ไมโครโฟนคุณภาพดี</li>
              <li>• หลีกเลี่ยงเสียงรบกวน</li>
              <li>• พูดในระยะห่างที่เหมาะสม (15-30 ซม.)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Recording Status */}
      {isRecording && (
        <div className="fixed bottom-6 right-6 bg-[#00A651] text-white px-6 py-3 rounded-full shadow-2xl animate-pulse">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
            <span className="font-medium">กำลังบันทึกเสียง...</span>
          </div>
        </div>
      )}
    </div>
  );
}
