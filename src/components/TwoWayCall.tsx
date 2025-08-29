"use client";

import React, { useState, useRef, useEffect } from 'react';
import { postSpeak } from '@/lib/api';

interface TwoWayCallProps {
  onCallEnd: () => void;
}

export default function TwoWayCall({ onCallEnd }: TwoWayCallProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [callStatus, setCallStatus] = useState<'idle' | 'connecting' | 'connected' | 'ended'>('idle');
  const [emotion, setEmotion] = useState('');
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const mockAIResponse = "สวัสดีครับ ยินดีที่ได้พูดคุยกับคุณ ผมเป็น AI Call Center ที่พร้อมให้บริการครับ";

  const startCall = async () => {
    try {
      setCallStatus('connecting');
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
        setTranscription('สวัสดีครับ ผมต้องการสอบถามข้อมูล');
        setEmotion('positive');
        setTimeout(() => {
          setAiResponse(mockAIResponse);
          playAIResponse(mockAIResponse);
        }, 1000);
      };

      setCallStatus('connected');
    } catch (error) {
      console.error('Error starting call:', error);
      setCallStatus('idle');
      alert('ไม่สามารถเข้าถึงไมโครโฟนได้');
    }
  };

  const stopCall = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (mediaRecorderRef.current?.stream) {
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
    setCallStatus('ended');
    setIsRecording(false);
    onCallEnd();
  };

  const startRecording = () => {
    if (mediaRecorderRef.current && callStatus === 'connected') {
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setTranscription('');
      setAiResponse('');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const playAIResponse = async (text: string) => {
    try {
      const response = await postSpeak(text);
      if (response.tts_audio_url && audioRef.current) {
        audioRef.current.src = response.tts_audio_url;
        audioRef.current.play();
        setIsPlaying(true);
      }
    } catch (error) {
      console.error('Error playing AI response:', error);
    }
  };

  const handleAudioEnded = () => setIsPlaying(false);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current?.stream) {
        mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="space-y-6">
      <div className="text-center">
        <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium ${
          callStatus === 'idle' ? 'bg-gray-100 text-gray-600' :
          callStatus === 'connecting' ? 'bg-yellow-100 text-yellow-600' :
          callStatus === 'connected' ? 'bg-green-100 text-green-600' :
          'bg-red-100 text-red-600'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            callStatus === 'idle' ? 'bg-gray-400' :
            callStatus === 'connecting' ? 'bg-yellow-400 animate-pulse' :
            callStatus === 'connected' ? 'bg-green-400' :
            'bg-red-400'
          }`}></div>
          {callStatus === 'idle' && 'พร้อมเริ่มการสนทนา'}
          {callStatus === 'connecting' && 'กำลังเชื่อมต่อ...'}
          {callStatus === 'connected' && 'เชื่อมต่อแล้ว'}
          {callStatus === 'ended' && 'การสนทนาสิ้นสุด'}
        </div>
      </div>

      <div className="flex justify-center gap-4">
        {callStatus === 'idle' && (
          <button onClick={startCall} className="btn-jump-primary px-8 py-4 text-lg font-medium">
            📞 เริ่มการสนทนา
          </button>
        )}

        {callStatus === 'connected' && (
          <>
            <button
              onClick={startRecording}
              disabled={isRecording}
              className={`btn-jump-accent px-6 py-4 text-lg font-medium ${
                isRecording ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              🎤 {isRecording ? 'กำลังบันทึก...' : 'เริ่มบันทึก'}
            </button>

            {isRecording && (
              <button onClick={stopRecording} className="btn-jump-secondary px-6 py-4 text-lg font-medium">
                ⏹️ หยุดบันทึก
              </button>
            )}

            <button onClick={stopCall} className="btn-jump-outline px-6 py-4 text-lg font-medium">
              📞 จบการสนทนา
            </button>
          </>
        )}

        {callStatus === 'ended' && (
          <button onClick={() => setCallStatus('idle')} className="btn-jump-primary px-8 py-4 text-lg font-medium">
            🔄 เริ่มใหม่
          </button>
        )}
      </div>

      {transcription && (
        <div className="bg-gradient-to-r from-[#00A651]/10 to-[#00A651]/5 rounded-2xl p-6 border border-[#00A651]/20">
          <h4 className="text-lg font-semibold text-[#1A1A1A] mb-3 flex items-center gap-2">🎤 ข้อความที่คุณพูด</h4>
          <p className="text-[#1A1A1A] text-lg leading-relaxed">{transcription}</p>
          {emotion && (
            <div className="mt-3">
              <span className="text-sm text-[#666]">อารมณ์ที่ตรวจพบ: </span>
              <span className="inline-flex items-center gap-1 bg-gradient-to-r from-[#00A651]/20 to-[#0066CC]/20 px-3 py-1 rounded-full text-sm font-medium text-[#00A651]">
                {emotion === 'positive' ? '😊 บวก' : emotion === 'negative' ? '😔 ลบ' : '😐 เป็นกลาง'}
              </span>
            </div>
          )}
        </div>
      )}

      {aiResponse && (
        <div className="bg-gradient-to-r from-[#0066CC]/10 to-[#0066CC]/5 rounded-2xl p-6 border border-[#0066CC]/20">
          <h4 className="text-lg font-semibold text-[#1A1A1A] mb-3 flex items-center gap-2">🤖 คำตอบจาก AI</h4>
          <p className="text-[#1A1A1A] text-lg leading-relaxed">{aiResponse}</p>
          <div className="mt-4 flex items-center gap-3">
            <button
              onClick={() => playAIResponse(aiResponse)}
              disabled={isPlaying}
              className={`btn-jump-secondary px-4 py-2 text-sm ${isPlaying ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              🔊 {isPlaying ? 'กำลังเล่น...' : 'ฟังอีกครั้ง'}
            </button>
            {isPlaying && (
              <div className="flex items-center gap-2 text-sm text-[#666]">
                <div className="w-4 h-4 border-2 border-[#0066CC] border-t-transparent rounded-full animate-spin"></div>
                กำลังเล่นเสียง...
              </div>
            )}
          </div>
        </div>
      )}

      <audio ref={audioRef} onEnded={handleAudioEnded} className="hidden" />
    </div>
  );
}
