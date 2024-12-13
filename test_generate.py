from typhoon2_audio.modeling_typhoon2audio import (
    Typhoon2AudioForConditionalGeneration,
    Typhoon2Audio2AudioForConditionalGeneration,
)
from transformers import AutoModel
import soundfile as sf
import torch

conversation = [
    {"role": "system", "content": "You are a helpful female assistant named ไต้ฝุ่น."},
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": "examples/tmp-2860cd0a094b64043226167340af03a3.wav",
            },
            {"type": "text", "text": "Transcribe this audio"},
        ],
    },
]

model0 = Typhoon2AudioForConditionalGeneration.from_pretrained(
    "scb10x/llama3.1-typhoon2-audio-8b-instruct-241210",
    torch_dtype=torch.float16,  # otherwise default to float32
)

model0.to("cuda")
print("model0.dtype =", model0.dtype)
print("model0.device =", model0.device)

response0 = model0.generate(conversation)

print(response0)

model = Typhoon2Audio2AudioForConditionalGeneration.from_pretrained(
    "scb10x/llama3.1-typhoon2-audio-8b-instruct-241210",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

model.to("cuda")
print("model.dtype =", model.dtype)
print("model.device =", model.device)

conversation2 = [
    {"role": "system", "content": "You are a helpful female assistant named ไต้ฝุ่น."},
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": "examples/tmp-2860cd0a094b64043226167340af03a3.wav",
            },
            {
                "type": "text",
                "text": "Respond conversationally to the speech provided in the language it is spoken in.",
            },
        ],
    },
]

x = model.generate(conversation=conversation2)
sf.write("examples/speechout.wav", x["audio"]["array"], x["audio"]["sampling_rate"])
print(x)
print("passed: test single-turn")


conversation2_multi_turn = [
    {"role": "system", "content": "You are a helpful female assistant named ไต้ฝุ่น. Respond conversationally to the speech provided in the language it is spoken in."},
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": "examples/tmp-2860cd0a094b64043226167340af03a3.wav",
            },
            {
                "type": "text",
                "text": "",
            },
        ],  
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "โอเคค่ะ, ฉันจะบอกชื่อเมืองใหญ่ๆ ในอเมริกาให้คุณฟัง:\n\n1. นิวยอร์ก\n2. ลอสแอนเจลิส\n3. ชิคาโก\n4. ฮิวสตัน\n5. ฟิลาเดลเฟีย\n6. บอสตัน\n7. ซานฟรานซิสโก\n8. วอชิงตัน ดี.ซี. (Washington D.C.)\n9. แอตแลนต้า\n10. ซีแอตเทิล\n\nถ้าคุณต้องการข้อมูลเพิ่มเติมหรือมีคำถามอื่นๆ กรุณาถามได้เลยค่ะ'",
            },
        ],  
    },
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": "examples/tmp-2284cd76e1c875525ff75327a2fc3610.wav",
            },
            {
                "type": "text",
                "text": "",
            },
        ],  
    },
]
x = model.generate(conversation=conversation2_multi_turn)

print(x)
print("passed: test multi-turn")

import ipdb; ipdb.set_trace()
print("finish testing generate")
