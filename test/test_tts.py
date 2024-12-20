import sys
import os
# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typhoon2_audio.modeling_typhoon2audio import Typhoon2AudioForConditionalGeneration, Typhoon2Audio2AudioForConditionalGeneration
from typhoon2_audio.configuration_typhoon2audio import Typhoon2AudioConfig
from transformers import AutoModel
import soundfile as sf
import librosa
import torch

# print(response0)
model = Typhoon2Audio2AudioForConditionalGeneration.from_pretrained(
    "save_weights/llama3.1-typhoon2-audio-8b-instruct/",
    torch_dtype=torch.float16, # otherwise default to float32
)

model.to("cuda")
print("model.dtype =", model.dtype)
print("model.device =", model.device)

text = "ธนาคารไทยพาณิชย์ ตอกย้ำกลยุทธ์ Digital Bank with Human Touch มุ่งสู่การเป็นดิจิทัลแบงก์ที่เป็นอันดับหนึ่งด้านการบริหารความมั่งคั่ง พร้อมมอบประสบการณ์การให้บริการที่เข้าถึงใจ"
y = model.synthesize_speech(text)
import ipdb; ipdb.set_trace()
sf.write("tmp/output1.wav", y['array'], y['sampling_rate'])
print("finish testing TTS")