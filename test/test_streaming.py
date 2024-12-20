import sys
import os
# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typhoon2_audio.modeling_typhoon2audio import (
    DEFAULT_MODEL_SAMPLING_RATE,
    TensorStreamer,
    Typhoon2AudioForConditionalGeneration,
    Typhoon2Audio2AudioForConditionalGeneration,
)
from transformers import TextIteratorStreamer
import soundfile as sf
import torch
import numpy as np

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

model_name = "scb10x/llama3.1-typhoon2-audio-8b-instruct"
s2t_model = Typhoon2AudioForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

s2t_model.to("cuda")
print("s2t_model.dtype =", s2t_model.dtype)
print("s2t_model.device =", s2t_model.device)

streamer = TextIteratorStreamer(s2t_model.llama_tokenizer)
s2t_response = s2t_model.generate(conversation, streamer=streamer)

text = ""
for txt in s2t_response:
    print(txt)
    text += txt

print("\n---")
print("passed speech to text streaming")

s2s_model = Typhoon2Audio2AudioForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

s2s_model.to("cuda")
print("s2s_model.dtype =", s2s_model.dtype)
print("s2s_model.device =", s2s_model.device)

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

streamer_unit = TensorStreamer()

generated_text = ""
generated_wav = None
for wav, txt in s2s_model.generate_stream(
    conversation=conversation2,
    streamer=streamer,
    streamer_unit=streamer_unit,
    streaming_unit_gen=True,
):
    print(txt)
    generated_text += txt

    if wav is not None:
        if generated_wav is None:
            generated_wav = wav.cpu().numpy()
        else:
            generated_wav = np.concatenate((generated_wav, wav.cpu().numpy()))

generated_wav = generated_wav.reshape(1, -1)
sf.write(
    "tmp/generated.wav", generated_wav.T, DEFAULT_MODEL_SAMPLING_RATE, format="WAV"
)
