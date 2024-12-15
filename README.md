# Typhoon2-Audio
The repository of Typhoon2-Audio, speech/audio-language model that supports speech-in and speech-out. It is built upon the Typhoon2 LLM, and it is optimized for Thai and English languages.

<p align="center">
    <img src="assets/typhoon2_audio.png" width="25%"> <br>
</p>


## Usage

### Load Model
```python
import torch
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "scb10x/llama3.1-typhoon2-audio-8b-instruct-241213",
    torch_dtype=torch.float16, 
    trust_remote_code=True
)
model.to("cuda")
```

### Inference - Single turn example
```python
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
x = model.generate(
    conversation=conversation,
    max_new_tokens=500,
    do_sample=True,
    num_beams=1,
    top_p=0.9,
    repetition_penalty=1.0,
    length_penalty=1.0,
    temperature=0.7,
)
# x => x['text'] (text), x['audio'] (numpy array)
# to save the audio output
# import soundfile as sf
# sf.write("examples/speechout.wav", x["audio"]["array"], x["audio"]["sampling_rate"])
```

### Inference - Multi turn example
```python
conversation_multi_turn = [
    {
        "role": "system",
        "content": "You are a helpful female assistant named ไต้ฝุ่น. Respond conversationally to the speech provided in the language it is spoken in.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": "examples/tmp-2860cd0a094b64043226167340af03a3.wav",
                # บอกชื่อเมืองใหญ่ๆในอเมริกามาให้หน่อยสิ -- "List some names of US cities"
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
                # แล้วถ้าเป็นประเทศอังกฤษล่ะ -- "How about the UK"

            },
        ],
    },
]
x = model.generate(conversation=conversation_multi_turn)
# x => x['text'] (text), x['audio'] (numpy array)
# to save the audio output
# import soundfile as sf
# sf.write("examples/speechout.wav", x["audio"]["array"], x["audio"]["sampling_rate"])
```

### TTS functionality
```python
y = model.synthesize_speech("Hello, my name is ไต้ฝุ่น I am a language model specialized in Thai")
# y => numpy array
```

## To Do
- [x] Merge LoRA weights
- [x] Integrate Encoder + LLM + Generator + Vocoder
- [x] Local build to upload to HF
- [x] Implement `Typhoon2AudioForConditionalGeneration` and `Typhoon2Audio2AudioForConditionalGeneration`
- [x] Test loading normal and auto class
- [x] Implement `.forward()` for `Typhoon2AudioForConditionalGeneration`
- [x] Implement `.generate()` for `Typhoon2AudioForConditionalGeneration`
- [x] Implement `.forward()` for `Typhoon2Audio2AudioForConditionalGeneration`
- [x] Implement `.generate()` for `Typhoon2Audio2AudioForConditionalGeneration`
- [x] Allow streaming for `.generate()` for `Typhoon2AudioForConditionalGeneration`
- [ ] Allow streaming for `.generate()` for `Typhoon2Audio2AudioForConditionalGeneration`
- [x] Allow multi-turn for `.generate()` for `Typhoon2AudioForConditionalGeneration`
- [x] Allow multi-turn for `.generate()` for `Typhoon2Audio2AudioForConditionalGeneration`
- [x] Add TTS functionality to `Typhoon2Audio2AudioForConditionalGeneration`
- [x] Move prompt pattern to Qwen2-Audio input style: https://github.com/vllm-project/vllm/pull/9248
- [ ] Write doc & method string
- [ ] Allow flash_attention for LLM
- [ ] Allow `device_map="auto"`
- [ ] Make the code self-contained (LLM) -- tried but initialization is very slow
- [x] Make the code self-contained (Vocoder) -- done but requires import fairseq

## Build a model (only works locally)
```
python local_build.py
```

## To test locally
```
python test_load_model.py
python test_generate.py
python test_tts.py
```

## Acknowledgements

We are grateful to the previous open-source projects that provide useful resources for the development of Typhoon2-Audio, with notable projects including:
- SALMONN: https://github.com/bytedance/SALMONN
- Llama-Omni: https://github.com/ictnlp/LLaMA-Omni

## Citation
The first Typhoon-Audio work, focusing on improved understanding and instruction following as well as Thai performance):
```
@article{manakul2024enhancing,
  title={Enhancing low-resource language and instruction following capabilities of audio language models},
  author={Manakul, Potsawee and Sun, Guangzhi and Sirichotedumrong, Warit and Tharnpipitchai, Kasima and Pipatanakul, Kunat},
  journal={arXiv preprint arXiv:2409.10999},
  year={2024}
}
```