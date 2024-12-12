# Typhoon2-Audio
The repository of Typhoon2-Audio, Thai audio-language model that supports speech-in and speech-out

## Usage

```python
# load model
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "scb10x/llama3.1-typhoon2-audio-8b-instruct-241210",
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

# inference
prompt_pattern="<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant named ไต้ฝุ่น.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<Speech><SpeechHere></Speech> {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
prompt="Respond conversationally to the speech provided.",
audio, sr = sf.read("speech_input.wav")
x = model.generate(audio=audio, prompt=prompt, prompt_pattern=prompt_pattern)
# x => x['text'] (text), x['audio'] (numpy array)

# TTS functionality
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
- [ ] Allow multi-turn for `.generate()` for `Typhoon2AudioForConditionalGeneration`
- [ ] Allow multi-turn for `.generate()` for `Typhoon2Audio2AudioForConditionalGeneration`
- [x] Add TTS functionality to `Typhoon2Audio2AudioForConditionalGeneration`
- [ ] Move prompt pattern to Qwen2-Audio input style: https://github.com/vllm-project/vllm/pull/9248
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
