# Typhoon2-Audio
The repository of Typhoon2-Audio, Thai audio-language model that supports speech-in and speech-out

## To do
[x] Merge LoRA weights
[x] Integrate Encoder + LLM + Generator + Vocoder
[x] Local build to upload to HF
[x] Implement `Typhoon2AudioForConditionalGeneration` and `Typhoon2Audio2AudioForConditionalGeneration`
[x] Test loading normal and auto class
[x] Implement `.generate()` for `Typhoon2AudioForConditionalGeneration`
[] Implement `.generate()` for `Typhoon2Audio2AudioForConditionalGeneration`
[] Implement `.forward()` for `Typhoon2AudioForConditionalGeneration`
[] Allow flash_attention for LLM
[] Allow `device_map="auto"`
[] Make the code self-contained (LLM)
[] Make the code self-contained (Vocoder)

```
cp typhoon2_audio/configuration_typhoon2audio.py save_weights/typhoon2-audio-241208/.
cp typhoon2_audio/modeling_typhoon2audio.py save_weights/typhoon2-audio-241208/.
```