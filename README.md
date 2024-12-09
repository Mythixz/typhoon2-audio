# Typhoon2-Audio
The repository of Typhoon2-Audio, Thai audio-language model that supports speech-in and speech-out

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
- [ ] Write doc & method string
- [ ] Allow flash_attention for LLM
- [ ] Allow `device_map="auto"`
- [ ] Make the code self-contained (LLM)
- [ ] Make the code self-contained (Vocoder)

## To test locally
```
python test_load_model.py
python test_generation.py
```
```
# to test AutoModel
cp typhoon2_audio/configuration_typhoon2audio.py save_weights/typhoon2-audio-241208/.
cp typhoon2_audio/modeling_typhoon2audio.py save_weights/typhoon2-audio-241208/.
cp typhoon2_audio/configuration_typhoon2audio.py /cache/.cache/huggingface/modules/transformers_modules/.
cp typhoon2_audio/modeling_typhoon2audio.py /cache/.cache/huggingface/modules/transformers_modules/.
```
