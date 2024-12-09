from typhoon2_audio.modeling_typhoon2audio import Typhoon2AudioForConditionalGeneration, Typhoon2Audio2AudioForConditionalGeneration
from typhoon2_audio.configuration_typhoon2audio import Typhoon2AudioConfig
from transformers import AutoModel, WhisperModel
import soundfile as sf
import librosa
import torch
from peft import LoraConfig, TaskType, get_peft_model

# model = Typhoon2Audio2AudioForConditionalGeneration.from_pretrained(
#     "save_weights/typhoon2-audio-241208/",
#     torch_dtype=torch.float16, # otherwise default to float32
# )
# ----------------------------------------------------------------------- #
# config = Typhoon2AudioConfig()
# model = Typhoon2AudioForConditionalGeneration(config=config)

# print("Load BEATs")
# # load_beats(beats_path):
# beats_checkpoint = torch.load("/workspace/typhoonaudio_hf/typhoonaudio/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt", map_location='cpu')
# model.beats.load_state_dict(beats_checkpoint['model'])

# print("Load Whisper")
# model.speech_encoder = WhisperModel.from_pretrained("biodatlab/whisper-th-large-v3-combined" ).encoder

# print("Load LoRA")
# # load llama (LoRA) checkpoint
# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM, 
#     inference_mode=True, 
#     r=8, 
#     lora_alpha=32, 
#     lora_dropout=0.0,
# )
# model.llama_model = get_peft_model(model.llama_model, peft_config)
# lora_ckpt_dict = torch.load("/workspace/typhoon-audio-salmonn/outputs/typhoon_audio2_exp3_sft_v3.13_lora/202411111855/checkpoint_2.pth")['model']
# model.load_state_dict(lora_ckpt_dict, strict=False)
# model.llama_model = model.llama_model.merge_and_unload()
# model.eval()
# model.half()
# ----------------------------------------------------------------------- #

# read wav
audio, sr = sf.read("tmp/tmp-2860cd0a094b64043226167340af03a3.wav")
prompt_pattern="<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful female assistant named ไต้ฝุ่น.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<Speech><SpeechHere></Speech> {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# outputs = model.forward(
#     audio=audio,
#     prompt="Respond conversationally to this audio in Thai",
#     prompt_pattern=prompt_pattern,
# )

# response = model.generate(
#     audio=audio,
#     prompt="Respond conversationally to this audio in Thai",
#     # prompt="transcribe this audio",
#     prompt_pattern=prompt_pattern,
#     do_sample=False,
#     max_new_tokens=512,
#     repetition_penalty=1.1,
#     num_beams=1,
#     # temperature=0.4,
#     # top_p=0.9,
# )

model0 = Typhoon2AudioForConditionalGeneration.from_pretrained(
    "save_weights/typhoon2-audio-241208/",
    torch_dtype=torch.float16, # otherwise default to float32
)

model0.to("cuda")
print("model0.dtype =", model0.dtype)
print("model0.device =", model0.device)

response0 = model0.generate(
    audio=audio,
    prompt="Transcribe this audio",
    prompt_pattern=prompt_pattern,
)

# print(response0)
model = Typhoon2Audio2AudioForConditionalGeneration.from_pretrained(
    "save_weights/typhoon2-audio-241208/",
    torch_dtype=torch.float16, # otherwise default to float32
)
model.init_vocoder()

model.to("cuda")
print("model.dtype =", model.dtype)
print("model.device =", model.device)

x = model.generate(
    audio=audio,
    prompt="Respond conversationally to the speech provided in the language it is spoken in.",
    prompt_pattern=prompt_pattern,
)

import ipdb; ipdb.set_trace()
print("finish testing generate")