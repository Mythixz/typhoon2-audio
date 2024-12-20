import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import WhisperModel
from typhoon2_audio.configuration_typhoon2audio import Typhoon2AudioConfig
from typhoon2_audio.modeling_typhoon2audio import Typhoon2AudioForConditionalGeneration, Typhoon2Audio2AudioForConditionalGeneration


print("Instatiation: Typhoon2-Audio")
config = Typhoon2AudioConfig()
model = Typhoon2Audio2AudioForConditionalGeneration(config=config)

print("Load BEATs")
# load_beats(beats_path):
beats_checkpoint = torch.load("/workspace/typhoonaudio_hf/typhoonaudio/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt", map_location='cpu')
model.beats.load_state_dict(beats_checkpoint['model'])

# load Whisper Encoder
print("Load Whisper")
model.speech_encoder = WhisperModel.from_pretrained("biodatlab/whisper-th-large-v3-combined").encoder

print("Load LoRA")
# load llama (LoRA) checkpoint
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=True, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.0,
)
model.llama_model = get_peft_model(model.llama_model, peft_config)
lora_ckpt_dict = torch.load("/workspace/typhoon-audio-salmonn/outputs/typhoon_audio2_exp3_sft_v3.13_lora/202411111855/checkpoint_2.pth")['model']
model.load_state_dict(lora_ckpt_dict, strict=False)
model.llama_model = model.llama_model.merge_and_unload()

# load speech decoder
print("Load Generator")
decoder_ckpt_dict = torch.load("/workspace/exp-punpun/omni-trainer/runs/typhoon-audio-sdec-exp8.6s/checkpoint-29046/speech_decoder_step_29046.pth")
model.speech_generator.load_state_dict(decoder_ckpt_dict)

model.eval()
model.half() # convert to float16

# initialize & load vocoder
print("Load Unit Vocoder")
model.init_vocoder(
    checkpoint_path="/workspace3/exp-boom/unit-vocoder/unit-vocoder-trainer/checkpoints/gcp_vocoder_v1/g_00206600"
)

print("registering: Typhoon2-Audio")
Typhoon2AudioConfig.register_for_auto_class()
Typhoon2Audio2AudioForConditionalGeneration.register_for_auto_class("AutoModel")

print("saving: Typhoon2-Audio")
# Save Local
model.save_pretrained("./save_weights/llama3.1-typhoon2-audio-8b-instruct-241213", safe_serialization=False)

# Upload to HF
model.push_to_hub("scb10x/llama3.1-typhoon2-audio-8b-instruct-241213", safe_serialization=False)

print("built: Typhoon2-Audio")