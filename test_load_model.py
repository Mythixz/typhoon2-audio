from typhoon2_audio.modeling_typhoon2audio import Typhoon2AudioForConditionalGeneration, Typhoon2Audio2AudioForConditionalGeneration
from transformers import AutoModel

model1 = Typhoon2Audio2AudioForConditionalGeneration.from_pretrained("save_weights/llama3.1-typhoon2-audio-8b-instruct-241210/")
print("load local1 okay")

model2 = Typhoon2AudioForConditionalGeneration.from_pretrained("save_weights/llama3.1-typhoon2-audio-8b-instruct-241210/")
print("load local2 okay")

model3 = AutoModel.from_pretrained("scb10x/llama3.1-typhoon2-audio-8b-instruct-241210",trust_remote_code=True)
# model2 = AutoModel.from_pretrained("potsawee/typhoon2-audio-241208", trust_remote_code=True)
# model1 = AutoModel.from_pretrained("scb10x/llama-3-typhoon-audio-8b-2411", trust_remote_code=True)
print("load auto okay")
