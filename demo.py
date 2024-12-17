from typhoon2_audio.modeling_typhoon2audio import (
    DEFAULT_MODEL_SAMPLING_RATE,
    Typhoon2Audio2AudioForConditionalGeneration,
)

import shutil
import os
import gradio as gr
from gradio import ChatMessage
import numpy as np
import torch
import xxhash
import soundfile as sf
from dotenv import load_dotenv
from datasets import Audio

load_dotenv()

OUT_PATH = "out"

if os.path.exists(OUT_PATH):
    shutil.rmtree(OUT_PATH)
    os.makedirs(OUT_PATH)

model = Typhoon2Audio2AudioForConditionalGeneration.from_pretrained(
    "scb10x/llama3.1-typhoon2-audio-8b-instruct-241213",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.to("cuda")

resampler = Audio(sampling_rate=DEFAULT_MODEL_SAMPLING_RATE)

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#f7f7fd",
        c100="#dfdef8",
        c200="#c4c1f2",
        c300="#a29eea",
        c400="#8f8ae6",
        c500="#756fe0",
        c600="#635cc1",
        c700="#4f4a9b",
        c800="#433f83",
        c900="#302d5e",
        c950="#302d5e",
    ),
    secondary_hue="rose",
    neutral_hue="stone",
)


DEFAULT_SYSTEM_PROMPT = "You are a helpful female assistant named ไต้ฝุ่น."
messages = []


def run_inference(audio_input, system_prompt, chat_box):
    if audio_input is None:
        return (
            gr.update(value=None),
            gr.Button(value="Record Audio to Submit!", interactive=False),
            "",
            gr.update(visible=False),
        )
    sr, y = audio_input
    y = y.astype(np.float32)
    y = y / np.max(np.abs(y))
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )
    array_hash = str(xxhash.xxh32(a["array"].tostring()).hexdigest())
    wav_path = f"./tmp/{array_hash}.wav"
    os.makedirs("tmp", exist_ok=True)
    if not os.path.exists(wav_path):
        sf.write(wav_path, a["array"], a["sampling_rate"], format="wav")

    messages.append(
        {"role": "user", "content": [{"type": "audio", "audio_url": wav_path}]}
    )

    updated_history = [{"role": "system", "content": system_prompt}] + messages

    # Typhoon-Audio model
    response = model.generate(conversation=updated_history)

    text = response["text"]
    audio = response["audio"]
    temp_out_path = f"./tmp/out-{array_hash}.wav"
    audio_arr = audio["array"].astype(np.float32)
    sf.write(temp_out_path, audio_arr, audio["sampling_rate"], format="wav")

    output_widget = gr.Audio(temp_out_path)
    chat_box.append(ChatMessage(role="user", content=gr.Audio(wav_path)))
    chat_box.append(ChatMessage(role="assistant", content=text))
    chat_box.append(ChatMessage(role="assistant", content=output_widget))

    messages.append(
        {
            "role": "assistant",
            "content": text,
        }
    )

    return (
        gr.update(value=None),
        gr.Button(value="Please upload audio", interactive=False, variant="primary"),
        chat_box,
        gr.update(visible=True),
    )


def is_able_to_start(audio_input):
    if audio_input is not None:
        return gr.Button(
            value="Click to run inference", interactive=True, variant="primary"
        )
    return gr.Button(value="Please upload audio", interactive=False, variant="primary")


def gradio_reset_all():
    messages = []

    return (
        gr.update(value=None),
        gr.update(value="Record Audio to Submit!", interactive=False),
        gr.update(value=None),
        gr.update(value=DEFAULT_SYSTEM_PROMPT),
        gr.update(visible=False),
    )


with gr.Blocks(theme=theme) as omni_demo:
    gr.HTML(
        """
    <div style='text-align: center'>
        <h1>
            Converse With Typhoon2 Audio
        </h1>
    </div>
    """
    )
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                system_prompt = gr.Textbox(
                    value=DEFAULT_SYSTEM_PROMPT,
                    label="System Prompt",
                    placeholder="You can control the model's behavior by specifying the system prompt here.",
                )
            with gr.Row():
                cur_dir = os.path.dirname(os.path.abspath(__file__))
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    streaming=False,
                    label="Microphone Input",
                )
            with gr.Row():
                submit_button = gr.Button(
                    value="Record Audio to Submit!", interactive=False
                )

            with gr.Row():
                clear_button = gr.Button(value="Clear the recording", visible=False)
        with gr.Column(scale=8):
            with gr.Row():
                chat_box = gr.Chatbot(type="messages", height=1000)

            audio_input.change(is_able_to_start, [audio_input], [submit_button])

            submit_button.click(
                fn=run_inference,
                inputs=[audio_input, system_prompt, chat_box],
                outputs=[audio_input, submit_button, chat_box, clear_button],
            )

            clear_button.click(
                fn=gradio_reset_all,
                inputs=[],
                outputs=[
                    audio_input,
                    submit_button,
                    chat_box,
                    system_prompt,
                    clear_button,
                ],
                queue=False,
            )


# --- Processing Demo ---
def is_able_to_start(audio_input):
    if audio_input is not None:
        return gr.Button(
            value="Click to run inference", interactive=True, variant="primary"
        )
    return gr.Button(value="Please upload audio", interactive=False, variant="primary")


def gradio_reset_all():
    return (
        gr.update(value=None),
        gr.update(
            value="",
            label="Text Prompt",
            placeholder="Optional (blank = speech instruction following), แปลงเสียงเป็นข้อความ, อธิบายเสียง, etc",
        ),
        gr.update(value="Record Audio to Submit!", interactive=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def run_inference(audio_input, text_prompt):
    yield (
        gr.Button(
            value="Waiting in queue for GPU time...",
            interactive=False,
            variant="primary",
        ),
        "",
        gr.update(visible=False),
    )
    if audio_input is None:
        return (
            gr.Button(value="Record Audio to Submit!", interactive=False),
            "",
            gr.update(visible=False),
        )
    sr, y = audio_input
    y = y.astype(np.float32)
    y = y / np.max(np.abs(y))
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )
    array_hash = str(xxhash.xxh32(a["array"].tostring()).hexdigest())
    wav_path = f"out/{array_hash}.wav"

    if not os.path.exists(wav_path):
        sf.write(wav_path, a["array"], a["sampling_rate"], format="wav")

    typhoon_output = model.generate(
        conversation=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "audio", "audio_url": wav_path},
                ],
            }
        ]
    )
    yield (
        gr.Button(value="Click to run inference", interactive=True, variant="primary"),
        gr.Textbox(
            value=typhoon_output["text"],
            visible=True,
        ),
        gr.update(visible=True),
    )


text_input = ""
with gr.Blocks(theme=theme) as processing_demo:
    gr.HTML(
        """
    <div style='text-align: center'>
        <h1>
            Audio Processing With Typhoon2 Audio
        </h1>
    </div>
    """
    )
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                cur_dir = os.path.dirname(os.path.abspath(__file__))
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    streaming=False,
                    label="Microphone Input",
                )

        with gr.Column(scale=8):
            with gr.Row():
                text_input = gr.Textbox(
                    value="",
                    label="Text Prompt",
                    placeholder="Optional (blank = speech instruction following), แปลงเสียงเป็นข้อความ, อธิบายเสียง, etc",
                )
            with gr.Row():
                submit_button = gr.Button(
                    value="Record Audio to Submit!", interactive=False
                )

            with gr.Row():
                output_ta = gr.Textbox(visible=False)

            with gr.Row():
                clear_button = gr.Button(value="Clear the recording", visible=False)

            audio_input.change(is_able_to_start, [audio_input], [submit_button])

            submit_button.click(
                fn=run_inference,
                inputs=[audio_input, text_input],
                outputs=[submit_button, output_ta, clear_button],
            )

            clear_button.click(
                fn=gradio_reset_all,
                inputs=[],
                outputs=[
                    audio_input,
                    text_input,
                    submit_button,
                    output_ta,
                    clear_button,
                ],
                queue=False,
            )


# --- TTS Demo ---


def generate_speech(text: str):
    file_name = f"out/{xxhash.xxh32(text).hexdigest()}.wav"
    output = model.synthesize_speech(text)
    audio = output["array"].astype(np.float32)
    sf.write(
        file_name,
        audio,
        output["sampling_rate"],
    )

    return file_name


default_text = ""
with gr.Blocks(theme=theme) as tts_demo:
    gr.HTML(
        """
    <div style='text-align: center'>
        <h1>
            Text-to-Speech Generation With Typhoon2 Audio
        </h1>
    </div>
    """
    )
    gr.Interface(
        fn=generate_speech,
        inputs=[
            gr.Textbox(
                value=default_text,
                label="Input text",
                placeholder="Type something here..",
            ),
        ],
        outputs=gr.Audio(label=""),
        flagging_mode="never",
    )

with gr.Blocks(
    theme=theme, title="🌪️ Typhoon2 Audio: Native Thai End-to-End Audio-Language Model"
) as demo:
    gr.Markdown(
        """<center><h1>🌪️ Typhoon2 Audio: Native Thai End-to-End Audio-Language Model</h1></center><br/>
        Typhoon2-Audio understands **Speech** as well as **Audio** events, and it responds in Thai and English texts
- Its capabilities include transcription, speech translation, speech instruction following, spoken-document Q&A, audio captioning, etc.
- The research preview model may not work well on long audio clips, particularly those exceeding 30 seconds. Text prompt can be empty for speech instruction following."""
    )
    gr.TabbedInterface(
        [omni_demo, processing_demo, tts_demo],
        ["Conversation Demo", "Audio Processing Demo", "Text-to-Speech Demo"],
        theme=theme,
    )
    gr.Markdown(
        """### Disclaimer
The responses generated by this Artificial Intelligence (AI) system are autonomously constructed and do not necessarily reflect the views or positions of the developing organizations, their affiliates, or any of their employees. These AI-generated responses do not represent those of the organizations. The organizations do not endorse, support, sanction, encourage, verify, or agree with the comments, opinions, or statements generated by this AI. The information produced by this AI is not intended to malign any religion, ethnic group, club, organization, company, individual, anyone, or anything. It is not the intent of the organizations to malign any group or individual. The AI operates based on its programming and training data and its responses should not be interpreted as the explicit intent or opinion of the organizations.

### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. Audio language models are prone to hallucinations to a greater extent compared to text-only LLMs. A known limitation is that the current version cannot handle audio longer than 30 seconds. The service may collect user dialogue data for future research.

### License
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses. The content of this project itself is licensed under the Apache license 2.0."""
    )

demo.queue()
demo.launch(ssr_mode=False)
