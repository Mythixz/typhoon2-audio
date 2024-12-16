from typhoon2_audio.modeling_typhoon2audio import (
    DEFAULT_MODEL_SAMPLING_RATE,
    TensorStreamer,
    Typhoon2Audio2AudioForConditionalGeneration,
)

from dataclasses import dataclass, field
from transformers import TextIteratorStreamer
import shutil
import os
from typing import Tuple
import gradio as gr
import numpy as np
from gradio_webrtc import (
    ReplyOnPause,
    WebRTC,
    AdditionalOutputs,
    get_twilio_turn_credentials,
)
import torch
import xxhash
import soundfile as sf
from dotenv import load_dotenv
from datasets import Audio

load_dotenv()

# Twilio TURN credentials configuration
rtc_configuration = get_twilio_turn_credentials()
print(rtc_configuration)
OUT_CHANNELS = 1

OUT_PATH = "out"
LOADER_STR = "♫"
yield_index = 0

if os.path.exists(OUT_PATH):
    shutil.rmtree(OUT_PATH)
    os.makedirs(OUT_PATH)

model = Typhoon2Audio2AudioForConditionalGeneration.from_pretrained(
    "scb10x/llama3.1-typhoon2-audio-8b-instruct-241213",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.to("cuda")


resampler = Audio(sampling_rate=16000)


@torch.no_grad()
def inference(conversations, temperature=0.4, system_prompt=None):
    updated_conversations = []

    if system_prompt and system_prompt.strip():
        updated_conversations.append({"role": "system", "content": system_prompt})

    for conv in conversations:
        if isinstance(conv["content"], str):
            updated_conversations.append(
                {"role": conv["role"], "content": conv["content"]}
            )
        else:
            if isinstance(conv["content"], dict):
                audio_path = conv["content"]["path"]
            else:
                audio_path = conv["content"].file.path
            updated_conversations.append(
                {
                    "role": conv["role"],
                    "content": [{"type": "audio", "audio_url": audio_path}],
                }
            )

    streamer = TextIteratorStreamer(
        model.llama_tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=15,
    )
    streamer_unit = TensorStreamer(timeout=15)
    generator = model.generate_stream(
        conversation=conversations,
        temperature=temperature,
        top_p=0.95,
        max_new_tokens=512,
        streaming_unit_gen=True,
        streamer=streamer,
        streamer_unit=streamer_unit,
    )
    for wav_np, text in generator:
        yield wav_np, text


@dataclass
class AppState:
    conversation: list = field(default_factory=list)
    system_prompt: str = "You are always answer in Thai."
    stopped: bool = False


def response(audio_tuple: Tuple[int, np.ndarray], state: AppState, system_prompt: str):
    global yield_index

    if system_prompt != state.system_prompt:
        state.system_prompt = system_prompt

    if not audio_tuple:
        return AppState(system_prompt=system_prompt)

    sr, audio = audio_tuple
    file_name = f"out/{xxhash.xxh32(bytes(audio)).hexdigest()}.wav"
    assert len(audio.shape) == 2
    sf.write(file_name, audio[0], sr, format="wav")

    state.conversation.append(
        {"role": "user", "content": {"path": file_name, "mime_type": "audio/wav"}}
    )

    state.conversation.append(
        {
            "role": "assistant",
            "content": LOADER_STR,
        }
    )

    decode_sample_rate = DEFAULT_MODEL_SAMPLING_RATE
    buff = None
    for wav, text in inference(state.conversation, system_prompt=state.system_prompt):
        if state.conversation[-1]["content"] == LOADER_STR:
            state.conversation[-1]["content"] = text
        else:
            state.conversation[-1]["content"] = state.conversation[-1]["content"] + text
        if wav is not None:
            if buff is None:
                buff = wav.cpu().numpy()
            else:
                buff = np.concatenate((buff, wav.cpu().numpy()))
        if buff.shape[0] > (decode_sample_rate * 2):
            yield_index += 1
            yield (
                decode_sample_rate,
                buff.reshape(1, -1),
                "mono",
            ), AdditionalOutputs(dict(state=state, chatbot=state.conversation))
            buff = None

    if buff is not None and buff.shape[0] > 0:
        yield (decode_sample_rate, buff.reshape(1, -1), "mono"), AdditionalOutputs(
            dict(
                state=AppState(
                    conversation=state.conversation, system_prompt=state.system_prompt
                ),
                chatbot=state.conversation,
            )
        )


with gr.Blocks() as omni_demo:
    gr.HTML(
        """
    <div style='text-align: center'>
        <h1>
            Talk To Typhoon 2 Audio (Powered by WebRTC ⚡️)
        </h1>
        <p>
            Each conversation is limited to 60 seconds. Once the time limit is up you can rejoin the conversation.
        </p>
    </div>
    """
    )
    with gr.Column():
        with gr.Row():
            system_prompt = gr.Textbox(
                label="System Prompt",
                value="You are always answer in Thai.",
                info="Customize the system prompt to guide the AI's behavior",
            )
        with gr.Group():
            webrtc = WebRTC(
                label="Stream",
                rtc_configuration=rtc_configuration,
                mode="send-receive",
                modality="audio",
            )
        with gr.Row():
            chatbot = gr.Chatbot(label="Conversation", type="messages")
        state = gr.State(value=AppState())
        webrtc.stream(
            fn=ReplyOnPause(
                response,
                output_sample_rate=DEFAULT_MODEL_SAMPLING_RATE,
                input_sample_rate=DEFAULT_MODEL_SAMPLING_RATE,
                output_frame_size=480,
            ),
            inputs=[webrtc, state, system_prompt],
            outputs=[webrtc],
            time_limit=60,
        )

        def update_state(response):
            state = response["state"]
            chatbot = response["chatbot"]
            return state, chatbot

        webrtc.on_additional_outputs(
            update_state,
            inputs=[],
            outputs=[state, chatbot],
            queue=False,
            show_progress="hidden",
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
    print(typhoon_output["text"])
    yield (
        gr.Button(value="Click to run inference", interactive=True, variant="primary"),
        gr.Textbox(
            value=typhoon_output["text"],
            visible=True,
        ),
        gr.update(visible=True),
    )


text_input = ""
with gr.Blocks() as processing_demo:
    gr.HTML(
        """
    <div style='text-align: center'>
        <h1>
            Audio Processing With Typhoon 2 Audio
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
with gr.Blocks() as tts_demo:
    gr.HTML(
        """
    <div style='text-align: center'>
        <h1>
            Text-to-Speech Generation With Typhoon 2 Audio
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

demo = gr.TabbedInterface(
    [omni_demo, processing_demo, tts_demo],
    ["Live Conversation Demo", "Audio Processing Demo", "Text-to-Speech Demo"],
    title="Typhoon 2 Audio - Demo",
)

demo.queue()
demo.launch(ssr_mode=False)
