from dataclasses import dataclass, field
from typing import Tuple
from typhoon2_audio.modeling_typhoon2audio import (
    DEFAULT_MODEL_SAMPLING_RATE,
    Typhoon2Audio2AudioForConditionalGeneration,
    TensorStreamer,
)
import copy
from transformers import TextIteratorStreamer
import os
import gradio as gr
from gradio import ChatMessage
import numpy as np
import torch
import xxhash
import soundfile as sf
from dotenv import load_dotenv
from datasets import Audio
from pathlib import Path
from gradio_webrtc import (
    AlgoOptions,
    ReplyOnPause,
    WebRTC,
    AdditionalOutputs,
    get_twilio_turn_credentials,
)

load_dotenv()

OUT_PATH = Path("tmp/out")
OUT_PATH.mkdir(exist_ok=True, parents=True)

model = Typhoon2Audio2AudioForConditionalGeneration.from_pretrained(
    "scb10x/llama3.1-typhoon2-audio-8b-instruct",
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


DEFAULT_SYSTEM_PROMPT = "You are a helpful female assistant named ไต้ฝุ่น. Respond conversationally to the speech provided in the language it is spoken in."
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
    wav_path = OUT_PATH / f"{array_hash}.wav"
    os.makedirs("tmp", exist_ok=True)
    if not os.path.exists(wav_path):
        sf.write(wav_path, a["array"], a["sampling_rate"], format="wav")

    # adding system prompt seems to be more effective than as system prompt
    # this is a quick hack, but more investigate should be conducted
    if len(messages) == 0:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "audio", "audio_url": wav_path},
                ],
            }
        )
    else:
        messages.append(
            {"role": "user", "content": [{"type": "audio", "audio_url": wav_path}]}
        )

    # Typhoon-Audio model
    response = model.generate(conversation=messages)

    text = response["text"]
    audio = response["audio"]
    temp_out_path = OUT_PATH / f"out-{array_hash}.wav"
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
    global messages
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
                    label="Text Prompt",
                    placeholder="You can control the model's behavior by specifying the prompt here.",
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

DEFAULT_SYSTEM_PROMPT_PROCESSING = "Listen to the audio and answer the question"


def is_able_to_start(audio_input):
    if audio_input is not None:
        return gr.Button(
            value="Click to run inference", interactive=True, variant="primary"
        )
    return gr.Button(value="Please upload audio", interactive=False, variant="primary")


def gradio_reset_all():
    sys_prompt = DEFAULT_SYSTEM_PROMPT_PROCESSING
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
    wav_path = OUT_PATH / f"{array_hash}.wav"

    if not os.path.exists(wav_path):
        sf.write(wav_path, a["array"], a["sampling_rate"], format="wav")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "audio", "audio_url": wav_path},
            ],
        }
    ]

    typhoon_output = model.generate(conversation=conversation)
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
            with gr.Column():
                text_input = gr.Textbox(
                    value=DEFAULT_SYSTEM_PROMPT_PROCESSING,
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
    file_name = OUT_PATH / f"{xxhash.xxh32(text).hexdigest()}.wav"
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
        flagging_mode="never",  # some version of gradio will result in an error -- comment this out
    )


### WebRTC demo


@torch.no_grad()
def inference_rtc(conversations, temperature=0.4, system_prompt=None):
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

    if updated_conversations[-1]["role"] == "assistant":
        updated_conversations = updated_conversations[:-1]

    streamer = TextIteratorStreamer(
        model.llama_tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=15,
    )
    streamer_unit = TensorStreamer(timeout=15)
    generator = model.generate_stream(
        conversation=updated_conversations,
        temperature=temperature,
        top_p=0.95,
        max_new_tokens=512,
        streaming_unit_gen=True,
        streamer=streamer,
        streamer_unit=streamer_unit,
    )
    for wav_np, text in generator:
        yield wav_np, text


rtc_configuration = get_twilio_turn_credentials()


@dataclass
class RTCAppState:
    conversation: list = field(default_factory=list)
    system_prompt: str = "You are always answer in Thai."
    stopped: bool = False


LOADER_STR = "♫"
IS_RUNNING_RTC = False
AUDIO_CHUCK_DURATION = 0.6


def response_rtc(
    audio_tuple: Tuple[int, np.ndarray], state: RTCAppState, system_prompt: str
):
    global IS_RUNNING_RTC
    if IS_RUNNING_RTC or not audio_tuple:
        print("skip prediction; currently in responding")
        return state

    IS_RUNNING_RTC = True
    if system_prompt != state.system_prompt:
        state.system_prompt = system_prompt

    sr, audio = audio_tuple
    print("audio length ", audio.shape[-1] / DEFAULT_MODEL_SAMPLING_RATE)
    if audio.shape[-1] / DEFAULT_MODEL_SAMPLING_RATE < AUDIO_CHUCK_DURATION * 3:
        print("too short, maybe misfired")
        return state

    file_name = (
        (OUT_PATH / f"rt_{xxhash.xxh32(bytes(audio)).hexdigest()}.wav")
        .absolute()
        .as_posix()
    )
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
    for wav, text in inference_rtc(
        copy.deepcopy(state.conversation), system_prompt=state.system_prompt
    ):
        assert state.conversation[-1]["role"] == "assistant"
        if state.conversation[-1]["content"] == LOADER_STR:
            state.conversation[-1]["content"] = text
        else:
            state.conversation[-1]["content"] = state.conversation[-1]["content"] + text

        # wav might be none on last step
        if wav is None:
            continue

        if buff is None:
            buff = wav.cpu().numpy()
        else:
            buff = np.concatenate((buff, wav.cpu().numpy()))
        if buff.shape[0] > (decode_sample_rate * 2):
            yield (
                decode_sample_rate,
                buff.reshape(1, -1),
                "mono",
            ), AdditionalOutputs(dict(state=state, chatbot=state.conversation))
            buff = None

    IS_RUNNING_RTC = False
    if buff is not None and buff.shape[0] > 0:
        yield (decode_sample_rate, buff.reshape(1, -1), "mono"), AdditionalOutputs(
            dict(
                state=RTCAppState(
                    conversation=state.conversation, system_prompt=state.system_prompt
                ),
                chatbot=state.conversation,
            )
        )


with gr.Blocks(theme=theme) as rtc_demo:
    gr.HTML(
        """
    <div style='text-align: center'>
        <h1>
            Talk To Typhoon 2 Audio (Powered by WebRTC ⚡️)
        </h1>
        <p>
            Each conversation is limited to 60 seconds. Once the time limit is up you can rejoin the conversation.
        </p>
        <p>
            Please use earphone, due to voice activity detection is fired on output speech on speaker.
        </p>
    </div>
    """
    )
    with gr.Column():
        with gr.Row():
            system_prompt = gr.Textbox(
                label="System Prompt",
                value=DEFAULT_SYSTEM_PROMPT,
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
            chatbot_component = gr.Chatbot(label="Conversation", type="messages")
        state = gr.State(value=RTCAppState())
        vad_options = AlgoOptions(
            audio_chunk_duration=AUDIO_CHUCK_DURATION,
            started_talking_threshold=0.2,
            speech_threshold=0.1,
        )
        webrtc.stream(
            fn=ReplyOnPause(
                response_rtc,
                output_sample_rate=DEFAULT_MODEL_SAMPLING_RATE,
                input_sample_rate=DEFAULT_MODEL_SAMPLING_RATE,
                output_frame_size=480,
                algo_options=vad_options,
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
            outputs=[state, chatbot_component],
            queue=False,
            show_progress="hidden",
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
        [omni_demo, processing_demo, tts_demo, rtc_demo],
        [
            "Conversation Demo",
            "Audio Processing Demo",
            "Text-to-Speech Demo",
            "Real-Time Conversation Demo (WebRTC)",
        ],
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
# demo.launch(share=True)
