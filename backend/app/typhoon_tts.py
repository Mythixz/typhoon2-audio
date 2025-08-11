import os
import threading
from typing import Optional

import numpy as np

# Lazy import to avoid heavy deps unless enabled
_TTS_MODEL = None
_TTS_LOCK = threading.Lock()
_ENABLED = os.getenv("USE_TYPHOON_TTS", "false").lower() in {"1", "true", "yes"}
_MODEL_NAME = os.getenv("TYPHOON_MODEL", "scb10x/llama3.1-typhoon2-audio-8b-instruct")
_DEVICE = os.getenv("TYPHOON_DEVICE", "cuda")  # cuda | cpu | mps


def is_enabled() -> bool:
    return _ENABLED


def _load_model_if_needed():
    global _TTS_MODEL
    if _TTS_MODEL is not None:
        return
    if not is_enabled():
        return
    with _TTS_LOCK:
        if _TTS_MODEL is not None:
            return
        try:
            import torch
            from transformers import AutoModel
            dtype = torch.float16 if _DEVICE == "cuda" else torch.float32
            _TTS_MODEL = AutoModel.from_pretrained(
                _MODEL_NAME,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            _TTS_MODEL.to(_DEVICE)
        except Exception as e:
            # Disable on failure
            print(f"[typhoon_tts] Failed to load model: {e}")
            _TTS_MODEL = None


def synthesize(text: str) -> Optional[tuple[np.ndarray, int]]:
    """Return (waveform[float32], sampling_rate) or None if disabled/failed."""
    if not is_enabled():
        return None
    _load_model_if_needed()
    if _TTS_MODEL is None:
        return None
    try:
        y = _TTS_MODEL.synthesize_speech(text)
        # y => numpy array (shape (1, T) or (T,)), sampling rate DEFAULT 16000 per repo
        arr = y["array"] if isinstance(y, dict) else y
        sr = y.get("sampling_rate", 16000) if isinstance(y, dict) else 16000
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        # Expect shape (1, T) float32
        return arr.astype(np.float32), int(sr)
    except Exception as e:
        print(f"[typhoon_tts] synthesize failed: {e}")
        return None 