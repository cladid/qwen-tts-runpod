"""Small, queue-based Qwen3-TTS benchmark worker."""
import base64
import io
import os
import threading
import time

import numpy as np
import runpod
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/qwen3tts")
MAX_TEXT_CHARS = 600
LOCK = threading.Lock()
MODEL = None
MODEL_LOAD_S = None
CALLS = 0


def validate_input(job):
    inp = job.get("input")
    if not isinstance(inp, dict):
        raise ValueError("input must be an object")
    text = inp.get("text")
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_TEXT_CHARS:
        raise ValueError("text must contain between 1 and 600 characters")
    voice = inp.get("voice", "Ryan")
    language = inp.get("language", "Italian")
    instruct = inp.get("instruct", "")
    if not isinstance(voice, str) or not isinstance(language, str):
        raise ValueError("voice and language must be strings")
    if not isinstance(instruct, str) or len(instruct) > 300:
        raise ValueError("instruct must be a string of at most 300 characters")
    return text.strip(), voice, language, instruct


def load_model():
    global MODEL, MODEL_LOAD_S
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    started = time.perf_counter()
    MODEL = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    torch.cuda.synchronize()
    MODEL_LOAD_S = time.perf_counter() - started


def handler(job):
    global CALLS
    text, voice, language, instruct = validate_input(job)
    with LOCK:
        if MODEL is None:
            raise RuntimeError("model not initialized")
        if voice.lower() not in MODEL.get_supported_speakers():
            raise ValueError("unsupported voice")
        if language.lower() not in MODEL.get_supported_languages():
            raise ValueError("unsupported language; use Italian for Italian audio")
        started = time.perf_counter()
        torch.cuda.synchronize()
        synthesis_started = time.perf_counter()
        with torch.inference_mode():
            wavs, sr = MODEL.generate_custom_voice(
                text=text, speaker=voice, language=language,
                instruct=instruct, max_new_tokens=2048,
            )
        torch.cuda.synchronize()
        inference_s = time.perf_counter() - synthesis_started
        wav = np.asarray(wavs[0], dtype=np.float32)
        if wav.ndim != 1 or not len(wav) or not np.isfinite(wav).all() or sr <= 0:
            raise RuntimeError("invalid audio output")
        duration_s = len(wav) / sr
        # A cap hit can silently truncate speech: fail rather than score it as complete.
        if duration_s >= 169:
            raise RuntimeError("audio approaches generation limit; split the text")
        with io.BytesIO() as out:
            sf.write(out, wav, sr, format="WAV", subtype="PCM_16")
            audio_b64 = base64.b64encode(out.getvalue()).decode("ascii")
        CALLS += 1
        return {
            "model_id": MODEL_ID, "voice": voice, "language": language,
            "sample_rate": int(sr), "duration_s": duration_s,
            "inference_s": inference_s,
            "handler_s": time.perf_counter() - started,
            "model_load_s": MODEL_LOAD_S,
            "first_request_on_worker": CALLS == 1,
            "request_number_on_worker": CALLS,
            "rtf": inference_s / duration_s,
            "characters": len(text), "gpu": torch.cuda.get_device_name(0),
            "attention": "sdpa", "wav_base64": audio_b64,
        }


if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
