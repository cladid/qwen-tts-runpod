"""VoiceDesign experiment with explicit voice and delivery instructions."""
import base64
from contextlib import nullcontext
import io
import os
import threading
import time

import numpy as np
import runpod
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
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
    language = inp.get("language", "Italian")
    instruct = inp.get("instruct")
    seed = inp.get("seed")
    if set(inp) - {"text", "language", "instruct", "seed"}:
        raise ValueError("unknown input fields")
    if not isinstance(language, str):
        raise ValueError("language must be a string")
    if not isinstance(instruct, str) or not 1 <= len(instruct.strip()) <= 1000:
        raise ValueError("instruct must contain between 1 and 1000 characters")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32 - 1):
        raise ValueError("seed must be an integer between 0 and 2**32-1")
    return text.strip(), language, instruct.strip(), seed



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
    text, language, instruct, seed = validate_input(job)
    with LOCK:
        if MODEL is None:
            raise RuntimeError("model not initialized")
        if language.lower() not in MODEL.get_supported_languages():
            raise ValueError("unsupported language; use Italian for Italian audio")
        started = time.perf_counter()
        torch.cuda.synchronize()
        synthesis_started = time.perf_counter()
        rng_context = torch.random.fork_rng(devices=[0]) if seed is not None else nullcontext()
        with rng_context, torch.inference_mode():
            if seed is not None:
                torch.manual_seed(seed)
            wavs, sr = MODEL.generate_voice_design(
                text=text, language=language, instruct=instruct,
                max_new_tokens=2048, non_streaming_mode=True,
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
            "model_id": MODEL_ID, "instruct": instruct, "language": language,
            "sample_rate": int(sr), "duration_s": duration_s,
            "worker_version": "0.4.0-voice-design",
            "non_streaming_mode": True, "seed": seed,
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
