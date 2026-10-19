FROM ghcr.io/cladid/qwen-tts-runpod@sha256:7e0f7bd87b10e4c5a5a93438a57e1c8f7b645c3e44d83b35c686aad1d9994aaf
WORKDIR /app
COPY handler.py .
RUN python -m py_compile handler.py
RUN python -c "import inspect; from importlib.metadata import version; from qwen_tts import Qwen3TTSModel; assert version('qwen-tts') == '0.1.1'; assert 'non_streaming_mode' in inspect.signature(Qwen3TTSModel.generate_voice_clone).parameters"
CMD ["python", "-u", "handler.py"]
