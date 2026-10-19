FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 sox && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && python -m pip install -r requirements.txt && python -m pip check
ARG MODEL_REVISION=main
RUN MODEL_REVISION=${MODEL_REVISION} python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', revision=os.environ['MODEL_REVISION'], local_dir='/models/qwen3tts')"
ENV MODEL_PATH=/models/qwen3tts HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY handler.py .
RUN python -m py_compile handler.py
RUN python -c "import inspect; from importlib.metadata import version; from qwen_tts import Qwen3TTSModel; from pathlib import Path; assert version('qwen-tts') == '0.1.1'; assert 'instruct' in inspect.signature(Qwen3TTSModel.generate_voice_design).parameters; assert Path('/models/qwen3tts/speech_tokenizer/config.json').is_file()"
CMD ["python", "-u", "handler.py"]
