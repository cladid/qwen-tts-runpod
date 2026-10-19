# Qwen VoiceDesign experiment

Separate worker for Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign, version 0.4.0.
Input: text (1–600 chars), language (Italian), instruct (1–1000 chars), optional
integer seed (0..2**32-1). Uses generate_voice_design, non_streaming_mode=True,
bfloat16/SDPA, max_new_tokens=2048. No reference audio, no voice cloning.
PyTorch RNG is isolated per seeded request. Output includes prompt, seed,
model ID, worker version, GPU/timing metadata and PCM16 WAV base64.

Build context: design/. Workflow publishes a distinct 0.4.0-voice-design tag
and design-sha commit tag. Existing CustomVoice and Base workers/images remain.
Weights downloaded during build, runtime Hugging Face offline.
Contract tests simulate model/CUDA and verify instructions reach inference.
This is an experiment; prompt/seed alone do not guarantee identical timbre
across independent generations or consistent intonation.
