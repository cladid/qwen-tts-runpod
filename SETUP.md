# Qwen TTS on RunPod Serverless

## Publish

Open Actions > Build and publish Qwen TTS > Run workflow (main).
The workflow builds on GitHub, downloads model weights including the speech
tokenizer, checks imports, and publishes to GHCR. No RunPod key is required.
After success, open your GitHub profile > Packages > qwen-tts-runpod > Package
settings > Change visibility > Public. Public repository visibility alone does
not guarantee public package visibility. Use the sha256 digest printed by the
build for an immutable deployment, or the version tag:

    ghcr.io/cladid/qwen-tts-runpod:0.1.0

The first build downloads several GB. GitHub Actions/storage allowances depend
on your account; RunPod credit does not pay for GitHub usage. No GPU runs in CI.
The model revision defaults to main and is frozen inside the built image;
MODEL_REVISION can be pinned to a Hugging Face commit for subsequent builds.

## RunPod

Create a Docker queue-based Serverless endpoint using the public image above.
Use one RTX 4090 24 GB GPU, active/min workers 0, max workers 1, idle timeout
5 seconds, execution timeout 600 seconds, FlashBoot enabled. Verify the actual
Serverless Flex rate in the console; Pod prices do not apply. The worker uses
PyTorch SDPA attention with bfloat16; FlashAttention 2 is not installed. Treat
results as a baseline for this implementation, not maximum Qwen performance.

First test (Console > Requests; include the input wrapper):

```json
{
  "input": {
    "text": "La sicurezza sul lavoro richiede attenzione, formazione e responsabilità. Prima di iniziare un'attività, verifica i rischi e scegli le misure di prevenzione più adatte.",
    "language": "Italian",
    "voice": "Ryan",
    "instruct": "Speak clearly, with a calm and natural educational tone."
  }
}
```

Voice names are predefined speakers, not cloned voices. Ryan is a native
English speaker; Italian pronunciation must be evaluated by listening.
Input is limited to 600 characters and instructions to 300. Test longer lessons
later as multiple chunks, checking voice consistency across joins.

## Measure

The response contains PCM16 WAV as wav_base64, real sample_rate, duration_s,
inference_s (synthesis and decoding), model_load_s, handler_s, and rtf.
RTF = inference seconds / audio seconds; lower is faster, <1 faster than playback.
first_request_on_worker identifies a worker's first request, not a guaranteed
cold start (FlashBoot may retain state). model_load_s is reported on every
request but is incurred once per worker load; do not sum it across requests.
No text or base64 audio is explicitly logged by our handler; RunPod job data
and provider-side retention still apply.

Record RunPod job delayTime, executionTime and client wall time separately.
Queue delay is not itself a measure of billable GPU time. Billing includes
startup, execution and idle worker time; compare account usage before/after
a bounded sequential test batch. Cost per audio minute is total test cost
divided by total generated audio minutes. Do not infer actual cost solely
from inference_s. Keep the first test short, then compare repeated requests.
600-second job timeout and max workers 1 are not a dollar spending cap.
After testing set max workers to 0 and confirm workers have stopped.

Build/import checks do not verify GPU inference. Validate the first deployment
logs, one short Italian sample, and its complete spoken content before a batch.
