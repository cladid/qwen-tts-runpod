"""VoiceDesign contract tests; model and CUDA are simulated."""
import base64
import importlib.util
import io
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import soundfile as sf


class DesignContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None, get_device_name=lambda _: 'mock GPU'),
            random=types.SimpleNamespace(fork_rng=Mock(side_effect=lambda **kw: nullcontext())), manual_seed=Mock(), inference_mode=nullcontext)
        spec = importlib.util.spec_from_file_location('design_under_test', Path(__file__).parent / 'handler.py')
        self.worker = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': self.torch, 'runpod': types.SimpleNamespace(), 'qwen_tts': types.SimpleNamespace(Qwen3TTSModel=None)}):
            spec.loader.exec_module(self.worker)
        def generate(**kwargs):
            self.calls.append(kwargs)
            return [np.sin(np.arange(24000)/30).astype(np.float32)*0.1], 24000
        self.worker.MODEL = types.SimpleNamespace(get_supported_languages=lambda: ['italian'], generate_voice_design=generate)
        self.worker.MODEL_LOAD_S = 0.1
        self.input = dict(text='Una frase. Una seconda frase.', language='Italian', instruct='A calm Italian male educator.', seed=42)

    def test_instructions_and_seed_reach_design_inference(self):
        result = self.worker.handler({'input': self.input})
        self.assertEqual(self.calls[0]['instruct'], self.input['instruct'])
        self.assertIs(self.calls[0]['non_streaming_mode'], True)
        self.torch.manual_seed.assert_called_once_with(42)
        self.torch.random.fork_rng.assert_called_once_with(devices=[0])
        self.assertEqual(result['instruct'], self.input['instruct'])
        self.assertEqual(result['model_id'], 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign')
        self.assertEqual(result['worker_version'], '0.4.0-voice-design')
        audio, sr = sf.read(io.BytesIO(base64.b64decode(result['wav_base64'])))
        self.assertEqual((len(audio), sr), (24000,24000))

    def test_invalid_or_clone_inputs_fail_before_inference(self):
        for patch_input in [dict(instruct=''), dict(instruct='x'*1001), dict(seed=True), dict(seed=-1),
                            dict(seed=2**32), dict(ref_audio_base64='not-supported'), dict(temperature=0.2)]:
            with self.subTest(patch_input=patch_input), self.assertRaises(ValueError):
                self.worker.handler({'input':dict(self.input,**patch_input)})
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
