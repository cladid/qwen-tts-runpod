"""Worker contract tests without downloading models or allocating a GPU."""
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


class CloneContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: None, get_device_name=lambda _: 'mock GPU'),
            inference_mode=nullcontext, random=types.SimpleNamespace(fork_rng=Mock(side_effect=lambda **kw: nullcontext())),
            manual_seed=Mock())
        self.torch = torch
        spec = importlib.util.spec_from_file_location('worker_under_test', Path(__file__).parent / 'handler.py')
        self.worker = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': torch, 'runpod': types.SimpleNamespace(),
                                     'qwen_tts': types.SimpleNamespace(Qwen3TTSModel=None)}):
            spec.loader.exec_module(self.worker)
        def generate(**kwargs):
            self.calls.append(kwargs)
            return [np.sin(np.arange(24000) / 30).astype(np.float32) * 0.1], 24000
        self.worker.MODEL = types.SimpleNamespace(get_supported_languages=lambda: ['italian'], generate_voice_clone=generate)
        self.worker.MODEL_LOAD_S = 0.1
        f = io.BytesIO()
        sf.write(f, np.sin(np.arange(3 * 24000) / 30) * 0.1, 24000, format='WAV', subtype='PCM_16')
        self.input = dict(text='Una frase. Una seconda frase.', language='Italian', ref_text='Voce di riferimento.',
                          ref_audio_base64=base64.b64encode(f.getvalue()).decode())

    def test_existing_contract_keeps_default_mode_and_no_seed(self):
        result = self.worker.handler({'input': self.input})
        self.assertIs(self.calls[0]['non_streaming_mode'], False)
        self.assertIs(result['non_streaming_mode'], False)
        self.assertIsNone(result['seed'])
        self.torch.manual_seed.assert_not_called()
        self.torch.random.fork_rng.assert_not_called()

    def test_mode_and_seed_reach_inference_and_metadata(self):
        for mode in [False, True]:
            result = self.worker.handler({'input': dict(self.input, non_streaming_mode=mode, seed=42)})
            self.assertIs(self.calls[-1]['non_streaming_mode'], mode)
            self.assertEqual(self.calls[-1]['ref_audio'][1], 24000)
            self.assertEqual(self.calls[-1]['ref_text'], self.input['ref_text'])
            self.assertIs(self.calls[-1]['x_vector_only_mode'], False)
            self.assertIs(result['non_streaming_mode'], mode)
            self.assertEqual(result['seed'], 42)
            audio, sr = sf.read(io.BytesIO(base64.b64decode(result['wav_base64'])))
            self.assertEqual((len(audio), sr), (24000, 24000))
        self.torch.manual_seed.assert_called_with(42)
        self.torch.random.fork_rng.assert_called_with(devices=[0])

    def test_invalid_controls_are_rejected_before_inference(self):
        for values in [dict(non_streaming_mode='false'), dict(non_streaming_mode=1),
                       dict(seed=True), dict(seed=-1), dict(seed=2**32), dict(seed=1.5),
                       dict(temperature=0.1), dict(ref_audio_base64='!!!')]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.worker.handler({'input': dict(self.input, **values)})
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
