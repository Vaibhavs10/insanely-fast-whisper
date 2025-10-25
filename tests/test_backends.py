import pathlib
import sys
import unittest
from contextlib import nullcontext
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from insanely_fast_whisper import backends


class AvailabilityTests(unittest.TestCase):
    def test_is_mlx_available_when_whisper_installed(self):
        with mock.patch("importlib.util.find_spec", side_effect=lambda name: object() if name == "mlx_whisper" else None):
            self.assertTrue(backends.is_mlx_available())

    def test_is_mlx_available_when_parakeet_installed(self):
        with mock.patch("importlib.util.find_spec", side_effect=lambda name: object() if name == "mlx_parakeet" else None):
            self.assertTrue(backends.is_mlx_available())

    def test_is_mlx_unavailable_when_no_packages_found(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(backends.is_mlx_available())


class SelectBackendTests(unittest.TestCase):
    def test_auto_prefers_transformers_on_cuda(self):
        choice = backends.select_backend(
            device_id="0", requested_backend="auto", platform_name="Linux", mlx_available=False
        )
        self.assertEqual(choice, "transformers")

    def test_auto_requires_mlx_packages_on_mac(self):
        with self.assertRaises(backends.BackendSelectionError):
            backends.select_backend(
                device_id="mps",
                requested_backend="auto",
                platform_name="Darwin",
                mlx_available=False,
            )

    def test_manual_mlx_requires_mac(self):
        with self.assertRaises(backends.BackendSelectionError):
            backends.select_backend(
                device_id="0", requested_backend="mlx", platform_name="Linux", mlx_available=True
            )

    def test_manual_mlx_allowed_on_mac(self):
        choice = backends.select_backend(
            device_id="mps", requested_backend="mlx", platform_name="Darwin", mlx_available=True
        )
        self.assertEqual(choice, "mlx")


class RunMlxBackendTests(unittest.TestCase):
    def setUp(self):
        self.progress_patch = mock.patch.object(
            backends, "_progress", new=lambda *args, **kwargs: nullcontext()
        )
        self.progress_patch.start()

    def tearDown(self):
        self.progress_patch.stop()

    def test_whisper_word_timestamps(self):
        calls = []

        class FakeWhisper:
            @staticmethod
            def transcribe(audio_path, *, path_or_hf_repo, word_timestamps, verbose, **decode_options):
                calls.append(
                    {
                        "audio_path": audio_path,
                        "repo": path_or_hf_repo,
                        "word_timestamps": word_timestamps,
                        "decode_options": decode_options,
                    }
                )
                return {
                    "text": "hello world",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "hello world",
                            "words": [
                                {"start": 0.0, "end": 0.5, "word": "hello"},
                                {"start": 0.5, "end": 1.0, "word": "world"},
                            ],
                        }
                    ],
                }

        with mock.patch.object(backends, "_import_mlx_module", return_value=FakeWhisper()):
            result = backends.run_mlx_backend(
                audio_path="sample.wav",
                model_name="openai/whisper-large-v3",
                task="transcribe",
                language="en",
                timestamp="word",
                batch_size=24,
                mlx_model="whisper",
            )

        self.assertEqual(result["text"], "hello world")
        self.assertEqual(
            result["chunks"],
            [
                {"timestamp": (0.0, 0.5), "text": "hello"},
                {"timestamp": (0.5, 1.0), "text": "world"},
            ],
        )
        self.assertEqual(calls[0]["repo"], "mlx-community/whisper-large-v3")
        self.assertTrue(calls[0]["word_timestamps"])
        self.assertEqual(calls[0]["decode_options"], {"task": "transcribe", "language": "en"})

    def test_parakeet_requires_model_override(self):
        with mock.patch.object(backends, "_import_mlx_module", return_value=object()):
            with self.assertRaises(backends.BackendSelectionError):
                backends.run_mlx_backend(
                    audio_path="sample.wav",
                    model_name="openai/whisper-large-v3",
                    task="transcribe",
                    language=None,
                    timestamp="chunk",
                    batch_size=24,
                    mlx_model="parakeet",
                )

    def test_parakeet_chunk_timestamps(self):
        calls = []

        class FakeParakeet:
            @staticmethod
            def transcribe(audio_path, *, path_or_hf_repo, word_timestamps, verbose, **decode_options):
                calls.append(
                    {
                        "audio_path": audio_path,
                        "repo": path_or_hf_repo,
                        "word_timestamps": word_timestamps,
                        "decode_options": decode_options,
                    }
                )
                return {
                    "text": "segment text",
                    "segments": [
                        {
                            "start": 1.0,
                            "end": 2.5,
                            "text": "segment text",
                        }
                    ],
                }

        with mock.patch.object(backends, "_import_mlx_module", return_value=FakeParakeet()):
            result = backends.run_mlx_backend(
                audio_path="sample.wav",
                model_name="mlx-community/parakeet-large",
                task="translate",
                language="fr",
                timestamp="chunk",
                batch_size=24,
                mlx_model="parakeet",
            )

        self.assertEqual(result["text"], "segment text")
        self.assertEqual(result["chunks"], [{"timestamp": (1.0, 2.5), "text": "segment text"}])
        self.assertEqual(calls[0]["repo"], "mlx-community/parakeet-large")
        self.assertFalse(calls[0]["word_timestamps"])
        self.assertEqual(calls[0]["decode_options"], {"task": "translate", "language": "fr"})


if __name__ == "__main__":
    unittest.main()
