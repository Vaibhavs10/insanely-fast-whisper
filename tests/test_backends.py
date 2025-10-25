import importlib
import pathlib
import sys
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from insanely_fast_whisper import backends


def test_is_mlx_available_checks_both_packages():
    specs = {"mlx_whisper": object(), "mlx_parakeet": None}
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: specs.get(name)):  # type: ignore[arg-type]
        assert backends.is_mlx_available()


@pytest.mark.parametrize(
    "device_id,platform_name,mlx_available,expected",
    [
        ("0", "Linux", False, "transformers"),
        ("mps", "Darwin", True, "mlx"),
    ],
)
def test_select_backend_auto(device_id, platform_name, mlx_available, expected):
    assert (
        backends.select_backend(
            device_id=device_id,
            requested_backend="auto",
            platform_name=platform_name,
            mlx_available=mlx_available,
        )
        == expected
    )


def test_select_backend_auto_requires_mlx_on_mac():
    with pytest.raises(backends.BackendSelectionError):
        backends.select_backend(
            device_id="mps",
            requested_backend="auto",
            platform_name="Darwin",
            mlx_available=False,
        )


def test_manual_mlx_requires_mac():
    with pytest.raises(backends.BackendSelectionError):
        backends.select_backend(
            device_id="0",
            requested_backend="mlx",
            platform_name="Linux",
            mlx_available=True,
        )


def test_run_mlx_backend_word_timestamps():
    class FakeWhisper:
        @staticmethod
        def transcribe(audio_path, *, path_or_hf_repo, word_timestamps, verbose, **decode_options):
            assert audio_path == "sample.wav"
            assert path_or_hf_repo == "mlx-community/whisper-large-v3"
            assert word_timestamps is True
            assert decode_options == {"task": "transcribe", "language": "en"}
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

    with mock.patch.object(importlib, "import_module", return_value=FakeWhisper()):
        result = backends.run_mlx_backend(
            audio_path="sample.wav",
            model_name="openai/whisper-large-v3",
            task="transcribe",
            language="en",
            timestamp="word",
            mlx_model="whisper",
        )

    assert result["text"] == "hello world"
    assert result["chunks"] == [
        {"timestamp": (0.0, 0.5), "text": "hello"},
        {"timestamp": (0.5, 1.0), "text": "world"},
    ]


def test_run_mlx_backend_parakeet_requires_model_override():
    with mock.patch.object(importlib, "import_module", return_value=object()):
        with pytest.raises(backends.BackendSelectionError):
            backends.run_mlx_backend(
                audio_path="sample.wav",
                model_name="openai/whisper-large-v3",
                task="transcribe",
                language=None,
                timestamp="chunk",
                mlx_model="parakeet",
            )


def test_run_mlx_backend_parakeet_uses_chunks():
    class FakeParakeet:
        @staticmethod
        def transcribe(audio_path, *, path_or_hf_repo, word_timestamps, verbose, **decode_options):
            assert path_or_hf_repo == "mlx-community/parakeet-large"
            assert word_timestamps is False
            assert decode_options == {"task": "translate", "language": "fr"}
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

    with mock.patch.object(importlib, "import_module", return_value=FakeParakeet()):
        result = backends.run_mlx_backend(
            audio_path="sample.wav",
            model_name="mlx-community/parakeet-large",
            task="translate",
            language="fr",
            timestamp="chunk",
            mlx_model="parakeet",
        )

    assert result["chunks"] == [{"timestamp": (1.0, 2.5), "text": "segment text"}]
