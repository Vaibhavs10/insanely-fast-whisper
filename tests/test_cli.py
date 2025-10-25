import json
import sys
from pathlib import Path
from typing import List

import pytest

from insanely_fast_whisper import cli

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO = PROJECT_ROOT / "test.wav"


@pytest.fixture(scope="session")
def audio_file() -> Path:
    """Return the vendored WAV sample."""
    header = TEST_AUDIO.read_bytes()[:4]
    if header != b"RIFF":
        raise RuntimeError("Vendored test.wav is not a PCM WAV file.")
    return TEST_AUDIO


def _run_cli(monkeypatch, tmp_path, audio_path: Path, extra_args: List[str]) -> dict:
    transcript_path = tmp_path / "transcript.json"
    argv = [
        "insanely-fast-whisper",
        "--file-name",
        str(audio_path),
        "--transcript-path",
        str(transcript_path),
    ]
    argv.extend(extra_args)
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    payload = json.loads(transcript_path.read_text())
    assert "text" in payload
    assert "chunks" in payload and payload["chunks"]
    return payload


def test_transformers_backend_smoke(audio_file, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    try:
        result = _run_cli(
            monkeypatch,
            tmp_path,
            audio_file,
            [
                "--backend",
                "transformers",
                "--model-name",
                "openai/whisper-tiny",
                "--device-id",
                "cpu",
                "--batch-size",
                "1",
            ],
        )
    except RuntimeError as exc:
        if "torchcodec" in str(exc):
            pytest.skip("torchcodec runtime dependencies are unavailable in this environment.")
        raise

    assert result["text"].strip()


def test_mlx_whisper_backend_word(audio_file, monkeypatch, tmp_path):
    pytest.importorskip("mlx_whisper")

    result = _run_cli(
        monkeypatch,
        tmp_path,
        audio_file,
        [
            "--backend",
            "mlx",
            "--mlx-family",
            "whisper",
            "--model-name",
            "mlx-community/whisper-tiny",
            "--timestamp",
            "word",
        ],
    )

    first_chunk = result["chunks"][0]
    assert isinstance(first_chunk["timestamp"], (list, tuple))


def test_mlx_parakeet_backend_chunk(audio_file, monkeypatch, tmp_path):
    pytest.importorskip("parakeet_mlx")

    result = _run_cli(
        monkeypatch,
        tmp_path,
        audio_file,
        [
            "--backend",
            "mlx",
            "--mlx-family",
            "parakeet",
            "--model-name",
            "mlx-community/parakeet-tdt_ctc-110m",
            "--timestamp",
            "chunk",
        ],
    )

    assert any(chunk["timestamp"][0] is not None for chunk in result["chunks"])
