import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from insanely_fast_whisper.cli import (
    _build_multipart_body,
    _normalize_transcription_response,
    transcribe_openai_compatible,
)


def test_normalize_verbose_json_segments():
    result = _normalize_transcription_response(
        {
            "text": "hello world",
            "segments": [
                {"start": 0.0, "end": 0.8, "text": "hello"},
                {"start": 0.8, "end": 1.4, "text": " world"},
            ],
        }
    )

    assert result == {
        "text": "hello world",
        "chunks": [
            {"text": "hello", "timestamp": [0.0, 0.8]},
            {"text": " world", "timestamp": [0.8, 1.4]},
        ],
    }


def test_build_multipart_body_rejects_urls():
    with pytest.raises(ValueError, match="local file path"):
        _build_multipart_body({"model": "iic/SenseVoiceSmall"}, "https://example.com/audio.wav")


def test_transcribe_openai_compatible_posts_audio(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF....WAVE")
    args = SimpleNamespace(
        file_name=str(audio),
        model_name="iic/SenseVoiceSmall",
        language="zh",
        openai_compatible_url="http://127.0.0.1:8000/v1/audio/transcriptions",
        openai_compatible_api_key="token",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "text": "你好",
                    "segments": [{"start": 0, "end": 1, "text": "你好"}],
                }
            ).encode("utf-8")

    with patch("insanely_fast_whisper.cli.request.urlopen", return_value=FakeResponse()) as urlopen:
        result = transcribe_openai_compatible(args)

    sent_request = urlopen.call_args.args[0]
    body = sent_request.data

    assert sent_request.full_url == "http://127.0.0.1:8000/v1/audio/transcriptions"
    assert sent_request.headers["Authorization"] == "Bearer token"
    assert b'name="model"' in body
    assert b"iic/SenseVoiceSmall" in body
    assert b'name="language"' in body
    assert b"zh" in body
    assert b'name="file"; filename="sample.wav"' in body
    assert result == {"text": "你好", "chunks": [{"text": "你好", "timestamp": [0, 1]}]}
