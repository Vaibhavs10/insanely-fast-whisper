import sys
import types

import pytest

from insanely_fast_whisper.cli import disable_torchcodec_if_broken


class RaisingFinder:
    """Import hook that makes `import torchcodec` fail like a bad wheel does."""

    def find_spec(self, name, path=None, target=None):
        if name == "torchcodec":
            raise OSError("Could not load this library: libtorchcodec_core8.dylib")
        return None


@pytest.fixture
def no_torchcodec(monkeypatch):
    monkeypatch.delitem(sys.modules, "torchcodec", raising=False)
    finder = RaisingFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def test_broken_torchcodec_is_disabled(no_torchcodec):
    from transformers.pipelines import automatic_speech_recognition

    assert disable_torchcodec_if_broken() is True
    assert automatic_speech_recognition.is_torchcodec_available() is False


def test_healthy_torchcodec_is_left_alone(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    assert disable_torchcodec_if_broken() is False
