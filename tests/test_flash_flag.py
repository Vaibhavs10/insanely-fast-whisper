import argparse

import pytest

from insanely_fast_whisper.cli import parser, str2bool

BASE_ARGS = ["--file-name", "audio.wav"]


def parse_flash(extra):
    return parser.parse_args(BASE_ARGS + extra).flash


def test_flash_defaults_to_false():
    assert parse_flash([]) is False


def test_flash_bare_flag_enables():
    assert parse_flash(["--flash"]) is True


@pytest.mark.parametrize("value", ["True", "true", "1", "yes", "Y"])
def test_flash_truthy_values(value):
    assert parse_flash(["--flash", value]) is True


@pytest.mark.parametrize("value", ["False", "false", "0", "no", "N"])
def test_flash_falsy_values(value):
    assert parse_flash(["--flash", value]) is False


def test_flash_rejects_garbage():
    with pytest.raises(SystemExit):
        parser.parse_args(BASE_ARGS + ["--flash", "maybe"])


def test_str2bool_rejects_garbage_directly():
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("maybe")
