"""Backend helpers for insanely-fast-whisper."""
from __future__ import annotations

import importlib
import importlib.util
import platform
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional


class BackendSelectionError(RuntimeError):
    """Raised when an invalid backend is requested."""


class BackendDependencyError(RuntimeError):
    """Raised when optional backend dependencies are missing."""


def is_mlx_available() -> bool:
    """Return True when at least one MLX speech package can be located."""

    return any(
        importlib.util.find_spec(package) is not None  # type: ignore[attr-defined]
        for package in ("mlx_whisper", "mlx_parakeet")
    )


def select_backend(
    *,
    device_id: str,
    requested_backend: str,
    platform_name: Optional[str] = None,
    mlx_available: Optional[bool] = None,
) -> str:
    """Resolve the backend that should be used for transcription."""

    if platform_name is None:
        platform_name = platform.system()

    if requested_backend == "transformers":
        return "transformers"

    if requested_backend == "mlx":
        if platform_name != "Darwin":
            raise BackendSelectionError("The MLX backend is only supported on macOS.")
        if not (mlx_available if mlx_available is not None else is_mlx_available()):
            raise BackendSelectionError(
                "The MLX backend requires the optional MLX dependencies. "
                "Install insanely-fast-whisper[mac]."
            )
        return "mlx"

    # Auto mode
    if device_id != "mps":
        return "transformers"

    if platform_name != "Darwin":
        return "transformers"

    available = mlx_available if mlx_available is not None else is_mlx_available()
    if not available:
        raise BackendSelectionError(
            "Detected macOS/Metal but the MLX dependencies are missing. "
            "Install insanely-fast-whisper[mac] or select --backend transformers."
        )

    return "mlx"


@contextmanager
def _progress(task_description: str):
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task(task_description, total=None)
        yield


def _import_mlx_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError) as exc:  # pragma: no cover - exercised in real envs
        raise BackendDependencyError(
            "The MLX backend requires optional dependencies. Install "
            "insanely-fast-whisper[mac] and ensure the '{}' package is available.".format(module_name)
        ) from exc


def _segments_to_chunks(segments: Iterable[Dict], timestamp_mode: str) -> List[Dict]:
    chunks: List[Dict] = []

    if timestamp_mode == "word":
        for segment in segments:
            for word in segment.get("words") or []:
                start = word.get("start")
                end = word.get("end")
                if start is None or end is None:
                    continue
                text = word.get("word") or word.get("text", "")
                chunks.append({"timestamp": (start, end), "text": text})
        if chunks:
            return chunks

    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        if start is None or end is None:
            continue
        chunks.append({"timestamp": (start, end), "text": segment.get("text", "")})

    return chunks


def run_transformers_backend(
    *,
    audio_path: str,
    model_name: str,
    device_id: str,
    batch_size: int,
    task: str,
    language: Optional[str],
    timestamp: str,
    flash: bool,
) -> Dict:
    from transformers import pipeline
    import torch

    attn_impl = "flash_attention_2" if flash else "sdpa"
    ts = "word" if timestamp == "word" else True

    generate_kwargs: Dict[str, str] = {"task": task}
    if language:
        generate_kwargs["language"] = language

    if model_name.split(".")[-1] == "en":
        generate_kwargs.pop("task", None)

    with _progress("[yellow]Transcribing with Transformers..."):
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16,
            device="mps" if device_id == "mps" else f"cuda:{device_id}",
            model_kwargs={"attn_implementation": attn_impl},
        )

        outputs = pipe(
            audio_path,
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=ts,
        )

    if device_id == "mps":
        torch.mps.empty_cache()

    if "chunks" not in outputs:
        chunks = _segments_to_chunks(outputs.get("segments", []), timestamp)
        outputs["chunks"] = chunks

    return outputs


def _resolve_whisper_repo(model_name: str) -> str:
    if model_name == "openai/whisper-large-v3":
        return "mlx-community/whisper-large-v3"
    return model_name


def _resolve_parakeet_repo(model_name: str) -> str:
    if model_name == "openai/whisper-large-v3":
        raise BackendSelectionError(
            "Specify --model-name with an MLX-compatible Parakeet checkpoint when using --mlx-model parakeet."
        )
    return model_name


def run_mlx_backend(
    *,
    audio_path: str,
    model_name: str,
    task: str,
    language: Optional[str],
    timestamp: str,
    batch_size: int,
    mlx_model: str,
) -> Dict:
    decode_options: Dict[str, str] = {"task": task}
    if language:
        decode_options["language"] = language

    if mlx_model == "whisper":
        module = _import_mlx_module("mlx_whisper")
        repo = _resolve_whisper_repo(model_name)
        if repo.split(".")[-1] == "en":
            decode_options.pop("task", None)

        with _progress("[yellow]Transcribing with MLX Whisper..."):
            result = module.transcribe(
                audio_path,
                path_or_hf_repo=repo,
                word_timestamps=timestamp == "word",
                verbose=False,
                **decode_options,
            )

    elif mlx_model == "parakeet":
        module = _import_mlx_module("mlx_parakeet")
        repo = _resolve_parakeet_repo(model_name)
        if repo.split(".")[-1] == "en":
            decode_options.pop("task", None)

        with _progress("[yellow]Transcribing with MLX Parakeet..."):
            result = module.transcribe(
                audio_path,
                path_or_hf_repo=repo,
                word_timestamps=timestamp == "word",
                verbose=False,
                **decode_options,
            )
    else:
        raise BackendSelectionError(f"Unknown MLX model '{mlx_model}'.")

    chunks = _segments_to_chunks(result.get("segments", []), timestamp)
    return {"text": result.get("text", ""), "chunks": chunks}
