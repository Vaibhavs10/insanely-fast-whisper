"""Minimal backend helpers for insanely-fast-whisper."""
from __future__ import annotations

import importlib
import importlib.util
import platform
from typing import Dict, Iterable, List, Optional


MLX_MODULES = {
    "whisper": "mlx_whisper",
    "parakeet": "parakeet_mlx",
}


class BackendSelectionError(RuntimeError):
    """Raised when a backend cannot be used in the current environment."""


class BackendDependencyError(RuntimeError):
    """Raised when an optional backend dependency is missing."""


def is_mlx_available() -> bool:
    """Return True when any MLX speech package can be imported."""

    return any(importlib.util.find_spec(module) is not None for module in MLX_MODULES.values())


def select_backend(
    *,
    device_id: str,
    requested_backend: str,
    platform_name: Optional[str] = None,
    mlx_available: Optional[bool] = None,
) -> str:
    """Resolve the backend for transcription based on CLI flags and environment."""

    platform_name = platform_name or platform.system()
    available = mlx_available if mlx_available is not None else is_mlx_available()

    if requested_backend == "transformers":
        return "transformers"

    if requested_backend == "mlx":
        if platform_name != "Darwin":
            raise BackendSelectionError("The MLX backend is only supported on macOS.")
        if not available:
            raise BackendSelectionError(
                "The MLX backend requires the optional MLX dependencies. Install insanely-fast-whisper[mac]."
            )
        return "mlx"

    if device_id == "mps" and platform_name == "Darwin":
        if available:
            return "mlx"
        raise BackendSelectionError(
            "Detected macOS/Metal but the MLX dependencies are missing. Install insanely-fast-whisper[mac] "
            "or select --backend transformers."
        )

    return "transformers"


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
        outputs["chunks"] = _segments_to_chunks(outputs.get("segments", []), timestamp)

    return outputs


def run_mlx_backend(
    *,
    audio_path: str,
    model_name: str,
    task: str,
    language: Optional[str],
    timestamp: str,
    mlx_model: str,
) -> Dict:
    module_name = MLX_MODULES.get(mlx_model)
    if module_name is None:
        raise BackendSelectionError(f"Unknown MLX model '{mlx_model}'.")

    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:  # pragma: no cover - exercised in real envs
        raise BackendDependencyError(
            "The MLX backend requires optional dependencies. Install insanely-fast-whisper[mac] and ensure the "
            f"'{module_name}' package is available."
        ) from exc

    if mlx_model == "whisper" and model_name == "openai/whisper-large-v3":
        repo = "mlx-community/whisper-large-v3"
    elif mlx_model == "parakeet":
        if model_name == "openai/whisper-large-v3":
            raise BackendSelectionError(
                "Specify --model-name with an MLX-compatible Parakeet checkpoint when using --mlx-model parakeet."
            )
        repo = model_name
    else:
        repo = model_name

    decode_options: Dict[str, str] = {"task": task}
    if language:
        decode_options["language"] = language

    if repo.split(".")[-1] == "en":
        decode_options.pop("task", None)

    result = module.transcribe(
        audio_path,
        path_or_hf_repo=repo,
        word_timestamps=timestamp == "word",
        verbose=False,
        **decode_options,
    )

    return {"text": result.get("text", ""), "chunks": _segments_to_chunks(result.get("segments", []), timestamp)}
