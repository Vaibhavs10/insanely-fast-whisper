import argparse
import json
from typing import Any, Dict, Iterable, Optional

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from .utils.diarization_pipeline import diarize
from .utils.result import build_result

DEFAULT_TRANSFORMERS_MODEL = "openai/whisper-large-v3"
MLX_DEFAULT_MODELS = {
    "whisper": "mlx-community/whisper-large-v3",
    "parakeet": "mlx-community/parakeet-tdt_ctc-110m",
}


parser = argparse.ArgumentParser(description="Automatic Speech Recognition")
parser.add_argument(
    "--file-name",
    required=True,
    type=str,
    help="Path or URL to the audio file to be transcribed.",
)
parser.add_argument(
    "--device-id",
    required=False,
    default="0",
    type=str,
    help='Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon. (default: "0")',
)
parser.add_argument(
    "--backend",
    required=False,
    default="transformers",
    choices=["transformers", "mlx"],
    help="Transcription backend to use. Use 'transformers' for the original PyTorch pipeline or 'mlx' for Apple MLX models. (default: transformers)",
)
parser.add_argument(
    "--mlx-family",
    required=False,
    default="whisper",
    choices=["whisper", "parakeet"],
    help="When using the MLX backend, choose which model family to load. (default: whisper)",
)
parser.add_argument(
    "--transcript-path",
    required=False,
    default="output.json",
    type=str,
    help="Path to save the transcription output. (default: output.json)",
)
parser.add_argument(
    "--model-name",
    required=False,
    default=DEFAULT_TRANSFORMERS_MODEL,
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
)
parser.add_argument(
    "--task",
    required=False,
    default="transcribe",
    type=str,
    choices=["transcribe", "translate"],
    help="Task to perform: transcribe or translate to another language. (default: transcribe)",
)
parser.add_argument(
    "--language",
    required=False,
    type=str,
    default="None",
    help='Language of the input audio. (default: "None" (Whisper auto-detects the language))',
)
parser.add_argument(
    "--batch-size",
    required=False,
    type=int,
    default=24,
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
)
parser.add_argument(
    "--flash",
    required=False,
    type=bool,
    default=False,
    help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
)
parser.add_argument(
    "--timestamp",
    required=False,
    type=str,
    default="chunk",
    choices=["chunk", "word"],
    help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
)
parser.add_argument(
    "--hf-token",
    required=False,
    default="no_token",
    type=str,
    help="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips",
)
parser.add_argument(
    "--diarization_model",
    required=False,
    default="pyannote/speaker-diarization-3.1",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization)",
)
parser.add_argument(
    "--num-speakers",
    required=False,
    default=None,
    type=int,
    help="Specifies the exact number of speakers present in the audio file. Useful when the exact number of participants in the conversation is known. Must be at least 1. Cannot be used together with --min-speakers or --max-speakers. (default: None)",
)
parser.add_argument(
    "--min-speakers",
    required=False,
    default=None,
    type=int,
    help="Sets the minimum number of speakers that the system should consider during diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be less than or equal to --max-speakers if both are specified. (default: None)",
)
parser.add_argument(
    "--max-speakers",
    required=False,
    default=None,
    type=int,
    help="Defines the maximum number of speakers that the system should consider in diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be greater than or equal to --min-speakers if both are specified. (default: None)",
)


def _progress() -> Progress:
    return Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    )


def _transformers_backend(args, return_timestamps: Any, generate_kwargs: Dict[str, Any]):
    try:
        from transformers import pipeline
    except ImportError:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Transformers backend requires the 'transformers' package. "
            "Install it with `uv add transformers accelerate torch`."
        )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Transformers backend requires PyTorch. "
            "Install it with `uv add torch`."
        ) from exc

    if args.device_id == "cpu":
        device = "cpu"
    elif args.device_id == "mps":
        device = "mps"
    else:
        device = f"cuda:{args.device_id}"

    torch_dtype = torch.float32 if device == "cpu" else torch.float16
    model_kwargs = (
        {"attn_implementation": "flash_attention_2"}
        if args.flash
        else {"attn_implementation": "sdpa"}
    )

    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs,
    )

    if device == "mps":
        torch.mps.empty_cache()

    with _progress() as progress:
        progress.add_task("[yellow]Transcribing...", total=None)
        outputs = pipe(
            args.file_name,
            chunk_length_s=30,
            batch_size=args.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=return_timestamps,
        )
    return outputs


def _chunks_from_segments(segments: Iterable[Dict[str, Any]]):
    for segment in segments:
        yield {
            "text": segment.get("text", ""),
            "timestamp": (segment.get("start"), segment.get("end")),
        }


def _chunks_from_words(segments: Iterable[Dict[str, Any]]):
    for segment in segments:
        words = segment.get("words", []) or []
        for word in words:
            text = word.get("word", word.get("text", ""))
            yield {
                "text": text,
                "timestamp": (word.get("start"), word.get("end")),
            }


def _mlx_backend(args, language: Optional[str], want_words: bool):
    if args.model_name == DEFAULT_TRANSFORMERS_MODEL:
        args.model_name = MLX_DEFAULT_MODELS[args.mlx_family]

    if args.mlx_family == "whisper":
        try:
            import mlx_whisper
        except ImportError:
            raise RuntimeError(
                "MLX Whisper backend requires 'mlx' and 'mlx-whisper'. "
                "Install them with `uv add .[mlx]`."
            )

        with _progress() as progress:
            progress.add_task("[yellow]Transcribing...", total=None)
            outputs = mlx_whisper.transcribe(
                args.file_name,
                path_or_hf_repo=args.model_name,
                task=args.task,
                language=language,
                word_timestamps=want_words,
            )

        segments = outputs.get("segments", [])
        if want_words:
            chunks = list(_chunks_from_words(segments))
            if not chunks:
                chunks = list(_chunks_from_segments(segments))
        else:
            chunks = list(_chunks_from_segments(segments))

        return {
            "text": outputs.get("text", ""),
            "chunks": chunks,
        }

    if args.mlx_family == "parakeet":
        if args.task == "translate":
            parser.error("Parakeet models do not currently support translation tasks.")

        try:
            from parakeet_mlx import from_pretrained
        except ImportError:
            raise RuntimeError(
                "Parakeet MLX backend requires 'mlx' and 'parakeet-mlx'. "
                "Install them with `uv add .[mlx]`."
            )

        with _progress() as progress:
            task_id = progress.add_task("[yellow]Transcribing...", total=1.0)

            def _on_chunk(current: float, total: float) -> None:
                if total > 0:
                    progress.update(
                        task_id, completed=min(current, total), total=total
                    )

            model = from_pretrained(args.model_name)
            result = model.transcribe(
                args.file_name,
                chunk_callback=_on_chunk,
            )

        if want_words:
            chunks = [
                {"text": token.text, "timestamp": (token.start, token.end)}
                for token in result.tokens
            ]
        else:
            chunks = [
                {"text": sentence.text, "timestamp": (sentence.start, sentence.end)}
                for sentence in result.sentences
            ]

        return {
            "text": result.text,
            "chunks": chunks,
        }

    parser.error(f"Unsupported MLX family '{args.mlx_family}'.")


def main():
    args = parser.parse_args()

    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        parser.error("--num-speakers cannot be used together with --min-speakers or --max-speakers.")

    if args.num_speakers is not None and args.num_speakers < 1:
        parser.error("--num-speakers must be at least 1.")

    if args.min_speakers is not None and args.min_speakers < 1:
        parser.error("--min-speakers must be at least 1.")

    if args.max_speakers is not None and args.max_speakers < 1:
        parser.error("--max-speakers must be at least 1.")

    if args.min_speakers is not None and args.max_speakers is not None and args.min_speakers > args.max_speakers:
        parser.error("--min-speakers cannot be greater than --max-speakers.")

    return_timestamps = "word" if args.timestamp == "word" else True
    language = None if args.language == "None" else args.language

    generate_kwargs = {"task": args.task, "language": language}
    if args.model_name.split(".")[-1] == "en":
        generate_kwargs.pop("task")

    if args.backend == "mlx":
        outputs = _mlx_backend(args, language, args.timestamp == "word")
    else:
        outputs = _transformers_backend(args, return_timestamps, generate_kwargs)

    if args.hf_token != "no_token":
        speakers_transcript = diarize(args, outputs)
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result(speakers_transcript, outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed & speaker segmented go check it out over here 👉 {args.transcript_path}"
        )
    else:
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result([], outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed go check it out over here 👉 {args.transcript_path}"
        )
