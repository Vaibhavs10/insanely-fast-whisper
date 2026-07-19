import json
import argparse
import mimetypes
import uuid
from pathlib import Path
from typing import Any
from urllib import request
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

from .utils.result import build_result

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
    "--transcript-path",
    required=False,
    default="output.json",
    type=str,
    help="Path to save the transcription output. (default: output.json)",
)
parser.add_argument(
    "--model-name",
    required=False,
    default="openai/whisper-large-v3",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
)
parser.add_argument(
    "--backend",
    required=False,
    default="transformers",
    type=str,
    choices=["transformers", "openai-compatible"],
    help="ASR backend to use. Use openai-compatible for local servers such as FunASR/SenseVoice. (default: transformers)",
)
parser.add_argument(
    "--openai-compatible-url",
    required=False,
    default="http://127.0.0.1:8000/v1/audio/transcriptions",
    type=str,
    help="OpenAI-compatible transcription endpoint used when --backend openai-compatible.",
)
parser.add_argument(
    "--openai-compatible-api-key",
    required=False,
    default=None,
    type=str,
    help="Optional bearer token for --backend openai-compatible.",
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

def _normalize_transcription_response(response: dict[str, Any]) -> dict[str, Any]:
    chunks = []
    for segment in response.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        chunks.append(
            {
                "text": segment.get("text", ""),
                "timestamp": [segment.get("start"), segment.get("end")],
            }
        )

    return {"text": response.get("text", ""), "chunks": chunks}


def _build_multipart_body(fields: dict[str, str], file_path: str) -> tuple[bytes, str]:
    path = Path(file_path)
    if not path.is_file():
        raise ValueError("--backend openai-compatible requires --file-name to be a local file path")

    boundary = f"----insanely-fast-whisper-{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode())
        body.extend(b"\r\n")

    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    body.extend(path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def transcribe_openai_compatible(args) -> dict[str, Any]:
    language = None if args.language == "None" else args.language
    fields = {
        "model": args.model_name,
        "response_format": "verbose_json",
    }
    if language:
        fields["language"] = language

    body, content_type = _build_multipart_body(fields, args.file_name)
    headers = {"Content-Type": content_type}
    if args.openai_compatible_api_key:
        headers["Authorization"] = f"Bearer {args.openai_compatible_api_key}"

    transcription_request = request.Request(
        args.openai_compatible_url,
        data=body,
        headers=headers,
        method="POST",
    )
    with request.urlopen(transcription_request) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return _normalize_transcription_response(payload)


def transcribe_transformers(args) -> dict[str, Any]:
    import torch
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        torch_dtype=torch.float16,
        device="mps" if args.device_id == "mps" else f"cuda:{args.device_id}",
        model_kwargs={"attn_implementation": "flash_attention_2"} if args.flash else {"attn_implementation": "sdpa"},
    )

    if args.device_id == "mps":
        torch.mps.empty_cache()
    # elif not args.flash:
        # pipe.model = pipe.model.to_bettertransformer()

    ts = "word" if args.timestamp == "word" else True

    language = None if args.language == "None" else args.language

    generate_kwargs = {"task": args.task, "language": language}

    if args.model_name.split(".")[-1] == "en":
        generate_kwargs.pop("task")

    return pipe(
        args.file_name,
        chunk_length_s=30,
        batch_size=args.batch_size,
        generate_kwargs=generate_kwargs,
        return_timestamps=ts,
    )


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
        if args.min_speakers > args.max_speakers:
            parser.error("--min-speakers cannot be greater than --max-speakers.")

    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Transcribing...", total=None)

        if args.backend == "openai-compatible":
            outputs = transcribe_openai_compatible(args)
        else:
            outputs = transcribe_transformers(args)

    if args.hf_token != "no_token":
        from .utils.diarization_pipeline import diarize

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
