import json
import argparse
import torch
from transformers import pipeline
from pyannote.audio import Pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
import torch


from .utils.diarize import (
    diarize_audio,
    preprocess_inputs,
    post_process_segments_and_transcripts,
)

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
    "--hf_token",
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


def main():
    args = parser.parse_args()

    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        torch_dtype=torch.float16,
        device="mps" if args.device_id == "mps" else f"cuda:{args.device_id}",
        model_kwargs={"use_flash_attention_2": args.flash},
    )

    if args.device_id == "mps":
        torch.mps.empty_cache()
    elif not args.flash:
        pipe.model = pipe.model.to_bettertransformer()

    ts = "word" if args.timestamp == "word" else True

    language = None if args.language == "None" else args.language

    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Transcribing...", total=None)

        outputs = pipe(
            args.file_name,
            chunk_length_s=30,
            batch_size=args.batch_size,
            generate_kwargs={"task": args.task, "language": language},
            return_timestamps=ts,
        )

    if args.hf_token != "no_token":
        diarization_pipeline = Pipeline.from_pretrained(
            checkpoint_path=args.diarization_model,
            use_auth_token=args.hf_token,
        )
        diarization_pipeline.to(
            torch.device("mps" if args.device_id == "mps" else f"cuda:{args.device_id}")
        )
        with Progress(
            TextColumn("🤗 [progress.description]{task.description}"),
            BarColumn(style="yellow1", pulse_style="white"),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task("[yellow]Segmenting...", total=None)

            inputs, diarizer_inputs = preprocess_inputs(inputs=args.file_name)

            segments = diarize_audio(diarizer_inputs, diarization_pipeline)

            segmented_transcript = post_process_segments_and_transcripts(
                segments, outputs["chunks"], group_by_speaker=False
            )

        segmented_transcript.append(outputs)

        with open(args.transcript_path, "w", encoding="utf8") as fp:
            json.dump(segmented_transcript, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed & speaker segmented go check it out over here 👉 {args.transcript_path}"
        )
    else:
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            json.dump(outputs, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed go check it out over here 👉 {args.transcript_path}"
        )
