import json

import argparse
import torch
from transformers import pipeline

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
    help='Device ID for your GPU (just pass the device ID number). (default: "0")',
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
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v2)",
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
    default="en",
    help='Language of the input audio. (default: "en" (English))',
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
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
)
parser.add_argument(
    "--timestamp",
    required=False,
    type=str,
    default="chunk",
    choices=["chunk", "word"],
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
)


def main():
    args = parser.parse_args()

    if args.flash == True:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model_name,
            torch_dtype=torch.float16,
            device=f"cuda:{args.device_id}",
            model_kwargs={"use_flash_attention_2": True},
        )
    else:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model_name,
            torch_dtype=torch.float16,
            device=f"cuda:{args.device_id}",
        )

        pipe.model = pipe.model.to_bettertransformer()

    if args.timestamp == "word":
        ts = "word"

    else:
        ts = True

    outputs = pipe(
        args.file_name,
        chunk_length_s=30,
        batch_size=args.batch_size,
        generate_kwargs={"task": args.task, "language": args.language},
        return_timestamps=ts,
    )

    with open(args.transcript_path, "w") as fp:
        json.dump(outputs, fp)

    print(
        f"Voila! Your file has been transcribed go check it out over here! {args.transcript_path}"
    )
