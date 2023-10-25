import json

import typer
import torch

from transformers import pipeline

# TODO: Check if accelerate is installed


def main(file_name: str, device_id: str, transcript_path: str):
    pipe = pipeline(
        "automatic-speech-recognition",
        "openai/whisper-large-v2",
        torch_dtype=torch.float16,
        device=f"cuda:{device_id}",
    )

    pipe.model = pipe.model.to_bettertransformer()

    outputs = pipe(file_name, chunk_length_s=30, batch_size=24, return_timestamps=True)

    with open(transcript_path, "w") as fp:
        json.dump(outputs, fp)


if __name__ == "__main__":
    typer.run(main)
