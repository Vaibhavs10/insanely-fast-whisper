import json

import typer
import torch

from transformers import pipeline
from typing_extensions import Annotated
from typing import Optional

# TODO: Check if optimum is installed
# TODO: Add a function to check hardware/ GPU and choose an appropriate model


def main(
    file_name: Annotated[str, typer.Argument(default=None)],
    device_id: Annotated[str, typer.Argument(default=None)],
    transcript_path: Annotated[str, typer.Argument(default=None)],    
    model_name: Annotated[Optional[str], typer.Argument(default="openai/whisper-large-v2")],
    task: Annotated[Optional[str], typer.Argument(default="transcribe")],
    language: Annotated[Optional[str], typer.Argument(default="en")],
):
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch.float16,
        device=f"cuda:{device_id}",
    )

    pipe.model = pipe.model.to_bettertransformer()

    outputs = pipe(
        file_name,
        chunk_length_s=30,
        batch_size=24,
        generate_kwargs={"task": task, "language": language},
        return_timestamps=True,
    )

    with open(transcript_path, "w") as fp:
        json.dump(outputs, fp)


if __name__ == "__main__":
    typer.run(main)
