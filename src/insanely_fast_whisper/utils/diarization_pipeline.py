import torch
from pyannote.audio import Pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

from .diarize import post_process_segments_and_transcripts, diarize_audio, \
    preprocess_inputs


def diarize(args, outputs):
    diarization_pipeline = Pipeline.from_pretrained(
        checkpoint_path=args.diarization_model,
        use_auth_token=args.hf_token,
    )
    torchDevice = None
    if args.device_id == "mps":
        torchDevice = "mps"
    elif args.diarize_CPU == 1:
        torchDevice = "cpu"
    else:
        torchDevice = f"cuda:{args.device_id}"
    diarization_pipeline.to(
        torch.device(torchDevice)
    )

    with Progress(
            TextColumn("🤗 [progress.description]{task.description}"),
            BarColumn(style="yellow1", pulse_style="white"),
            TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Segmenting...", total=None)

        inputs, diarizer_inputs = preprocess_inputs(inputs=args.file_name)

        segments = diarize_audio(diarizer_inputs, diarization_pipeline, args.num_speakers, args.min_speakers, args.max_speakers)

        return post_process_segments_and_transcripts(
            segments, outputs["chunks"], group_by_speaker=False
        )
