from typing import Any
import torch
import numpy as np
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
    pipeline,
)
from pyannote.audio import Pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Loads whisper models into memory to make running multiple predictions efficient"""
        self.model_cache = "model_cache"
        model_id = "openai/whisper-large-v3"
        torch_dtype = torch.float16
        self.device = "cuda:0"
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            cache_dir=self.model_cache,
        ).to(self.device)

        tokenizer = WhisperTokenizerFast.from_pretrained(
            model_id, cache_dir=self.model_cache
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_id, cache_dir=self.model_cache
        )

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            model_kwargs={"use_flash_attention_2": True},
            torch_dtype=torch_dtype,
            device=self.device,
        )
        self.diarization_pipeline = None

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        task: str = Input(
            choices=["transcribe", "translate"],
            default="transcribe",
            description="Task to perform: transcribe or translate to another language. (default: transcribe).",
        ),
        language: str = Input(
            default="None",
            choices=["None"] + sorted(list(LANGUAGES.values())),
            description="Language spoken in the audio, specify 'None' to perform language detection.",
        ),
        batch_size: int = Input(
            default=24,
            description="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24).",
        ),
        timestamp: str = Input(
            default="chunk",
            choices=["chunk", "word"],
            description="Whisper supports both chunked as well as word level timestamps. (default: chunk).",
        ),
        diarise_audio: bool = Input(
            default=False,
            description="Use Pyannote.audio to diarise the audio clips. You will need to provide hf_token below too.",
        ),
        hf_token: str = Input(
            default=None,
            description="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips. You need to agree to the terms in 'https://huggingface.co/pyannote/speaker-diarization-3.1' and 'https://huggingface.co/pyannote/segmentation-3.0' first.",
        ),
    ) -> Any:
        """Transcribes and optionally translates a single audio file"""

        if diarise_audio:
            assert (
                hf_token is not None
            ), "Please provide hf_token to diarise the audio clips"

        outputs = self.pipe(
            str(audio),
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs={
                "task": task,
                "language": None if language == "None" else language,
            },
            return_timestamps="word" if timestamp == "word" else True,
        )

        if diarise_audio:
            if self.diarization_pipeline is None:
                try:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token,
                        cache_dir=self.model_cache,
                    )
                    self.diarization_pipeline.to(torch.device(self.device))
                    print("diarization_pipeline loaded!")
                except Exception as e:
                    print(
                        f"https://huggingface.co/pyannote/speaker-diarization-3.1 cannot be loaded, please check the hf_token provided.: {e}"
                    )
            if self.diarization_pipeline is not None:
                print("Segmenting the audio clips.")
                inputs, diarizer_inputs = preprocess_inputs(inputs=str(audio))
                segments = diarize_audio(diarizer_inputs, self.diarization_pipeline)
                segmented_transcript = post_process_segments_and_transcripts(
                    segments, outputs["chunks"], group_by_speaker=False
                )
                segmented_transcript.append(outputs)
                print("Voila!✨ Your file has been transcribed & speaker segmented!")
                return segmented_transcript

        print("Voila!✨ Your file has been transcribed!")
        return outputs


def preprocess_inputs(inputs):
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, 16000)

    if isinstance(inputs, dict):
        # Accepting `"array"` which is the key defined in `datasets` for better integration
        if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
            raise ValueError(
                "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                "containing the sampling_rate associated with that array"
            )

        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            # Remove path which will not be used from `datasets`.
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        if in_sampling_rate != 16000:
            inputs = F.resample(
                torch.from_numpy(inputs), in_sampling_rate, 16000
            ).numpy()

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for ASRDiarizePipeline"
        )

    # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs


def diarize_audio(diarizer_inputs, diarization_pipeline):
    diarization = diarization_pipeline(
        {"waveform": diarizer_inputs, "sample_rate": 16000},
    )

    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "segment": {"start": segment.start, "end": segment.end},
                "track": track,
                "label": label,
            }
        )

    # diarizer output may contain consecutive segments from the same speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
    # we combine these segments to give overall timestamps for each speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        # check if we have changed speaker ("label")
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            # add the start/end times for the super-segment to the new list
            new_segments.append(
                {
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = segments[i]

    # add the last segment(s) if there was no speaker change
    new_segments.append(
        {
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": cur_segment["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        }
    )

    return new_segments


def post_process_segments_and_transcripts(new_segments, transcript, group_by_speaker):
    # get the end timestamps for each chunk from the ASR output
    end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
    segmented_preds = []

    # align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        # get the diarizer end timestamp
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append(
                {
                    "speaker": segment["speaker"],
                    "text": "".join(
                        [chunk["text"] for chunk in transcript[: upto_idx + 1]]
                    ),
                    "timestamp": (
                        transcript[0]["timestamp"][0],
                        transcript[upto_idx]["timestamp"][1],
                    ),
                }
            )
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        transcript = transcript[upto_idx + 1 :]
        end_timestamps = end_timestamps[upto_idx + 1 :]

    return segmented_preds
