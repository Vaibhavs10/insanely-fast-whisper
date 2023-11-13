from typing import Any
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
    pipeline,
)
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Loads whisper models into memory to make running multiple predictions efficient"""
        model_cache = "model_cache"
        local_files_only = True  # set to true after the model is cached to model_cache
        model_id = "openai/whisper-large-v3"
        torch_dtype = torch.float16
        device = "cuda:0"
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            cache_dir=model_cache,
            local_files_only=local_files_only,
        ).to(device)

        tokenizer = WhisperTokenizerFast.from_pretrained(
            model_id, cache_dir=model_cache, local_files_only=local_files_only
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_id, cache_dir=model_cache, local_files_only=local_files_only
        )

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            model_kwargs={"use_flash_attention_2": True},
            torch_dtype=torch_dtype,
            device=device,
        )

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        task: str = Input(
            choices=["transcribe", "translate"],
            default="transcribe",
            description="Task to perform: transcribe or translate to another language. (default: transcribe).",
        ),
        language: str = Input(
            default=None,
            description="Optional. Language spoken in the audio, specify None to perform language detection.",
        ),
        batch_size: int = Input(
            default=24,
            description="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24).",
        ),
        return_timestamps: bool = Input(
            default=True,
            description="Return timestamps information when set to True.",
        ),
    ) -> Any:
        """Transcribes and optionally translates a single audio file"""

        outputs = self.pipe(
            str(audio),
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs={"task": task, "language": language},
            return_timestamps=return_timestamps,
        )
        return outputs
