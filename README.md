# Insanely Fast Whisper

Powered by 🤗 Transformers & Optimum

**TL;DR** - A one stop shop walkthrough, to fast inference with Whisper (large) **5x faster** on a consumer GPU with **less than 8GB GPU VRAM**, all with comparable performance to full-finetuning. ⚡️

Not convinced? Here are some benchmarks we ran on a free Google Colab T4 GPU! 👇

| Optimisation type    | Inference time |
|------------------|------------------|
| Transformers (`fp32`)             | <1%              |
| Transformers (`fp16`)         | <0.9%            |
| Transformers (`fp16` + `batching`) | 100%             |
| Transformers (`fp16` + `batching` + `bettertransformer`) | 100%             |
| faster_whisper (`fp16`) | 100%             |
| faster_whisper (`int8_fp16`) | 100%             |

Here-in, we'll dive into optimisations that can make Whisper faster for fun and profit! Our goal is to be able to transcribe a 2-3 hour long audio in the fastest amount of time possible. We'll start with the most basic usage and work our way up to make it fast!

The only fitting test audio to use for our benchmark would be [Lex interviewing Sam Altman](https://www.youtube.com/watch?v=L_Guz73e6fw&t=8s). We'll use the audio file corresponding to his podcast. I uploaded it on a wee dataset on the hub [here](https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac).

## Installation

```python
pip install -q --upgrade torch torchvision torchaudio
pip install -q git+https://github.com/huggingface/transformers
pip install -q accelerate optimum bitsandbytes
pip install -q ipython-autotime
```

```python
wget https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/sam_altman_lex_podcast_367.flac
```

## Base case

```python
import torch
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v2",
                device="cuda:0")
```

```python
outputs = pipe("sam_altman_lex_podcast_367.flac", 
               chunk_length_s=30,
               return_timestamps=True)

outputs["text"][:200]
```

## Batching

```python
outputs = pipe("sam_altman_lex_podcast_367.flac", 
               chunk_length_s=30,
               batch_size=8,
               return_timestamps=True)

outputs["text"][:200]
```

## Half-Precision

```python
pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v2",
                torch_dtype=torch.float16,
                device="cuda:0")                
```

```python
outputs = pipe("sam_altman_lex_podcast_367.flac",
               chunk_length_s=30,
               batch_size=16,
               return_timestamps=True)

outputs["text"][:200]
```

## BetterTransformer w/ Optimum

```python
pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v2",
                torch_dtype=torch.float16,
                device="cuda:0")

pipe.model = pipe.model.to_bettertransformer()
```

```python
outputs = pipe("sam_altman_lex_podcast_367.flac",
               chunk_length_s=30,
               batch_size=24,
               return_timestamps=True,)["text"][:200]

outputs["text"][:200]
```