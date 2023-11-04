# Insanely Fast Whisper

Powered by 🤗 *Transformers* & *Optimum*

**TL;DR** - Transcribe **300** minutes (5 hours) of audio in less than **10** minutes - with [OpenAI's Whisper Large v2](https://huggingface.co/openai/whisper-large-v2). Blazingly fast transcription is now a reality!⚡️

Basically all you need to do is this:

```python
import torch
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v2",
                torch_dtype=torch.float16,
                device="cuda:0")

pipe.model = pipe.model.to_bettertransformer()

outputs = pipe("<FILE_NAME>",
               chunk_length_s=30,
               batch_size=24,
               return_timestamps=True)

outputs["text"]
```

Not convinced? Here are some benchmarks we ran on a free [Google Colab T4 GPU](https://colab.research.google.com/github/Vaibhavs10/insanely-fast-whisper/blob/main/infer_transformers_whisper_large_v2.ipynb)! 👇

| Optimisation type    | Time to Transcribe (150 mins of Audio) |
|------------------|------------------|
| Transformers (`fp32`)             | ~31 (*31 min 1 sec*)             |
| Transformers (`fp32` + `batching [8]`)           | ~13 (*13 min 19 sec*)             |
| Transformers (`fp16` + `batching [16]`) | ~6 (*6 min 13 sec*)             |
| Transformers (`fp16` + `batching [16]` + `bettertransformer`) | ~5.42 (*5 min 42 sec*)            |
| Transformers (`fp16` + `batching [24]` + `bettertransformer`) | ~5 (*5 min 2 sec*)            |
| Transformers (distil-whisper) (`fp16` + `batching [24]` + `bettertransformer`) | ~3 (*3 min 16 sec*)            |
| Faster Whisper (`fp16` + `beam_size [1]`) | ~9.23 (*9 min 23 sec*)            |
| Faster Whisper (`8-bit` + `beam_size [1]`) | ~8 (*8 min 15 sec*)            |

## 🆕 You can now access blazingly fast transcriptions via your terminal! ⚡️

We've added a v1 CLI to enable fast transcriptions. Here's how you can use it.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Transcribe your audio

```bash
python transcribe.py --file_name <filename or URL>
```
Note: The CLI is opinionated and currently only works for Nvidia GPUs. Make sure to check out the defaults and the list of options you can play around with to maximise your transcription throughput. Run `python transcribe.py --help` to get all the CLI arguments. 

### How does this all work?

Here-in, we'll dive into optimisations that can make Whisper faster for fun and profit! Our goal is to be able to transcribe a 2-3 hour long audio in the fastest amount of time possible. We'll start with the most basic usage and work our way up to make it fast!

The only fitting test audio to use for our benchmark would be [Lex interviewing Sam Altman](https://www.youtube.com/watch?v=L_Guz73e6fw&t=8s). We'll use the audio file corresponding to his podcast. I uploaded it on a wee dataset on the hub [here](https://huggingface.co/datasets/reach-vb/random-audios/blob/main/sam_altman_lex_podcast_367.flac).

## Installation

```python
pip install -q --upgrade torch torchvision torchaudio
pip install -q git+https://github.com/huggingface/transformers
pip install -q accelerate optimum
pip install -q ipython-autotime
```

Let's download the audio file corresponding to the podcast.

```python
wget https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/sam_altman_lex_podcast_367.flac
```

## Base Case

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

Sample output:
```
We have been a misunderstood and badly mocked org for a long time. When we started, we announced the org at the end of 2015 and said we were going to work on AGI, people thought we were batshit insan
```

*Time to transcribe the entire podcast*: **31min 1s**

## Batching

```python
outputs = pipe("sam_altman_lex_podcast_367.flac", 
               chunk_length_s=30,
               batch_size=8,
               return_timestamps=True)

outputs["text"][:200]
```

*Time to transcribe the entire podcast*: **13min 19s**

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

*Time to transcribe the entire podcast*: **6min 13s**

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
               return_timestamps=True)

outputs["text"][:200]
```

*Time to transcribe the entire podcast*: **5min 2s**

## Roadmap

- [ ] Add benchmarks for Whisper.cpp
- [ ] Add benchmarks for 4-bit inference
- [ ] Add a light CLI script
- [ ] Deployment script with Inference API

## Community showcase

@ochen1 created a brilliant MVP for a CLI here: https://github.com/ochen1/insanely-fast-whisper-cli (Try it out now!)
