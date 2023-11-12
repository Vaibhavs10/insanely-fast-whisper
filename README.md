# Insanely Fast Whisper

Powered by 🤗 *Transformers*, *Optimum* & *flash-attn*

**TL;DR** - Transcribe **300** minutes (5 hours) of audio in less than **98** seconds - with [OpenAI's Whisper Large v3](https://huggingface.co/openai/whisper-large-v3). Blazingly fast transcription is now a reality!⚡️

Not convinced? Here are some benchmarks we ran on a free [Google Colab T4 GPU](/notebooks/)! 👇

| Optimisation type    | Time to Transcribe (150 mins of Audio) |
|------------------|------------------|
| Transformers (`fp32`)             | ~31 (*31 min 1 sec*)             |
| Transformers (`fp16` + `batching [24]` + `bettertransformer`) | ~5 (*5 min 2 sec*)            |
| Transformers (`fp16` + `batching [24]` + `Flash Attention 2`) | ~2 (*1 min 38 sec*)            |
| distil-whisper (`fp16` + `batching [24]` + `bettertransformer`) | ~3 (*3 min 16 sec*)            |
| distil-whisper (`fp16` + `batching [24]` + `Flash Attention 2`) | ~1 (*1 min 18 sec*)            |
| Faster Whisper (`fp16` + `beam_size [1]`) | ~9.23 (*9 min 23 sec*)            |
| Faster Whisper (`8-bit` + `beam_size [1]`) | ~8 (*8 min 15 sec*)            |

## 🆕 Blazingly fast transcriptions via your terminal! ⚡️

We've added a CLI to enable fast transcriptions. Here's how you can use it:

Install `insanely-fast-whisper` with `pipx`:

```bash
pipx install insanely-fast-whisper
```

Run inference from any path on your computer:

```bash
insanely-fast-whisper --file-name <filename or URL>
```

🔥 You can run [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) w/ [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) from this CLI too:

```bash
insanely-fast-whisper --file-name <filename or URL> --flash True 
```

🌟 You can run [distil-whisper](https://huggingface.co/distil-whisper) directly from this CLI too:

```bash
insanely-fast-whisper --model-name distil-whisper/large-v2 --file-name <filename or URL> 
```

Don't want to install `insanely-fast-whisper`? Just use `pipx run`:

```bash
pipx run insanely-fast-whisper --file-name <filename or URL>
```

Note: The CLI is opinionated and currently only works for Nvidia GPUs. Make sure to check out the defaults and the list of options you can play around with to maximise your transcription throughput. Run `insanely-fast-whisper --help` or `pipx run insanely-fast-whisper --help` to get all the CLI arguments and defaults. 


## How to use it without a CLI?

For older GPUs, all you need to run is:

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

For newer (A10, A100, H100s), use [Flash Attention]():

```python
import torch
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v2",
                torch_dtype=torch.float16,
                model_kwargs={"use_flash_attention_2": True},
                device="cuda:0")

outputs = pipe("<FILE_NAME>",
               chunk_length_s=30,
               batch_size=24,
               return_timestamps=True)

outputs["text"]                
```

## Roadmap

- [x] Add a light CLI script
- [ ] Deployment script with Inference API

## Community showcase

@ochen1 created a brilliant MVP for a CLI here: https://github.com/ochen1/insanely-fast-whisper-cli (Try it out now!)
