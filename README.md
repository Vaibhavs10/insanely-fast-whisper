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

We've added a CLI to enable fast transcriptions. Here's how you can use it:

### Transcribe your audio

```bash
pipx run insanely-fast-whisper --file_name <filename or URL>
```

Note: The CLI is opinionated and currently only works for Nvidia GPUs. Make sure to check out the defaults and the list of options you can play around with to maximise your transcription throughput. Run `pipx run insanely-fast-whisper --help` to get all the CLI arguments. 

## Roadmap

- [ ] Add benchmarks for Whisper.cpp
- [ ] Add benchmarks for 4-bit inference
- [ ] Add a light CLI script
- [ ] Deployment script with Inference API

## Community showcase

@ochen1 created a brilliant MVP for a CLI here: https://github.com/ochen1/insanely-fast-whisper-cli (Try it out now!)
