# Insanely Fast Whisper

Powered by 🤗 *Transformers*, *Optimum* & *flash-attn*

**TL;DR** - Transcribe **150** minutes (2.5 hours) of audio in less than **98** seconds - with [OpenAI's Whisper Large v3](https://huggingface.co/openai/whisper-large-v3). Blazingly fast transcription is now a reality!⚡️

Not convinced? Here are some benchmarks we ran on a Nvidia A100 - 80GB 👇

| Optimisation type    | Time to Transcribe (150 mins of Audio) |
|------------------|------------------|
| Transformers (`fp32`)             | ~31 (*31 min 1 sec*)             |
| Transformers (`fp16` + `batching [24]` + `bettertransformer`) | ~5 (*5 min 2 sec*)            |
| **Transformers (`fp16` + `batching [24]` + `Flash Attention 2`)** | **~2 (*1 min 38 sec*)**            |
| distil-whisper (`fp16` + `batching [24]` + `bettertransformer`) | ~3 (*3 min 16 sec*)            |
| **distil-whisper (`fp16` + `batching [24]` + `Flash Attention 2`)** | **~1 (*1 min 18 sec*)**           |
| Faster Whisper (`fp16` + `beam_size [1]`) | ~9.23 (*9 min 23 sec*)            |
| Faster Whisper (`8-bit` + `beam_size [1]`) | ~8 (*8 min 15 sec*)            |

P.S. We also ran the benchmarks on a [Google Colab T4 GPU](/notebooks/) instance too!

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


## CLI Options

The `insanely-fast-whisper` repo provides an all round support for running Whisper in various settings. Note that as of today 20th Nov, `insanely-fast-whisper` only works on CUDA enabled devices.
```
  -h, --help            show this help message and exit
  --file-name FILE_NAME
                        Path or URL to the audio file to be transcribed.
  --device-id DEVICE_ID
                        Device ID for your GPU (just pass the device ID number). (default: "0")
  --transcript-path TRANSCRIPT_PATH
                        Path to save the transcription output. (default: output.json)
  --model-name MODEL_NAME
                        Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)
  --task {transcribe,translate}
                        Task to perform: transcribe or translate to another language. (default: transcribe)
  --language LANGUAGE   
                        Language of the input audio. (default: "None" (Whisper auto-detects the language))
  --batch-size BATCH_SIZE
                        Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)
  --flash FLASH         
                        Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)
  --timestamp {chunk,word}
                        Whisper supports both chunked as well as word level timestamps. (default: chunk)
```

## Frequently Asked Questions

**How to correctly install flash-attn to make it work with `insanely-fast-whisper`?**

Make sure to install it via `pipx runpip insanely-fast-whisper install flash-attn --no-build-isolation`. Massive kudos to @li-yifei for helping with this.

**How to solve an `AssertionError: Torch not compiled with CUDA enabled` error on Windows?**

The root cause of this problem is still unkown, however, you can resolve this by manually installing torch in the virtualenv like `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`. Thanks to @pto2k for all tdebugging this.

## How to use Whisper without a CLI?

<details>
<summary>For older GPUs, all you need to run is:</summary>

```python
import torch
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v3",
                torch_dtype=torch.float16,
                device="cuda:0")

pipe.model = pipe.model.to_bettertransformer()

outputs = pipe("<FILE_NAME>",
               chunk_length_s=30,
               batch_size=24,
               return_timestamps=True)

outputs["text"]
```
</details>

<details>

<summary>For newer (A10, A100, H100s), use [Flash Attention](https://github.com/Dao-AILab/flash-attention):</summary>

```python
import torch
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v3",
                torch_dtype=torch.float16,
                model_kwargs={"use_flash_attention_2": True},
                device="cuda:0")

outputs = pipe("<FILE_NAME>",
               chunk_length_s=30,
               batch_size=24,
               return_timestamps=True)

outputs["text"]                
```
</details>

## Acknowledgements

1. [OpenAI Whisper](https://github.com/openai/whisper) team for open sourcing such a brilliant check point.
2. Hugging Face Transformers team, specifically [Arthur](https://github.com/ArthurZucker), [Patrick](https://github.com/patrickvonplaten), [Sanchit](https://github.com/sanchit-gandhi) & [Yoach](https://github.com/ylacombe)  (alphabetical order) for continuing to maintain Whisper in Transformers.
3. Hugging Face [Optimum](https://github.com/huggingface/optimum) team for making the BetterTransformer API so easily accessible.
4. [Patrick Arminio](https://github.com/patrick91) for helping me tremendously to put together this CLI.

## Community showcase

@ochen1 created a brilliant MVP for a CLI here: https://github.com/ochen1/insanely-fast-whisper-cli (Try it out now!)
