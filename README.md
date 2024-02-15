# Insanely Fast Whisper

An opinionated CLI to transcribe Audio files w/ Whisper on-device! Powered by 🤗 *Transformers*, *Optimum* & *flash-attn*

**TL;DR** - Transcribe **150** minutes (2.5 hours) of audio in less than **98** seconds - with [OpenAI's Whisper Large v3](https://huggingface.co/openai/whisper-large-v3). Blazingly fast transcription is now a reality!⚡️

<p align="center">
<img src="https://huggingface.co/datasets/reach-vb/random-images/resolve/main/insanely-fast-whisper-img.png" width="615" height="308">
</p>

Not convinced? Here are some benchmarks we ran on a Nvidia A100 - 80GB 👇

| Optimisation type    | Time to Transcribe (150 mins of Audio) |
|------------------|------------------|
| large-v3 (Transformers) (`fp32`)             | ~31 (*31 min 1 sec*)             |
| large-v3 (Transformers) (`fp16` + `batching [24]` + `bettertransformer`) | ~5 (*5 min 2 sec*)            |
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2`)** | **~2 (*1 min 38 sec*)**            |
| distil-large-v2 (Transformers) (`fp16` + `batching [24]` + `bettertransformer`) | ~3 (*3 min 16 sec*)            |
| **distil-large-v2 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2`)** | **~1 (*1 min 18 sec*)**           |
| large-v2 (Faster Whisper) (`fp16` + `beam_size [1]`) | ~9.23 (*9 min 23 sec*)            |
| large-v2 (Faster Whisper) (`8-bit` + `beam_size [1]`) | ~8 (*8 min 15 sec*)            |

P.S. We also ran the benchmarks on a [Google Colab T4 GPU](/notebooks/) instance too!

P.P.S. This project originally started as a way to showcase benchmarks for Transformers, but has since evolved into a lightweight CLI for people to use. This is purely community driven. We add whatever community seems to have a strong demand for! 

Try the Relicate demo here: [![Replicate](https://replicate.com/cjwbw/insanely-fast-whisper/badge)](https://replicate.com/cjwbw/insanely-fast-whisper) 


## 🆕 Blazingly fast transcriptions via your terminal! ⚡️

We've added a CLI to enable fast transcriptions. Here's how you can use it:

Install `insanely-fast-whisper` with `pipx` (`pip install pipx` or `brew install pipx`):

```bash
pipx install insanely-fast-whisper
```
*Note: Due to a dependency on [`onnxruntime`, Python 3.12 is currently not supported](https://github.com/microsoft/onnxruntime/issues/17842). You can force a Python version (e.g. 3.11) by adding `--python python3.11` to the command.*

⚠️ If you have python 3.11.XX installed, `pipx` may parse the version incorrectly and install a very old version of `insanely-fast-whisper` without telling you (version `0.0.8`, which won't work anymore with the current `BetterTransformers`). In that case, you can install the latest version by passing `--ignore-requires-python` to `pip`:

```bash
pipx install insanely-fast-whisper --force --pip-args="--ignore-requires-python"
```

If you're installing with `pip`, you can pass the argument directly: `pip install insanely-fast-whisper --ignore-requires-python`.


Run inference from any path on your computer:

```bash
insanely-fast-whisper --file-name <filename or URL>
```
*Note: if you are running on macOS, you also need to add `--device-id mps` flag.*

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

> [!NOTE]
> The CLI is highly opinionated and only works on NVIDIA GPUs & Mac. Make sure to check out the defaults and the list of options you can play around with to maximise your transcription throughput. Run `insanely-fast-whisper --help` or `pipx run insanely-fast-whisper --help` to get all the CLI arguments along with their defaults. 


## CLI Options

The `insanely-fast-whisper` repo provides an all round support for running Whisper in various settings. Note that as of today 26th Nov, `insanely-fast-whisper` works on both CUDA and mps (mac) enabled devices.
```
  -h, --help            show this help message and exit
  --file-name FILE_NAME
                        Path or URL to the audio file to be transcribed.
  --device-id DEVICE_ID
                        Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon. (default: "0")
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
  --hf_token
                        Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips
```

## Frequently Asked Questions

**How to correctly install flash-attn to make it work with `insanely-fast-whisper`?**

Make sure to install it via `pipx runpip insanely-fast-whisper install flash-attn --no-build-isolation`. Massive kudos to @li-yifei for helping with this.

**How to solve an `AssertionError: Torch not compiled with CUDA enabled` error on Windows?**

The root cause of this problem is still unknown, however, you can resolve this by manually installing torch in the virtualenv like `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`. Thanks to @pto2k for all tdebugging this.

**How to avoid Out-Of-Memory (OOM) exceptions on Mac?**

The *mps* backend isn't as optimised as CUDA, hence is way more memory hungry. Typically you can run with `--batch-size 4` without any issues (should use roughly 12GB GPU VRAM). Don't forget to set `--device-id mps`.

## How to use Whisper without a CLI?

<details>
<summary>All you need to run is the below snippet:</summary>

```
pip install --upgrade transformers optimum accelerate
```

```python
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device="cuda:0", # or mps for Mac devices
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
)

outputs = pipe(
    "<FILE_NAME>",
    chunk_length_s=30,
    batch_size=24,
    return_timestamps=True,
)

outputs
```
</details>

## Acknowledgements

1. [OpenAI Whisper](https://github.com/openai/whisper) team for open sourcing such a brilliant check point.
2. Hugging Face Transformers team, specifically [Arthur](https://github.com/ArthurZucker), [Patrick](https://github.com/patrickvonplaten), [Sanchit](https://github.com/sanchit-gandhi) & [Yoach](https://github.com/ylacombe)  (alphabetical order) for continuing to maintain Whisper in Transformers.
3. Hugging Face [Optimum](https://github.com/huggingface/optimum) team for making the BetterTransformer API so easily accessible.
4. [Patrick Arminio](https://github.com/patrick91) for helping me tremendously to put together this CLI.

## Community showcase

1. @ochen1 created a brilliant MVP for a CLI here: https://github.com/ochen1/insanely-fast-whisper-cli (Try it out now!)
2. @arihanv created an app (Shush) using NextJS (Frontend) & Modal (Backend): https://github.com/arihanv/Shush (Check it outtt!)
3. @kadirnar created a python package on top of the transformers with optimisations: https://github.com/kadirnar/whisper-plus (Go go go!!!)
