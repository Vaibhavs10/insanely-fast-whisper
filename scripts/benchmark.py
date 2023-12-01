import time
import json
import logging

import torch
import transformers
from transformers import pipeline

transformers.utils.logging.set_verbosity_error()

models = ["openai/whisper-large-v3", "distil-whisper/large-v2"]
test_flash_attention = [True, False]
device = "cuda:0"
batch_sizes = [1, 8, 16, 24]

file_name = (
    "https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/ted_60.wav"
)

results = []

for model in models:
    print(f"Running Model: {model}")

    for fa2 in test_flash_attention:
        print(f"Flash Attention: {fa2}")

        for batch_size in batch_sizes:
            print(f"Batch Size: {batch_size}")

            torch.cuda.empty_cache()

            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                torch_dtype=torch.float16,
                device=device,
                model_kwargs={"use_flash_attention_2": fa2},
            )

            if fa2 == False:
                pipe.model = pipe.model.to_bettertransformer()

            start = time.time()
            outputs = pipe(
                file_name,
                chunk_length_s=30,
                batch_size=batch_size,
                return_timestamps=True,
            )
            end = time.time()
            total_time = end - start
            print(f"Total time: {total_time}")

            max_mem = torch.cuda.max_memory_reserved()
            max_mem_mb = max_mem / (1024 * 1024)
            print(f"Total memory: {max_mem_mb}")

            temp = {
                "Model": model,
                "Flash": fa2,
                "Batch": batch_size,
                "Time": total_time,
                "Memory": max_mem_mb,
            }

            results.append(temp)

            torch.cuda.reset_peak_memory_stats(device=device)

with open("benchmark.json", "w") as outfile:
    json.dump(results, outfile)
