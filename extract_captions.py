import subprocess
import sys
from glob import glob
from pprint import pprint

import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from utils import load_frames, run_nvidia_smi

# model_spec = "Salesforce/blip2-flan-t5-xxl"
# model_spec = "Salesforce/blip2-flan-t5-xl"
model_spec = "Salesforce/blip2-opt-6.7b"
processor = AutoProcessor.from_pretrained(model_spec)

model = Blip2ForConditionalGeneration.from_pretrained(
    model_spec, torch_dtype=torch.float16
)

run_nvidia_smi()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

bsz = 256

with torch.no_grad():
    with torch.autocast("cuda"):
        # src_images = glob("/home/c3-0/rohitg/datasets/trove/frames/40102/*.jpeg")
        dataset = "approve-literacy"
        dataset_locations = {
            "approve-math": "/home/c3-0/rohitg/datasets/trove/videos/*.mp4",
            "approve-literacy": "/home/c3-0/rohitg/datasets/approve/literacy/videos/*.mp4",
            "yt-46k": "/home/c3-0/rohitg/datasets/yt8m/videos/*.mkv",
        }
        src_videos = glob(dataset_locations[dataset])
        print(f"Found {len(src_videos)} videos in {dataset})")
        for idx, vid_path in enumerate(src_videos):
            vid_name = vid_path.split("/")[-1].split(".")[0]
            if idx % 10 == 0:
                print(f"Processing video # {idx} {vid_name}")
            if idx % 100 == 0:
                run_nvidia_smi()
            frames = load_frames(vid_path, iframes_only=False, fps=1)
            if frames is False:
                continue
            inputs = processor(images=frames, return_tensors="pt").to(
                device, torch.float16
            )
            n_frames = inputs["pixel_values"].shape[0]
            # if n_frames > 300:
            #     print(vid_name, inputs["pixel_values"].shape)
            n_batches = n_frames // bsz + (n_frames % bsz != 0)
            generated_texts = []
            for i in range(n_batches):
                generated_ids = model.generate(
                    pixel_values=inputs["pixel_values"][i * bsz : (i + 1) * bsz],
                    max_new_tokens=20,
                )
                # print(generated_ids.shape)
                output = processor.batch_decode(generated_ids, skip_special_tokens=True)
                generated_texts += output
            # print(generated_texts)
            with open(f"blip2_captions/{dataset}/{vid_name}.txt", "wb") as f:
                for caption in generated_texts:
                    f.write((caption.strip() + "\n").encode("latin-1", "ignore"))
