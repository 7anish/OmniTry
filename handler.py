import runpod
import torch
import base64
import io
import os
from PIL import Image
from diffusers import FluxFillPipeline
from huggingface_hub import snapshot_download

pipe = None

def download_models():
    if not os.path.exists("checkpoints/FLUX.1-Fill-dev"):
        print("Downloading FLUX model...")
        snapshot_download(
            'black-forest-labs/FLUX.1-Fill-dev',
            local_dir='checkpoints/FLUX.1-Fill-dev',
            token=os.environ.get("HF_TOKEN")
        )
    if not os.path.exists("checkpoints/omnitry_lora"):
        print("Downloading OmniTry LoRA...")
        snapshot_download(
            'Kunbyte/OmniTry',
            local_dir='checkpoints/omnitry_lora'
        )

def load_model():
    global pipe
    if pipe is not None:
        return
    download_models()
    pipe = FluxFillPipeline.from_pretrained(
        "checkpoints/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.load_lora_weights("checkpoints/omnitry_lora/omnitry_v1_unified.safetensors")
    print("Model loaded.")

# ... rest of handler stays the same