import runpod
import torch
import base64
import io
import os

print("Starting handler...")

try:
    from PIL import Image
    print("PIL imported")
    from diffusers import FluxFillPipeline
    print("diffusers imported")
    from huggingface_hub import snapshot_download
    print("huggingface_hub imported")
except Exception as e:
    print(f"Import error: {e}")
    raise

pipe = None

def download_models():
    if not os.path.exists("checkpoints/FLUX.1-Fill-dev"):
        print("Downloading FLUX model...")
        snapshot_download(
            'black-forest-labs/FLUX.1-Fill-dev',
            local_dir='checkpoints/FLUX.1-Fill-dev',
            token=os.environ.get("HF_TOKEN")
        )
        print("FLUX model downloaded!")
    
    if not os.path.exists("checkpoints/omnitry_lora"):
        print("Downloading OmniTry LoRA...")
        snapshot_download(
            'Kunbyte/OmniTry',
            local_dir='checkpoints/omnitry_lora'
        )
        print("LoRA downloaded!")

def load_model():
    global pipe
    if pipe is not None:
        return
    download_models()
    print("Loading model into GPU...")
    pipe = FluxFillPipeline.from_pretrained(
        "checkpoints/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.load_lora_weights("checkpoints/omnitry_lora/omnitry_v1_unified.safetensors")
    print("Model loaded successfully!")

def b64_to_pil(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def pil_to_b64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def handler(job):
    try:
        load_model()
        job_input = job["input"]

        person_image = b64_to_pil(job_input["person_image"])
        garment_image = b64_to_pil(job_input["garment_image"])
        prompt = job_input.get("prompt", "a photo of a person wearing the garment")
        num_steps = int(job_input.get("num_inference_steps", 30))
        guidance_scale = float(job_input.get("guidance_scale", 30.0))
        seed = job_input.get("seed", 42)

        generator = torch.Generator("cuda").manual_seed(seed)

        result = pipe(
            image=person_image,
            mask_image=garment_image,
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        return {"output_image": pil_to_b64(result), "status": "success"}
    
    except Exception as e:
        print(f"Handler error: {e}")
        return {"error": str(e), "status": "failed"}

print("Registering handler with RunPod...")
runpod.serverless.start({"handler": handler})