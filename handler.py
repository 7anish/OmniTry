import runpod
import torch
import base64
import io
import os
import shutil

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

# Use network volume if available, otherwise fall back to local
VOLUME_PATH = os.environ.get("MODEL_DIR", "/runpod-volume/checkpoints")
FLUX_PATH = f"{VOLUME_PATH}/FLUX.1-Fill-dev"
LORA_PATH = f"{VOLUME_PATH}/omnitry_lora"


def download_models():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set! Please add it in RunPod endpoint settings.")
    print(f"HF_TOKEN found: {hf_token[:8]}...")

    os.makedirs(FLUX_PATH, exist_ok=True)
    os.makedirs(LORA_PATH, exist_ok=True)

    # Check if FLUX model is fully downloaded
    if not os.path.exists(f"{FLUX_PATH}/model_index.json"):
        print("FLUX model missing or incomplete, downloading...")
        if os.path.exists(FLUX_PATH):
            shutil.rmtree(FLUX_PATH)
        os.makedirs(FLUX_PATH, exist_ok=True)
        snapshot_download(
            'black-forest-labs/FLUX.1-Fill-dev',
            local_dir=FLUX_PATH,
            token=hf_token
        )
        print("FLUX model downloaded!")
    else:
        print("FLUX model already exists, skipping download.")

    # Check if LoRA is fully downloaded
    if not os.path.exists(f"{LORA_PATH}/omnitry_v1_unified.safetensors"):
        print("LoRA missing or incomplete, downloading...")
        if os.path.exists(LORA_PATH):
            shutil.rmtree(LORA_PATH)
        os.makedirs(LORA_PATH, exist_ok=True)
        snapshot_download(
            'Kunbyte/OmniTry',
            local_dir=LORA_PATH
        )
        print("LoRA downloaded!")
    else:
        print("LoRA already exists, skipping download.")


def load_model():
    global pipe
    if pipe is not None:
        return
    download_models()
    print("Loading model into GPU...")
    pipe = FluxFillPipeline.from_pretrained(
        FLUX_PATH,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.load_lora_weights(f"{LORA_PATH}/omnitry_v1_unified.safetensors")
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

        if "person_image" not in job_input:
            return {"error": "person_image is required", "status": "failed"}
        if "garment_image" not in job_input:
            return {"error": "garment_image is required", "status": "failed"}

        person_image = b64_to_pil(job_input["person_image"])
        garment_image = b64_to_pil(job_input["garment_image"])
        prompt = job_input.get("prompt", "a photo of a person wearing the garment")
        num_steps = int(job_input.get("num_inference_steps", 30))
        guidance_scale = float(job_input.get("guidance_scale", 30.0))
        seed = job_input.get("seed", 42)

        print(f"Running inference: steps={num_steps}, guidance={guidance_scale}, seed={seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        result = pipe(
            image=person_image,
            mask_image=garment_image,
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        print("Inference done!")
        return {"output_image": pil_to_b64(result), "status": "success"}

    except Exception as e:
        print(f"Handler error: {e}")
        return {"error": str(e), "status": "failed"}


print("Registering handler with RunPod...")
runpod.serverless.start({"handler": handler})