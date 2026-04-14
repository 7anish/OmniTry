import runpod
import torch
import base64
import io
import os
import shutil
import math
import random
import numpy as np
import torchvision.transforms as T

print("Starting handler...")

try:
    from PIL import Image
    print("PIL imported")
    from huggingface_hub import snapshot_download
    print("huggingface_hub imported")
    from peft import LoraConfig
    import peft
    from safetensors import safe_open
    from omnitry.models.transformer_flux import FluxTransformer2DModel
    from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline
    print("OmniTry pipeline imported")
except Exception as e:
    print(f"Import error: {e}")
    raise

pipe = None
transformer_model = None

VOLUME_PATH = os.environ.get("MODEL_DIR", "/runpod-volume/checkpoints")
FLUX_PATH = f"{VOLUME_PATH}/FLUX.1-Fill-dev"
LORA_PATH = f"{VOLUME_PATH}/omnitry_lora"

OBJECT_MAP = {
    'top clothes':    'replacing the top cloth',
    'bottom clothes': 'replacing the bottom cloth',
    'dress':          'replacing the dress',
    'shoe':           'replacing the shoe',
    'earrings':       'trying on earrings',
    'bracelet':       'trying on bracelet',
    'necklace':       'trying on necklace',
    'ring':           'trying on ring',
    'sunglasses':     'trying on sunglasses',
    'glasses':        'trying on glasses',
    'belt':           'trying on belt',
    'bag':            'trying on bag',
    'hat':            'trying on hat',
    'tie':            'trying on tie',
    'bow tie':        'trying on bow tie',
}

LORA_RANK  = 16
LORA_ALPHA = 16
device      = torch.device('cuda:0')
weight_dtype = torch.bfloat16


def download_models():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set!")
    print(f"HF_TOKEN found: {hf_token[:8]}...")

    os.makedirs(FLUX_PATH, exist_ok=True)
    os.makedirs(LORA_PATH, exist_ok=True)

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

    if not os.path.exists(f"{LORA_PATH}/omnitry_v1_unified.safetensors"):
        print("LoRA missing or incomplete, downloading...")
        if os.path.exists(LORA_PATH):
            shutil.rmtree(LORA_PATH)
        os.makedirs(LORA_PATH, exist_ok=True)
        snapshot_download('Kunbyte/OmniTry', local_dir=LORA_PATH)
        print("LoRA downloaded!")
    else:
        print("LoRA already exists, skipping download.")


def create_hacked_forward(module):
    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
            lora_A   = self.lora_A[active_adapter]
            lora_B   = self.lora_B[active_adapter]
            dropout  = self.lora_dropout[active_adapter]
            scaling  = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x))) * scaling
        return result

    def hacked_lora_forward(self, x, *args, **kwargs):
        return torch.cat((
            lora_forward(self, 'vtryon_lora',  x[:1], *args, **kwargs),
            lora_forward(self, 'garment_lora', x[1:], *args, **kwargs),
        ), dim=0)

    return hacked_lora_forward.__get__(module, type(module))


def load_model():
    global pipe, transformer_model
    if pipe is not None:
        return

    download_models()

    print("Loading transformer...")
    transformer_model = FluxTransformer2DModel.from_pretrained(
        f'{FLUX_PATH}/transformer'
    ).requires_grad_(False).to(dtype=weight_dtype)

    print("Loading pipeline...")
    pipe = FluxFillPipeline.from_pretrained(
        FLUX_PATH,
        transformer=transformer_model.eval(),
        torch_dtype=weight_dtype
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    # Insert dual LoRA adapters
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        init_lora_weights="gaussian",
        target_modules=[
            'x_embedder',
            'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0',
            'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out',
            'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2',
            'norm1_context.linear', 'norm1.linear', 'norm.linear', 'proj_mlp', 'proj_out'
        ]
    )
    transformer_model.add_adapter(lora_config, adapter_name='vtryon_lora')
    transformer_model.add_adapter(lora_config, adapter_name='garment_lora')

    lora_file = f"{LORA_PATH}/omnitry_v1_unified.safetensors"
    with safe_open(lora_file, framework="pt") as f:
        lora_weights = {k: f.get_tensor(k) for k in f.keys()}
    transformer_model.load_state_dict(lora_weights, strict=False)

    # Hack LoRA forward for dual-stream (person + garment) processing
    for n, m in transformer_model.named_modules():
        if isinstance(m, peft.tuners.lora.layer.Linear):
            m.forward = create_hacked_forward(m)

    print("Model loaded successfully!")


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        person_image  = b64_to_pil(job_input["person_image"])
        garment_image = b64_to_pil(job_input["garment_image"])
        object_class  = job_input.get("object_class", "dress")
        steps         = int(job_input.get("num_inference_steps", 20))
        guidance_scale = float(job_input.get("guidance_scale", 30.0))
        seed          = int(job_input.get("seed", -1))

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        seed_everything(seed)

        prompt = OBJECT_MAP.get(object_class, f"trying on {object_class}")
        print(f"object_class={object_class}, prompt={prompt}, steps={steps}, seed={seed}")

        # Resize person image to fit within 1024x1024 keeping aspect ratio
        max_area = 1024 * 1024
        oW, oH = person_image.width, person_image.height
        ratio = math.sqrt(max_area / (oW * oH))
        ratio = min(1, ratio)
        tW = int(oW * ratio) // 16 * 16
        tH = int(oH * ratio) // 16 * 16

        transform_person = T.Compose([T.Resize((tH, tW)), T.ToTensor()])
        person_tensor = transform_person(person_image)

        # Resize garment and center-pad to same dimensions as person
        g_ratio = min(tW / garment_image.width, tH / garment_image.height)
        transform_garment = T.Compose([
            T.Resize((int(garment_image.height * g_ratio), int(garment_image.width * g_ratio))),
            T.ToTensor()
        ])
        garment_padded = torch.ones_like(person_tensor)
        garment_tensor = transform_garment(garment_image)
        new_h, new_w   = garment_tensor.shape[1], garment_tensor.shape[2]
        min_x = (tW - new_w) // 2
        min_y = (tH - new_h) // 2
        garment_padded[:, min_y:min_y + new_h, min_x:min_x + new_w] = garment_tensor

        # Stack into dual-stream batch: [person, garment]
        img_cond = torch.stack([person_tensor, garment_padded]).to(dtype=weight_dtype, device=device)
        mask     = torch.zeros_like(img_cond).to(img_cond)

        with torch.no_grad():
            result = pipe(
                prompt=[prompt] * 2,
                height=tH,
                width=tW,
                img_cond=img_cond,
                mask=mask,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator(device).manual_seed(seed),
            ).images[0]

        print("Inference done!")
        return {"output_image": pil_to_b64(result), "status": "success"}

    except Exception as e:
        import traceback
        print(f"Handler error: {e}")
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


print("Registering handler with RunPod...")
runpod.serverless.start({"handler": handler})