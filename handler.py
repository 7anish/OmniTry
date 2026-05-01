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

VOLUME_PATH  = os.environ.get("MODEL_DIR", "/runpod-volume/checkpoints")
FLUX_PATH    = f"{VOLUME_PATH}/FLUX.1-Fill-dev"
LORA_PATH    = f"{VOLUME_PATH}/omnitry_lora"

OBJECT_MAP = {
    'top clothes':    'wearing the top garment naturally with proper fit and draping on the upper body',
    'bottom clothes': 'wearing the bottom garment with accurate fit around waist, hips and legs',
    'dress':          'wearing the dress with natural fabric flow, proper length and body-conforming fit',
    'shoe':           'wearing the shoes on feet with realistic placement, perspective and ground contact',
    'earrings':       'wearing earrings attached to ears with realistic metal reflection and proper hang',
    'bracelet':       'wearing bracelet around wrist with natural drape and jewelry details visible',
    'necklace':       'wearing necklace around neck with proper chain drape and pendant positioning',
    'ring':           'wearing ring on finger with accurate size, metal shine and gemstone details',
    'sunglasses':     'wearing sunglasses on face with proper nose bridge fit and temple alignment',
    'glasses':        'wearing eyeglasses on face with correct lens position and frame fit',
    'belt':           'wearing belt around waist with proper buckle position and leather texture',
    'bag':            'carrying bag naturally on shoulder or in hand with realistic weight distribution',
    'hat':            'wearing hat on head with proper fit, angle and shadowing on face',
    'tie':            'wearing tie with proper knot around collar and appropriate length',
    'bow tie':        'wearing bow tie centered at collar with symmetrical bow shape',
}

LORA_RANK    = 16
LORA_ALPHA   = 16
device       = torch.device('cuda:0')
weight_dtype = torch.bfloat16   # bfloat16 is safer than float16 for FLUX — avoids NaN


def download_models():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set!")
    print(f"HF_TOKEN found: {hf_token[:8]}...")

    os.makedirs(FLUX_PATH, exist_ok=True)
    os.makedirs(LORA_PATH, exist_ok=True)

    if not os.path.exists(f"{FLUX_PATH}/model_index.json"):
        print("Downloading FLUX model...")
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
        print("Downloading LoRA weights...")
        if os.path.exists(LORA_PATH):
            shutil.rmtree(LORA_PATH)
        os.makedirs(LORA_PATH, exist_ok=True)
        snapshot_download('Kunbyte/OmniTry', local_dir=LORA_PATH)
        print("LoRA downloaded!")
    else:
        print("LoRA already exists, skipping download.")


def create_hacked_forward(module):
    """
    Dual-stream LoRA forward pass — CRITICAL for OmniTry.
    First image in batch uses vtryon_lora (person stream).
    Second image uses garment_lora (garment stream).
    ⚠️ Do NOT use torch.compile — it breaks this custom forward.
    """
    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
            lora_A  = self.lora_A[active_adapter]
            lora_B  = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
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

    # Enable TF32 for faster matmul on Ampere+ GPUs (A100, RTX 3090+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("Loading transformer...")
    transformer_model = FluxTransformer2DModel.from_pretrained(
        f'{FLUX_PATH}/transformer'
    ).requires_grad_(False).to(dtype=weight_dtype, device=device)

    print("Loading pipeline...")
    pipe = FluxFillPipeline.from_pretrained(
        FLUX_PATH,
        transformer=transformer_model.eval(),
        torch_dtype=weight_dtype
    )

    # Keep full pipeline on GPU — no CPU offloading (much faster)
    pipe.to(device)

    # VAE slicing: memory efficient without speed penalty
    pipe.vae.enable_slicing()

    # ⚠️ xFormers is intentionally DISABLED
    # OmniTry uses a custom FluxAttnProcessor2_0 that passes image_rotary_emb
    # and lens kwargs through cross_attention_kwargs. xFormers replaces this
    # processor with XFormersAttnProcessor which silently ignores those kwargs,
    # causing a tensor shape mismatch (512 vs 4042) and crashing inference.
    # PyTorch 2.2 scaled_dot_product_attention (SDPA) is used instead —
    # it is nearly as fast and fully compatible with OmniTry's attention.
    print("✓ Using PyTorch SDPA attention (xFormers disabled — incompatible with OmniTry)")

    # ⚠️ torch.compile is intentionally SKIPPED
    # The hacked dual-stream LoRA forward is a dynamic Python closure
    # torch.compile with fullgraph=True will crash on it.
    # Speedup from xFormers + TF32 + GPU-only pipeline is sufficient.

    print("Inserting dual LoRA adapters...")
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

    # Hack LoRA forward for dual-stream
    for n, m in transformer_model.named_modules():
        if isinstance(m, peft.tuners.lora.layer.Linear):
            m.forward = create_hacked_forward(m)

    print("Running warmup inference...")
    _warmup_model()
    print("✓ Model fully loaded and ready!")


def _warmup_model():
    """
    1-step warmup to prime CUDA kernels and cudnn benchmarking.
    Must use stack([person, garment]) shape — same as real inference.
    512 is too small for FLUX's positional embeddings; use 768x1024.
    """
    try:
        H, W = 768, 1024
        dummy_person  = torch.zeros(3, H, W, device=device, dtype=weight_dtype)
        dummy_garment = torch.zeros(3, H, W, device=device, dtype=weight_dtype)
        dummy_cond = torch.stack([dummy_person, dummy_garment])   # shape [2, 3, H, W]
        dummy_mask = torch.zeros_like(dummy_cond)
        with torch.no_grad():
            _ = pipe(
                prompt=["warmup"] * 2,
                height=H,
                width=W,
                img_cond=dummy_cond,
                mask=dummy_mask,
                num_inference_steps=1,
                guidance_scale=1.0,
            )
        print("✓ Warmup complete!")
    except Exception as e:
        print(f"⚠ Warmup failed (non-fatal): {e}")


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
        steps          = int(job_input.get("num_inference_steps", 20))
        guidance_scale = float(job_input.get("guidance_scale", 30.0))
        seed           = int(job_input.get("seed", -1))

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        seed_everything(seed)

        prompt = OBJECT_MAP.get(object_class, f"trying on {object_class}")
        print(f"Inference: class={object_class}, steps={steps}, seed={seed}")

        # Smart resolution — max 1024x1024, divisible by 16
        max_area = 1024 * 1024
        oW, oH = person_image.width, person_image.height
        ratio = min(1.0, math.sqrt(max_area / (oW * oH)))
        tW = int(oW * ratio) // 16 * 16
        tH = int(oH * ratio) // 16 * 16
        print(f"Resolution: {tW}x{tH} (original: {oW}x{oH})")

        # Resize person
        transform_person = T.Compose([T.Resize((tH, tW)), T.ToTensor()])
        person_tensor = transform_person(person_image)

        # Resize + center-pad garment onto white canvas
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

        # Stack person + garment for dual-stream processing
        img_cond = torch.stack([person_tensor, garment_padded]).to(
            dtype=weight_dtype, device=device, non_blocking=True
        )
        mask = torch.zeros_like(img_cond)  # Zero mask = no inpainting

        print("Running inference...")
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

        print("✓ Inference done!")
        return {
            "output_image": pil_to_b64(result),
            "status": "success",
            "metadata": {
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "resolution": f"{tW}x{tH}",
                "object_class": object_class
            }
        }

    except Exception as e:
        import traceback
        print(f"❌ Handler error: {e}")
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


print("Registering handler with RunPod...")
runpod.serverless.start({"handler": handler})