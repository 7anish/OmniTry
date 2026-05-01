FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone OmniTry repository
RUN git clone --recurse-submodules https://github.com/Kunbyte-AI/OmniTry.git .

# ✅ FIX 1: Pin torch + torchvision to MATCHING versions BEFORE requirements.txt
# This prevents requirements.txt from overwriting with an incompatible torchvision
RUN pip install --no-cache-dir --force-reinstall \
    "torch==2.2.0" \
    "torchvision==0.17.0" \
    "torchaudio==2.2.0" \
    --index-url https://download.pytorch.org/whl/cu121

# ✅ FIX 2: Install requirements.txt but ignore torch/torchvision lines to avoid overwrite
RUN grep -v -i "torch" requirements.txt > /tmp/requirements_notorch.txt && \
    pip install --no-cache-dir -r /tmp/requirements_notorch.txt

# Install RunPod and HuggingFace dependencies
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    protobuf \
    "numpy<2"

# Install performance optimization libraries
RUN pip install --no-cache-dir packaging wheel

# Flash Attention 2 for 30-40% speedup
RUN pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation || \
    echo "⚠️ Flash Attention installation failed, will use xFormers fallback"

# xFormers as fallback for memory-efficient attention
# ✅ FIX 3: Use xformers version matched to torch 2.2.0
RUN pip install --no-cache-dir "xformers==0.0.24" || \
    echo "⚠️ xFormers installation failed, will use default attention"

# Verify torch + torchvision are compatible
RUN python -c "\
import torch, torchvision; \
print(f'torch: {torch.__version__}'); \
print(f'torchvision: {torchvision.__version__}'); \
import torchvision.transforms as T; \
print('✓ torchvision.transforms imported OK'); \
from torchvision.ops import nms; \
print('✓ torchvision::nms operator OK'); \
print(f'CUDA: {torch.cuda.is_available()}') \
"

# Set environment variables for optimization
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_HOME=/usr/local/cuda
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV NVIDIA_TF32_OVERRIDE=1
ENV MODEL_DIR=/runpod-volume/checkpoints

# Copy optimized handler
COPY handler.py .

# Final health check
RUN python -c "import torch, transformers, diffusers, peft, torchvision; print('✓ All core dependencies installed and compatible')"

CMD ["python", "-u", "handler.py"]