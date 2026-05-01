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

# Install base requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install RunPod and HuggingFace dependencies
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    protobuf \
    "numpy<2"

# ✅ NEW: Install performance optimization libraries
# Flash Attention 2 for 30-40% speedup
RUN pip install --no-cache-dir packaging wheel
RUN pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation || \
    echo "⚠️ Flash Attention installation failed, will use xFormers fallback"

# ✅ NEW: Install xFormers as fallback for memory-efficient attention
RUN pip install --no-cache-dir xformers==0.0.23.post1 || \
    echo "⚠️ xFormers installation failed, will use default attention"

# ✅ NEW: Ensure we have PyTorch 2.0+ for torch.compile
# The base image has 2.2.0, so this is just a verification
RUN python -c "import torch; assert torch.__version__ >= '2.0.0', 'PyTorch 2.0+ required for torch.compile'" && \
    echo "✓ PyTorch $(python -c 'import torch; print(torch.__version__)') detected"

# ✅ NEW: Verify CUDA and GPU setup
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# ✅ NEW: Pre-compile some PyTorch operations for faster startup
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_HOME=/usr/local/cuda

# Copy optimized handler
COPY handler.py .

# ✅ NEW: Set environment variables for optimization
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1

# ✅ NEW: Enable TF32 for faster computation on Ampere+ GPUs
ENV NVIDIA_TF32_OVERRIDE=1

# ✅ OPTIONAL: Set model directory (can be overridden at runtime)
ENV MODEL_DIR=/runpod-volume/checkpoints

# Health check to ensure dependencies are installed
RUN python -c "import torch, transformers, diffusers, peft; print('✓ All core dependencies installed')"

CMD ["python", "-u", "handler.py"]