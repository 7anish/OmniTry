FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Fix NumPy conflict first
RUN pip install --no-cache-dir "numpy<2"

WORKDIR /app
RUN git clone --recurse-submodules https://github.com/Kunbyte-AI/OmniTry.git .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod huggingface_hub

COPY handler.py .

CMD ["python", "-u", "handler.py"]
