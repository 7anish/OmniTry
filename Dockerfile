FROM runpod/base:0.6.2-cuda12.1.0

RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --recurse-submodules https://github.com/Kunbyte-AI/OmniTry.git .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod huggingface_hub
RUN pip install flash-attn==2.6.3 --no-build-isolation

COPY handler.py .

CMD ["python", "-u", "handler.py"]