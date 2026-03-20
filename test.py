import runpod, base64, io
from PIL import Image

runpod.api_key = "ccccccccccccccc"

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

endpoint = runpod.Endpoint("62c6tnqko9x1u5")

result = endpoint.run_sync({
    "input": {
        "person_image": img_to_b64("person.jpeg"),
        "garment_image": img_to_b64("garment.jpeg"),
        "prompt": "a person wearing the garment, photorealistic",
        "num_inference_steps": 30,
        "seed": 42
    }
}, timeout=600)  # 10 min timeout for first run

img = Image.open(io.BytesIO(base64.b64decode(result["output_image"])))
img.save("result.png")
print("Done! Saved to result.png")