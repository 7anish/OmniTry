import runpod, base64, io
from PIL import Image

runpod.api_key = "..............."

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

endpoint = runpod.Endpoint("yb0llewa0pgex4")

result = endpoint.run_sync({
    "input": {
        "person_image": img_to_b64("person.jpeg"),
        "garment_image": img_to_b64("garment.jpeg"),
        "prompt": "a person wearing the garment, photorealistic",
        "num_inference_steps": 30,
        "seed": 42
    }
}, timeout=600)

# Print full result to see what came back
print("Full result:", result)

# Only decode if successful
if result.get("status") == "success":
    img = Image.open(io.BytesIO(base64.b64decode(result["output_image"])))
    img.save("result.png")
    print("Done! Saved to result.png")
else:
    print("Error:", result.get("error", "Unknown error"))