import modal
import io
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Define the Modal image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/fashn-AI/fashn-vton-1.5.git /root/fashn-vton-1.5 && "
        "cd /root/fashn-vton-1.5 && pip install -e ."
    )
    .pip_install(
        "fastapi",
        "pydantic",
        "python-multipart",
        "Pillow",
        "rembg",
        "onnxruntime",
    )
)

# Define the Modal app
app = modal.App("archiive-ml", image=image)

# Cache model weights in a Modal volume
volume = modal.Volume.from_name("archiive-fashn-weights", create_if_missing=True)
WEIGHTS_DIR = "/model-weights"

# Request schema
class TryOnRequest(BaseModel):
    person_image: str       # base64
    garment_image: str      # base64
    category: str           # "tops" | "bottoms" | "one-pieces"
    num_timesteps: Optional[int] = 50
    guidance_scale: Optional[float] = 2.0


@app.cls(
    gpu="A100-80GB",
    volumes={WEIGHTS_DIR: volume},
    timeout=300,
)
class FashnInference:

    @modal.enter()
    def load_model(self):
        import sys
        import os
        sys.path.insert(0, "/root/fashn-vton-1.5")
        from fashn_vton import TryOnPipeline

        # Download weights if not already cached
        if not os.path.exists(f"{WEIGHTS_DIR}/model.safetensors"):
            print("Downloading model weights...")
            import subprocess
            subprocess.run(
                ["python", "/root/fashn-vton-1.5/scripts/download_weights.py",
                 "--weights-dir", WEIGHTS_DIR],
                check=True
            )
            volume.commit()
            print("Weights downloaded and cached.")
        else:
            print("Loading model from cache...")

        self.pipeline = TryOnPipeline(weights_dir=WEIGHTS_DIR)
        print("Model loaded successfully.")

    def remove_background(self, image):
        """Remove background from garment image using rembg."""
        from rembg import remove
        from PIL import Image
        import numpy as np

        print("Removing garment background...")

        # Remove background — returns RGBA image
        result = remove(image)

        # Paste onto white background for FASHN VTON
        white_bg = Image.new("RGB", result.size, (255, 255, 255))
        white_bg.paste(result, mask=result.split()[3])  # use alpha as mask

        print("Background removed.")
        return white_bg

    @modal.method()
    def run_inference(self, request):
        from PIL import Image

        def decode_image(b64_str):
            img_bytes = base64.b64decode(b64_str)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        person_image = decode_image(request["person_image"])
        garment_image = decode_image(request["garment_image"])
        category = request["category"]
        num_timesteps = request.get("num_timesteps", 50)
        guidance_scale = request.get("guidance_scale", 2.0)

        # Remove background from garment image
        garment_image = self.remove_background(garment_image)

        print(f"Running inference — category: {category}")

        result = self.pipeline(
            person_image=person_image,
            garment_image=garment_image,
            category=category,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
        )

        # Encode output as base64
        buffer = io.BytesIO()
        result.images[0].save(buffer, format="PNG")
        output_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        print("Inference complete.")
        return {"image": output_b64}


# FastAPI web endpoint
web_app = FastAPI()

@app.function(
    image=image,
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    return web_app

@web_app.post("/try-on")
async def try_on(request: TryOnRequest):
    result = await FashnInference().run_inference.remote.aio({
        "person_image": request.person_image,
        "garment_image": request.garment_image,
        "category": request.category,
        "num_timesteps": request.num_timesteps,
        "guidance_scale": request.guidance_scale,
    })
    return result