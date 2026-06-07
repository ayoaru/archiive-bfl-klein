import modal
import io
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Define the Modal image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers @ git+https://github.com/huggingface/diffusers.git",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "Pillow",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "fastapi",
        "pydantic",
        "python-multipart",
    )
)

# Define the Modal app
app = modal.App("archiive-ml", image=image)

# Cache the model weights in a Modal volume so they don't re-download every run
volume = modal.Volume.from_name("archiive-model-weights", create_if_missing=True)
MODEL_DIR = "/model-weights"
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

# Request schema
class TryOnRequest(BaseModel):
    prompt: str
    reference_images: list[str]  # base64 encoded images
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    num_inference_steps: Optional[int] = 4
    guidance_scale: Optional[float] = 1.0


@app.cls(
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={MODEL_DIR: volume},
    timeout=300,
)
class FluxInference:

    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import Flux2KleinPipeline
        from huggingface_hub import snapshot_download
        import os

        # Download model weights to volume if not already cached
        if not os.path.exists(f"{MODEL_DIR}/model_index.json"):
            print("Downloading model weights...")
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=MODEL_DIR,
                token=os.environ["HF_TOKEN"],
            )
            volume.commit()
            print("Model weights downloaded and cached.")
        else:
            print("Loading model from cache...")

        # Load the pipeline
        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.enable_model_cpu_offload()
        print("Model loaded successfully.")

    @modal.method()
    def run_inference(self, request):
        import torch
        from PIL import Image

        # Extract fields from dict
        prompt = request["prompt"]
        reference_images = request["reference_images"]
        height = request.get("height", 1024)
        width = request.get("width", 1024)
        num_inference_steps = request.get("num_inference_steps", 4)
        guidance_scale = request.get("guidance_scale", 1.0)

        # Decode base64 reference images
        ref_images = []
        for img_b64 in reference_images:
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            ref_images.append(img)

        # Run inference
        result = self.pipe(
            prompt=prompt,
            image=ref_images,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(42),
        ).images[0]

        # Encode output image as base64
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        output_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image": output_b64}


# FastAPI web endpoint
web_app = FastAPI()

@app.function(
    secrets=[modal.Secret.from_name("huggingface")],
    image=image,
)
@modal.asgi_app()
def fastapi_app():
    return web_app

@web_app.post("/try-on")
async def try_on(request: TryOnRequest):
    result = await FluxInference().run_inference.remote.aio(request.dict())
    return result