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
    item_name: str
    item_brand: str
    item_category: str
    item_color: str
    face: str
    full_body: str
    item_front: str
    item_back: str
    model_front: str
    model_back: str
    # Person metadata
    height_cm: int
    weight_kg: int
    build: str  # slim, athletic, average, large
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    num_inference_steps: Optional[int] = 28


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
        num_inference_steps = request.get("num_inference_steps", 28)

        # Decode base64 reference images
        ref_images = []
        for img_b64 in reference_images:
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((768, 768))
            ref_images.append(img)

        # Run inference
        result = self.pipe(
            prompt=prompt,
            image=ref_images,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
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
    prompt = (
        f"The exact same person from the reference photos wearing a {request.item_color} {request.item_name} by {request.item_brand}. "
        f"Use the first and second reference images to preserve the person's face, skin tone, hair, and facial features exactly — identical likeness, same person. "
        f"The person is {request.height_cm}cm tall, weighs {request.weight_kg}kg, and has a {request.build} build. "
        f"Use the third reference image to preserve the person's body proportions exactly — correct leg to torso ratio, natural human proportions from head to toe. "
        f"Use the fourth and fifth reference images to reproduce the {request.item_category} exactly, "
        f"preserving all details, patches, graphics, colors, and textures precisely. "
        f"Use the sixth and seventh reference images to understand exactly how the garment fits, drapes, "
        f"and sits on a person's body, including sleeve length, shoulder fit, and overall silhouette. "
        f"Standing upright, neutral pose, arms relaxed at sides, facing forward, full body visible from head to toe. "
        f"Fashion photography, clean neutral background, high quality, photorealistic."
    )

    reference_images = [
        request.face,             # 1st - face reference
        request.face,             # 2nd - face reference repeated for stronger conditioning
        request.full_body,        # 3rd - body proportions
        request.item_front,       # 4th - item front
        request.item_back,        # 5th - item back
        request.model_front,      # 6th - item fit front
        request.model_back,       # 7th - item fit back
    ]

    result = await FluxInference().run_inference.remote.aio({
        "prompt": prompt,
        "reference_images": reference_images,
        "height": request.height,
        "width": request.width,
        "num_inference_steps": request.num_inference_steps,
    })
    return result