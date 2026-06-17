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
        "rembg",
        "onnxruntime",
        "numpy",
    )
)

# Define the Modal app
app = modal.App("archiive-ml", image=image)

# Separate volumes for each model's weights
fashn_volume = modal.Volume.from_name("archiive-fashn-weights", create_if_missing=True)
flux_volume = modal.Volume.from_name("archiive-model-weights", create_if_missing=True)

FASHN_WEIGHTS_DIR = "/fashn-weights"
FLUX_MODEL_DIR = "/flux-weights"
FLUX_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

# Request schema
class TryOnRequest(BaseModel):
    person_image: str           # base64
    garment_image: str          # base64
    category: str               # "tops" | "bottoms" | "one-pieces"
    item_name: str
    item_brand: str
    item_category: str
    item_color: str
    num_timesteps: Optional[int] = 50       # FASHN steps
    guidance_scale: Optional[float] = 2.0   # FASHN guidance
    flux_steps: Optional[int] = 28          # FLUX steps


@app.cls(
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        FASHN_WEIGHTS_DIR: fashn_volume,
        FLUX_MODEL_DIR: flux_volume,
    },
    timeout=300,
)
class HybridInference:

    @modal.enter()
    def load_models(self):
        import sys
        import os
        import torch
        from diffusers import Flux2KleinPipeline
        from huggingface_hub import snapshot_download

        # ============= Load FASHN VTON =============
        sys.path.insert(0, "/root/fashn-vton-1.5")
        from fashn_vton import TryOnPipeline

        if not os.path.exists(f"{FASHN_WEIGHTS_DIR}/model.safetensors"):
            print("Downloading FASHN VTON weights...")
            import subprocess
            subprocess.run(
                ["python", "/root/fashn-vton-1.5/scripts/download_weights.py",
                 "--weights-dir", FASHN_WEIGHTS_DIR],
                check=True
            )
            fashn_volume.commit()
            print("FASHN weights downloaded and cached.")
        else:
            print("Loading FASHN VTON from cache...")

        self.fashn_pipeline = TryOnPipeline(weights_dir=FASHN_WEIGHTS_DIR)
        print("FASHN VTON loaded successfully.")

        # ============= Load FLUX.2 [klein] =============
        if not os.path.exists(f"{FLUX_MODEL_DIR}/model_index.json"):
            print("Downloading FLUX.2 [klein] weights...")
            snapshot_download(
                repo_id=FLUX_MODEL_ID,
                local_dir=FLUX_MODEL_DIR,
                token=os.environ["HF_TOKEN"],
            )
            flux_volume.commit()
            print("FLUX weights downloaded and cached.")
        else:
            print("Loading FLUX.2 [klein] from cache...")

        self.flux_pipeline = Flux2KleinPipeline.from_pretrained(
            FLUX_MODEL_DIR,
            torch_dtype=torch.bfloat16,
        )
        self.flux_pipeline.enable_model_cpu_offload()
        print("FLUX.2 [klein] loaded successfully.")

    def remove_background(self, image):
        """Remove background from garment image using rembg."""
        from rembg import remove
        from PIL import Image

        print("Removing garment background...")
        result = remove(image)

        # Paste onto white background
        white_bg = Image.new("RGB", result.size, (255, 255, 255))
        white_bg.paste(result, mask=result.split()[3])

        print("Background removed.")
        return white_bg

    @modal.method()
    def run_inference(self, request):
        import torch
        from PIL import Image

        def decode_image(b64_str, size=None):
            img_bytes = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if size:
                img = img.resize(size, Image.LANCZOS)
            return img

        person_image = decode_image(request["person_image"])
        garment_image = decode_image(request["garment_image"])
        category = request["category"]
        item_name = request["item_name"]
        item_brand = request["item_brand"]
        item_category = request["item_category"]
        item_color = request["item_color"]
        num_timesteps = request.get("num_timesteps", 50)
        guidance_scale = request.get("guidance_scale", 2.0)
        flux_steps = request.get("flux_steps", 28)

        # Preprocess — remove garment background
        garment_clean = self.remove_background(garment_image)

        # ============= Stage 1 — FASHN VTON =============
        print("Stage 1: FASHN VTON — fitting garment to person...")

        stage1_result = self.fashn_pipeline(
            person_image=person_image,
            garment_image=garment_clean,
            category=category,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
        ).images[0]

        print("Stage 1 complete.")

        # ============= Stage 2 — FLUX.2 [klein] =============
        print("Stage 2: FLUX.2 [klein] — enhancing garment detail fidelity...")

        stage2_prompt = (
            f"A person wearing a {item_color} {item_name} by {item_brand}. "
            f"The {item_category} has exactly the same details, patches, graphics, colors, "
            f"and textures as shown in the reference garment image. "
            f"Enhance and sharpen all garment details to exactly match the reference. "
            f"Preserve the person's identity, face, body, and pose from the first reference image exactly. "
            f"Fashion photography, clean neutral background, high quality, photorealistic."
        )

        stage2_refs = [
            stage1_result.resize((768, 768), Image.LANCZOS),  # 1st - FASHN output
            garment_clean.resize((768, 768), Image.LANCZOS),  # 2nd - clean garment
            garment_clean.resize((768, 768), Image.LANCZOS),  # 3rd - repeated for stronger conditioning
        ]

        stage2_result = self.flux_pipeline(
            prompt=stage2_prompt,
            image=stage2_refs,
            height=1024,
            width=1024,
            num_inference_steps=flux_steps,
            generator=torch.Generator().manual_seed(42),
        ).images[0]

        print("Stage 2 complete.")

        # Encode final output as base64
        buffer = io.BytesIO()
        stage2_result.save(buffer, format="PNG")
        output_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image": output_b64}


# FastAPI web endpoint
web_app = FastAPI()

@app.function(
    secrets=[modal.Secret.from_name("huggingface")],
    image=image,
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    return web_app

@web_app.post("/try-on")
async def try_on(request: TryOnRequest):
    result = await HybridInference().run_inference.remote.aio({
        "person_image": request.person_image,
        "garment_image": request.garment_image,
        "category": request.category,
        "item_name": request.item_name,
        "item_brand": request.item_brand,
        "item_category": request.item_category,
        "item_color": request.item_color,
        "num_timesteps": request.num_timesteps,
        "guidance_scale": request.guidance_scale,
        "flux_steps": request.flux_steps,
    })
    return result