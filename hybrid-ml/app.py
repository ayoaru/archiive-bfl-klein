import modal
import io
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Define the Modal image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "cmake")
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
        # Face swap (Stage 3): restore the real face onto the FLUX output.
        "insightface",
        "opencv-python-headless",
        "onnx",
    )
)

# Define the Modal app
app = modal.App("archiive-ml", image=image)

# Separate volumes for each model's weights
fashn_volume = modal.Volume.from_name("archiive-fashn-weights", create_if_missing=True)
flux_volume = modal.Volume.from_name("archiive-model-weights", create_if_missing=True)
face_volume = modal.Volume.from_name("archiive-face-weights", create_if_missing=True)

FASHN_WEIGHTS_DIR = "/fashn-weights"
FLUX_MODEL_DIR = "/flux-weights"
FACE_WEIGHTS_DIR = "/face-weights"
FLUX_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"


def flux_dims(width, height, max_side=1216, multiple=16):
    """Scale (width, height) to FLUX-valid dimensions while preserving aspect
    ratio. The longest side is capped at `max_side` and both sides are rounded
    to a multiple of `multiple`. Preserving aspect ratio is what prevents the
    portrait person from being squished into a square (the 'fat'/compressed bug)."""
    scale = min(max_side / max(width, height), 1.0)
    w = max(multiple, round(width * scale / multiple) * multiple)
    h = max(multiple, round(height * scale / multiple) * multiple)
    return w, h

# Request schema
class TryOnRequest(BaseModel):
    person_image: str           # base64
    garment_image: str          # base64
    category: str               # "tops" | "bottoms" | "one-pieces"
    item_name: str
    item_brand: str
    item_category: str
    item_color: str
    garment_photo_type: Optional[str] = "auto"  # FASHN: "auto" | "model" | "flat-lay"
    num_timesteps: Optional[int] = 50       # FASHN steps
    guidance_scale: Optional[float] = 2.0   # FASHN guidance
    flux_steps: Optional[int] = 28          # FLUX steps
    seed: Optional[int] = None              # FLUX seed; None → random (for "regenerate")
    face_swap: Optional[bool] = True        # swap the real face back onto the FLUX output
    white_background: Optional[bool] = True # remove subject background → white backdrop
    debug: Optional[bool] = False           # also return raw Stage 1 + FLUX images


@app.cls(
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        FASHN_WEIGHTS_DIR: fashn_volume,
        FLUX_MODEL_DIR: flux_volume,
        FACE_WEIGHTS_DIR: face_volume,
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

        # ============= Load InsightFace (Stage 3 — face swap) =============
        # buffalo_l = face detection + embedding; inswapper_128 = the swap model.
        # Cache both in the face volume so cold starts don't re-download.
        os.environ["INSIGHTFACE_HOME"] = FACE_WEIGHTS_DIR
        import insightface
        from insightface.app import FaceAnalysis

        self.face_app = FaceAnalysis(
            name="buffalo_l", root=FACE_WEIGHTS_DIR,
            providers=["CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        swapper_path = f"{FACE_WEIGHTS_DIR}/models/inswapper_128.onnx"
        if not os.path.exists(swapper_path):
            # insightface's own auto-download (GitHub releases v0.7) is dead, so
            # fetch the weights from a HF mirror instead. To use your own file
            # instead, drop it at swapper_path in the archiive-face-weights volume.
            print("Fetching inswapper_128.onnx from HF mirror ...")
            os.makedirs(os.path.dirname(swapper_path), exist_ok=True)
            from huggingface_hub import hf_hub_download
            import shutil
            downloaded = hf_hub_download(
                repo_id="ezioruan/inswapper_128.onnx",
                filename="inswapper_128.onnx",
                token=os.environ.get("HF_TOKEN"),
            )
            shutil.copy(downloaded, swapper_path)
            face_volume.commit()
            print("inswapper cached.")
        else:
            print("Loading inswapper from cache...")

        self.face_swapper = insightface.model_zoo.get_model(
            swapper_path, providers=["CPUExecutionProvider"]
        )
        print("InsightFace loaded successfully.")

    def swap_face(self, target_image, source_image):
        """Swap the face from `source_image` (the real person) onto `target_image`
        (the FLUX output), aligned by facial landmarks. Returns (PIL image, status).
        On any missing-face case the target is returned unchanged so we never ship
        a broken frame."""
        import numpy as np
        import cv2
        from PIL import Image

        target_bgr = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
        source_bgr = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)

        source_faces = self.face_app.get(source_bgr)
        if not source_faces:
            return target_image, "skipped: no face in person image"
        target_faces = self.face_app.get(target_bgr)
        if not target_faces:
            return target_image, "skipped: no face in FLUX output"

        # Pick the largest face in each (most likely the subject).
        def largest(faces):
            return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        source_face = largest(source_faces)
        target_face = largest(target_faces)

        result_bgr = self.face_swapper.get(target_bgr, target_face, source_face, paste_back=True)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb), "swapped"

    def remove_background(self, image, label="garment"):
        """Remove the background using rembg and paste onto a white backdrop."""
        from rembg import remove
        from PIL import Image

        print(f"Removing {label} background...")
        result = remove(image)

        # Paste onto white background
        white_bg = Image.new("RGB", result.size, (255, 255, 255))
        white_bg.paste(result, mask=result.split()[3])

        print(f"{label.capitalize()} background removed.")
        return white_bg

    @modal.method()
    def run_inference(self, request):
        import torch
        import random
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
        garment_photo_type = request.get("garment_photo_type", "auto")
        num_timesteps = request.get("num_timesteps", 50)
        guidance_scale = request.get("guidance_scale", 2.0)
        flux_steps = request.get("flux_steps", 28)
        face_swap = request.get("face_swap", True)
        white_background = request.get("white_background", True)
        debug = request.get("debug", False)

        # Seed controls which FLUX rendering you get. None → random, so the
        # front-end "regenerate" button can re-call with no seed for a new result.
        seed = request.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        print(f"Using FLUX seed: {seed}")

        # Preprocess — remove garment background
        garment_clean = self.remove_background(garment_image)

        # ============= Stage 1 — FASHN VTON =============
        # Produces the fitted try-on that conditions FLUX (not shown to the user).
        print("Stage 1: FASHN VTON — fitting garment to person...")

        # Share the seed with FASHN so "regenerate" varies the fit too, and a
        # pinned seed reproduces the whole pipeline (FASHN defaults to 42 otherwise).
        stage1_result = self.fashn_pipeline(
            person_image=person_image,
            garment_image=garment_clean,
            category=category,
            garment_photo_type=garment_photo_type,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
        ).images[0]

        print("Stage 1 complete.")

        # ============= Stage 2 — FLUX.2 [klein] garment detail enhancement =============
        # This is the base result the user wants. Generate at the person's true
        # aspect ratio — forcing a square canvas stretched the body ('fat'/compressed).
        print("Stage 2: FLUX.2 [klein] — enhancing garment detail...")

        base_w, base_h = stage1_result.size
        gen_w, gen_h = flux_dims(base_w, base_h)

        stage2_prompt = (
            f"A person wearing a {item_color} {item_name} by {item_brand}. "
            f"Keep the person's face, body, proportions, pose and the background "
            f"exactly as in the first reference image — do not change the body shape or size. "
            f"Only refine the {item_category}: reproduce exactly the same details, patches, "
            f"graphics, colors, seams and fabric texture as the garment reference, "
            f"sharp and high-resolution. "
            f"Fashion photography, clean neutral background, high quality, photorealistic."
        )

        stage2_refs = [
            stage1_result.resize((gen_w, gen_h), Image.LANCZOS),  # 1st - person + layout (correct proportions)
            garment_clean,                                        # 2nd - clean garment (detail source)
            garment_clean,                                        # 3rd - repeated for stronger conditioning
        ]

        flux_out = self.flux_pipeline(
            prompt=stage2_prompt,
            image=stage2_refs,
            height=gen_h,
            width=gen_w,
            num_inference_steps=flux_steps,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]

        print("Stage 2 (FLUX) generation complete.")

        # Work at the person's native resolution from here on.
        flux_result = flux_out.resize((base_w, base_h), Image.LANCZOS)

        # ============= Stage 3 — Face swap =============
        # FLUX gives the garment/texture we want but alters the face. We keep the
        # FLUX image whole (so no body-merge artifacts like the clipped shoulder /
        # texture seams) and swap only the face — the real one from the person photo,
        # landmark-aligned by the swap model.
        swap_status = "disabled"
        final_result = flux_result
        if face_swap:
            print("Stage 3: Face swap — restoring the person's identity...")
            try:
                final_result, swap_status = self.swap_face(flux_result, person_image)
            except Exception as e:
                print(f"Face swap failed: {e}")
                final_result, swap_status = flux_result, f"error: {e}"
            print(f"Face swap: {swap_status}")

        # Optionally drop the subject onto a clean white backdrop.
        if white_background:
            final_result = self.remove_background(final_result, label="subject")

        def encode_png(img):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        # Return the seed (for pin/regenerate) and the face-swap status.
        response = {"image": encode_png(final_result), "seed": seed, "face_swap": swap_status}

        # In debug mode, also return the raw Stage 1 (FASHN) and FLUX outputs.
        if debug:
            response["stage1_image"] = encode_png(stage1_result)
            response["flux_image"] = encode_png(flux_result)

        return response


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
        "garment_photo_type": request.garment_photo_type,
        "num_timesteps": request.num_timesteps,
        "guidance_scale": request.guidance_scale,
        "flux_steps": request.flux_steps,
        "seed": request.seed,
        "face_swap": request.face_swap,
        "white_background": request.white_background,
        "debug": request.debug,
    })
    return result
