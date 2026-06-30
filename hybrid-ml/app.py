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
    white_background: Optional[bool] = True # remove subject background → white backdrop
    debug: Optional[bool] = False           # also return raw Stage 1 + FLUX images


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

    # SegFormer (mattmdjaga/segformer_b2_clothes) label ids per garment category.
    # 4=Upper-clothes, 5=Skirt, 6=Pants, 7=Dress
    CATEGORY_LABELS = {
        "tops": [4],
        "bottoms": [5, 6],
        "one-pieces": [7],
    }

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

        # ============= Load clothing segmentation (for garment masking) =============
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
        self.seg_processor = SegformerImageProcessor.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        )
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        )
        self.seg_model.eval()
        print("Clothing segmentation model loaded.")

    def segment_garment_region(self, image, category):
        """Return a raw boolean mask (numpy, image-sized) of the garment region on
        `image`, or None if no garment pixels are found. No dilation/feather —
        callers combine and shape it."""
        import torch
        import numpy as np

        label_ids = self.CATEGORY_LABELS.get(category, [4, 5, 6, 7])

        inputs = self.seg_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = self.seg_model(**inputs).logits  # (1, C, h, w)

        # Upsample logits back to the original image resolution, then argmax.
        upsampled = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        seg = upsampled.argmax(dim=1)[0].cpu().numpy()

        region = np.isin(seg, label_ids)
        if not region.any():
            return None
        return region

    def garment_composite_mask(self, stage1_image, category):
        """Build the composite mask from the Stage 1 garment region so FLUX detail
        covers the WHOLE garment (this is what makes the result detailed). The edge
        is feathered INWARD only — clamped to the region, with no outward dilation —
        so the soft edge never spills below the hem, which is what previously leaked
        a hallucinated orange waistband."""
        import numpy as np
        from PIL import Image, ImageFilter

        region = self.segment_garment_region(stage1_image, category)
        if region is None:
            return None  # no garment in Stage 1 → nothing safe to composite onto

        region255 = (region.astype(np.uint8) * 255)
        feathered = Image.fromarray(region255, mode="L").filter(ImageFilter.GaussianBlur(6))
        # Clamp the feather to inside the original region: full-strength FLUX across
        # the interior, a soft inner edge, and zero spill past the hem.
        clamped = np.minimum(np.array(feathered), region255)
        return Image.fromarray(clamped, mode="L")

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
        print("Stage 2: FLUX.2 [klein] — enhancing garment detail (masked)...")

        # Generate at the person's true aspect ratio. Forcing a square canvas is
        # what stretched the body horizontally (the 'fat'/compressed bug); FLUX
        # editing is a full regeneration, so the canvas shape directly shapes the body.
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

        # ============= Composite — only the garment region comes from FLUX =============
        # Diffusers has no FLUX.2 inpaint pipeline yet, so we mask manually: take the
        # enhanced garment pixels from FLUX (full detail across the garment) and keep
        # face/body/proportions/background from the correctly-proportioned Stage 1
        # result. The mask is feathered inward only, so FLUX content never spills
        # below the hem (which is what previously leaked an orange waistband).
        flux_aligned = flux_out.resize((base_w, base_h), Image.LANCZOS)

        garment_mask = None
        try:
            garment_mask = self.garment_composite_mask(stage1_result, category)
        except Exception as e:
            print(f"Garment segmentation failed: {e}")

        if garment_mask is None:
            # Nothing safe to composite onto — return the well-proportioned Stage 1
            # result rather than a full FLUX regeneration that may distort the body.
            print("No garment mask detected — returning Stage 1 result.")
            final_result = stage1_result
        else:
            import numpy as np

            base_arr = np.array(stage1_result).astype(np.float32)
            flux_arr = np.array(flux_aligned).astype(np.float32)
            m = np.array(garment_mask).astype(np.float32) / 255.0
            m3 = m[..., None]

            composited = (flux_arr * m3 + base_arr * (1.0 - m3)).astype(np.uint8)
            final_result = Image.fromarray(composited)
            print("Composite complete — face, body and background preserved from Stage 1.")

        # Optionally drop the subject onto a clean white backdrop.
        if white_background:
            final_result = self.remove_background(final_result, label="subject")

        def encode_png(img):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        # Return the seed so the front end can display it and let users pin /
        # reproduce a result they liked.
        response = {"image": encode_png(final_result), "seed": seed}

        # In debug mode, also return the raw Stage 1 (FASHN) and FLUX outputs so
        # we can see which stage a hem/fit artifact actually comes from.
        if debug:
            response["stage1_image"] = encode_png(stage1_result)
            response["flux_image"] = encode_png(flux_aligned)

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
        "white_background": request.white_background,
        "debug": request.debug,
    })
    return result