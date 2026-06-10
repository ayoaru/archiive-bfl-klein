import modal
import io
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

# Define the Modal image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "wget",
        "libgles2",
        "libegl1",
        "libglfw3",
        "libgles2-mesa",
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
        "opencv-python",
        "mediapipe",
        "numpy",
    )
    .run_commands(
        "wget -O /root/blaze_face_short_range.tflite "
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
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

        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.enable_model_cpu_offload()
        print("Model loaded successfully.")

    def detect_face_mask(self, image):
        """Use MediaPipe to detect face and generate an inpainting mask."""
        import mediapipe as mp
        import numpy as np
        import cv2
        from PIL import Image

        img_array = np.array(image)
        h, w = img_array.shape[:2]

        base_options = mp.tasks.BaseOptions(
            model_asset_path="/root/blaze_face_short_range.tflite"
        )
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.2,
        )

        with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            )
            results = detector.detect(mp_image)

        print(f"Face detections found: {len(results.detections)}")

        if not results.detections:
            print("No face detected in Stage 1 output — skipping Stage 2.")
            return None

        detection = results.detections[0]
        bbox = detection.bounding_box

        x = bbox.origin_x
        y = bbox.origin_y
        bw = bbox.width
        bh = bbox.height

        # Expand bounding box by 40% to include hairline and neck
        padding_x = int(bw * 0.4)
        padding_y = int(bh * 0.4)
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        bw = min(w - x, bw + padding_x * 2)
        bh = min(h - y, bh + padding_y * 2)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y:y+bh, x:x+bw] = 255

        return Image.fromarray(mask).convert("RGB")

    @modal.method()
    def run_inference(self, request):
        import torch
        from PIL import Image

        height = request.get("height", 1024)
        width = request.get("width", 1024)
        num_inference_steps = request.get("num_inference_steps", 28)
        height_cm = request["height_cm"]
        weight_kg = request["weight_kg"]
        build = request["build"]
        item_name = request["item_name"]
        item_brand = request["item_brand"]
        item_category = request["item_category"]
        item_color = request["item_color"]

        def decode_image(b64_str, size=(768, 768)):
            img_bytes = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return img.resize(size, Image.LANCZOS)

        # ============= Stage 1 — Outfit Generation =============
        print("Stage 1: Generating outfit...")

        stage1_prompt = (
            f"A person with {build} build, {height_cm}cm tall, {weight_kg}kg, "
            f"wearing a {item_color} {item_name} by {item_brand}. "
            f"Use the first reference image to preserve the person's body proportions, "
            f"posture, and build exactly — correct leg to torso ratio, natural human proportions from head to toe. "
            f"Use the second and third reference images to reproduce the {item_category} exactly, "
            f"preserving all details, graphics, colors, and textures precisely. "
            f"Use the fourth and fifth reference images to understand exactly how the garment fits, "
            f"drapes, and sits on the body, including sleeve length, shoulder fit, and overall silhouette. "
            f"Standing upright, neutral pose, arms relaxed at sides, facing forward, "
            f"full body visible from head to toe. "
            f"Fashion photography, clean neutral background, high quality, photorealistic."
        )

        stage1_refs = [
            decode_image(request["full_body"]),    # 1st - body proportions
            decode_image(request["item_front"]),   # 2nd - item front
            decode_image(request["item_back"]),    # 3rd - item back
            decode_image(request["model_front"]),  # 4th - garment fit front
            decode_image(request["model_back"]),   # 5th - garment fit back
        ]

        stage1_result = self.pipe(
            prompt=stage1_prompt,
            image=stage1_refs,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(42),
        ).images[0]

        print("Stage 1 complete.")

        # ============= Stage 2 — Face Inpainting =============
        print("Stage 2: Applying face...")

        face_mask = None
        try:
            face_mask = self.detect_face_mask(stage1_result)
        except Exception as e:
            print(f"Face detection failed: {e}")

        if face_mask is None:
            print("Skipping Stage 2 — no face detected.")
            buffer = io.BytesIO()
            stage1_result.save(buffer, format="PNG")
            stage1_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return {
                "image": stage1_b64,
                "stage": "stage1_only",
                "reason": "no face detected"
            }

        face_photo = decode_image(request["face"], size=(1024, 1024))

        stage2_prompt = (
            f"The exact same person from the reference photo. "
            f"Preserve the person's face, skin tone, hair color, eye color, "
            f"and all facial features exactly — identical likeness. "
            f"Photorealistic, high quality, natural lighting, clean neutral background, "
            f"full body shot, standing upright."
        )

        stage2_refs = [
            decode_image(request["face"], size=(1024, 1024)),
            decode_image(request["face"], size=(1024, 1024)),
            stage1_result.resize((768, 768)),  # Stage 1 result as additional context
        ]

        stage2_result = self.pipe(
            prompt=stage2_prompt,
            image=stage2_refs,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(42),
        ).images[0]

        print("Stage 2 complete.")

        # Composite Stage 2 face onto Stage 1 body using the mask
        from PIL import Image as PILImage
        import numpy as np

        # Resize everything to the same size
        stage1_arr = np.array(stage1_result.resize((width, height)))
        stage2_arr = np.array(stage2_result.resize((width, height)))
        mask_arr = np.array(face_mask.resize((width, height)).convert("L"))

        # Normalize mask to 0-1
        mask_normalized = mask_arr.astype(float) / 255.0
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)

        # Blend: use Stage 2 face where mask is white, Stage 1 body where mask is black
        composited = (stage2_arr * mask_3ch + stage1_arr * (1 - mask_3ch)).astype(np.uint8)
        final_result = PILImage.fromarray(composited)

        print("Compositing complete.")

        # Encode final output as base64
        buffer = io.BytesIO()
        final_result.save(buffer, format="PNG")
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
    result = await FluxInference().run_inference.remote.aio({
        "item_name": request.item_name,
        "item_brand": request.item_brand,
        "item_category": request.item_category,
        "item_color": request.item_color,
        "face": request.face,
        "full_body": request.full_body,
        "item_front": request.item_front,
        "item_back": request.item_back,
        "model_front": request.model_front,
        "model_back": request.model_back,
        "height_cm": request.height_cm,
        "weight_kg": request.weight_kg,
        "build": request.build,
        "height": request.height,
        "width": request.width,
        "num_inference_steps": request.num_inference_steps,
    })
    return result