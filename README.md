# archiive-bfl-klein

Python ML microservice that powers image-to-image virtual try-on generation for [Archiive](https://github.com/ayoaru/archiive). Deployed on [Modal](https://modal.com) with serverless GPU (A100-80GB) inference, exposed over a FastAPI HTTP layer.

This repository is for non-commercial/educational use.

## What it does

Given a photo of a person and a photo of a garment, the service returns a photorealistic image of that person wearing that garment — with the garment's actual graphics, texture, and color preserved, and the person's real face intact.

That sounds simple, but naively feeding a person + garment into a single diffusion model tends to either warp the person's body proportions, hallucinate the garment's details, or drift the person's face into someone else. This repo is the record of solving those failure modes, arriving at a **three-stage hybrid pipeline** after two earlier single-model approaches proved insufficient on their own.

## Architecture

```
person photo ─┐
              ├─► Stage 1: FASHN VTON  ──► fitted garment (drape/fit only, not shown to user)
garment photo ┘         │
                         ▼
              Stage 2: FLUX.2 [klein] ──► garment detail + texture enhancement
                         │                (regenerates the person at their true
                         │                 aspect ratio; FLUX also drifts the face)
                         ▼
              Stage 3: InsightFace   ──► swaps the real face back onto the
              (buffalo_l + inswapper)     FLUX output, landmark-aligned
                         │
                         ▼
                  final composite (+ optional white-background cutout)
```

**Why three stages instead of one model:**

- **FASHN VTON alone** ([`fashn-ml/app.py`](fashn-ml/app.py)) nails garment *fit* (how it drapes on the body) but is limited on fine-grained detail fidelity — logos, patch placement, fabric texture.
- **FLUX.2 [klein] alone** ([`flux-ml/app.py`](flux-ml/app.py)) can reproduce garment detail with much higher fidelity from reference images, but as a general-purpose image model it doesn't understand "virtual try-on" as a task — it will subtly reshape the body and, critically, regenerate a *different* face each run.
- **The hybrid pipeline** ([`hybrid-ml/app.py`](hybrid-ml/app.py)) uses FASHN's output purely as a conditioning signal for FLUX (never shown to the end user), letting FLUX do what it's good at — high-fidelity garment rendering — while constrained by a garment fit FASHN already solved. A dedicated face-swap stage then repairs FLUX's one weak spot (identity drift) without touching anything else FLUX generated.

An earlier version of Stage 3 tried to *composite* the FLUX garment onto the FASHN body using a segmentation mask (SegFormer). That merged two independently-generated bodies that don't pixel-align, producing clipped shoulders and visible texture seams. Switching to "keep the FLUX image whole, swap only the small self-contained face region" eliminated that entire class of artifact — see the [pipeline stage three](https://github.com/ayoaru/archiive-bfl-klein/commit/a493621) commit.

### Directory layout

| Path | Purpose |
|---|---|
| [`fashn-ml/`](fashn-ml/) | Standalone FASHN VTON service — garment fitting, background removal (`rembg`) |
| [`flux-ml/`](flux-ml/) | Standalone FLUX.2 [klein] service — two-stage generation (outfit + face inpaint) with MediaPipe-based face masking |
| [`hybrid-ml/`](hybrid-ml/) | Production pipeline combining both models plus InsightFace face-swap; includes [`test_pipeline.py`](hybrid-ml/test_pipeline.py), a one-shot local test harness that hits the deployed endpoint and saves/opens the result |
| [`inputs/`](inputs/), [`outputs/`](outputs/) | Local scratch space for test images and generated results (gitignored contents) |
| [`b64encoder.py`](b64encoder.py), [`b64decoder.py`](b64decoder.py) | Early manual dev tools for base64-encoding request images / decoding response images before `test_pipeline.py` automated the loop |

### Design decisions worth calling out

- **Serverless GPU via Modal**: each service defines its container image, model-weight volumes, and endpoint declaratively (`modal.Image`, `modal.Volume`, `@modal.asgi_app`). Weights are downloaded once per volume and cached, so cold starts don't re-pull multi-GB checkpoints.
- **Aspect-ratio-safe generation** (`flux_dims` in [`hybrid-ml/app.py`](hybrid-ml/app.py)): forcing FLUX onto a square canvas stretched portrait subjects ("fat"/compressed bug). Dimensions are now scaled to a FLUX-valid multiple-of-16 while preserving the person's true aspect ratio.
- **Seed propagation for reproducibility**: a single seed drives both FASHN and FLUX, is returned in the response, and can be pinned via `test_pipeline.py --seed` to reproduce or `regenerate` (by omitting it) for a fresh variation.
- **Graceful degradation**: face swap and background removal both fail open — if no face is detected in either image, or a stage errors, the pipeline returns the best image it has rather than a hard failure.
- **Debug mode**: `debug: true` on the `/try-on` request returns the intermediate FASHN and FLUX images alongside the final result, so a fit artifact can be traced to the exact stage that produced it.

### Known limitation

`inswapper_128.onnx` (Stage 3) has murky commercial licensing and was chosen to validate the visual approach quickly. It's flagged as a pre-launch swap-out for a commercially licensed face-swap model or API.

## Roadmap / Build Log

Summary of how the pipeline evolved, from initial commit to current state:

1. **Base FLUX model + API endpoints** — initial FastAPI/Modal scaffold, single-model image generation.
2. **FLUX two-stage pipeline** — split generation into outfit synthesis + face inpainting (MediaPipe face detection/masking) to fix identity drift from a single generation pass.
3. **FASHN VTON base model** — added a dedicated virtual try-on model for garment fit, as a second, independently deployed service.
4. **Background removal** — `rembg`-based garment/subject background stripping for cleaner conditioning and output.
5. **Hybrid pipeline base** — combined FASHN (fit) + FLUX (detail) into a single service, using FASHN output purely as FLUX conditioning.
6. **Testing script** (`test_pipeline.py`) — replaced the manual encode → Postman → decode loop with a one-shot CLI harness.
7. **VTO input tuning + masking** — iterated on garment/model reference inputs and mask generation for fit accuracy.
8. **Param tuning** — inference step counts, guidance scale, and prompt refinement across stages.
9. **Pipeline stage three** — added InsightFace-based face swap as a dedicated final stage, replacing the earlier segmentation-mask composite that produced body-merge artifacts (clipped shoulders, texture seams).

### Open items

- Swap `inswapper_128` for a production-licensed face-swap model before launch.
- FASHN Stage 1 waist-cinching artifact on certain garment types (tracked for `garment_photo_type`/segmentation-free fix).
- Garment blending into similarly-colored existing clothing in some inputs — mitigated by passing explicit `item_color`, further tuning pending.

## Running locally

Each service (`fashn-ml/`, `flux-ml/`, `hybrid-ml/`) deploys independently to Modal:

```bash
modal deploy hybrid-ml/app.py
```

Then exercise the deployed endpoint against local test images:

```bash
python hybrid-ml/test_pipeline.py --person person.jpg --garment garment.png \
    --item-name "Box Logo Hoodie" --item-brand Supreme --item-category hoodie --item-color black
```

See the docstring at the top of [`test_pipeline.py`](hybrid-ml/test_pipeline.py) for the full flag reference (endpoint config via `.env`, `--seed` for reproducibility, `--debug` for intermediate stage outputs).

## Related

Front-end / product repo: [archiive](https://github.com/ayoaru/archiive)
