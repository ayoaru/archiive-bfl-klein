"""
One-shot local tester for the hybrid try-on endpoint (hybrid-ml/app.py).

Replaces the manual loop (b64encoder.py -> Postman -> b64decoder.py): it encodes
the local images, calls the endpoint, decodes the response, saves the PNG, and
opens it.

Setup (once):
    pip install requests
    set the endpoint URL via any of:
      - .env file:  put `ARCHIIVE_ENDPOINT=https://...` in hybrid-ml/.env  (recommended)
      - env var:    $env:ARCHIIVE_ENDPOINT = "https://<workspace>--archiive-ml-fastapi-app.modal.run"
      - or --url flag on each call

Examples (run from the repo root):
    python hybrid-ml/test_pipeline.py --person ayo_fb2.JPG --garment item_front.png
    # Pull inputs from one folder, write results to another:
    python hybrid-ml/test_pipeline.py --input-dir test_images --output-dir results \
        --person ayo_fb2.JPG --garment item_front.png
    python hybrid-ml/test_pipeline.py --person ayo_fb2.JPG --garment item_front.png --category tops \
        --item-name "Box Logo Hoodie" --item-brand Supreme --item-category hoodie --item-color black
    # Reproduce a previous result:
    python hybrid-ml/test_pipeline.py --person ayo_fb2.JPG --garment item_front.png --seed 12345
"""

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests


def load_env_file(path: Path) -> None:
    """Minimal .env loader (no extra deps): reads KEY=VALUE lines into os.environ
    without clobbering variables already set in the real environment."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# Load hybrid-ml/.env (the directory this script lives in) before reading env vars.
load_env_file(Path(__file__).with_name(".env"))


def resolve_input(name: str, input_dir: Path) -> Path:
    """Resolve an image path: use it as-is if it points to a real file, otherwise
    look for it inside input_dir."""
    p = Path(name)
    if p.is_file():
        return p
    candidate = input_dir / name
    if candidate.is_file():
        return candidate
    sys.exit(f"Image not found: tried {p.resolve()} and {candidate.resolve()}")


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the hybrid try-on endpoint with local images.")
    parser.add_argument("--input-dir", default=".",
                        help="Folder to look for --person/--garment in (default: current dir). "
                             "Ignored for args that are already a valid path.")
    parser.add_argument("--output-dir", default="outputs",
                        help="Folder to save the result PNG in (default: outputs/).")
    parser.add_argument("--person", required=True, help="Person/model image (name within --input-dir, or a path).")
    parser.add_argument("--garment", required=True, help="Garment image (name within --input-dir, or a path).")
    parser.add_argument("--category", default="tops", choices=["tops", "bottoms", "one-pieces"])
    parser.add_argument("--item-name", default="Test Item")
    parser.add_argument("--item-brand", default="Test Brand")
    parser.add_argument("--item-category", default="hoodie")
    parser.add_argument("--item-color", default="black")
    parser.add_argument("--garment-photo-type", default="auto", choices=["auto", "model", "flat-lay"],
                        help="How FASHN interprets the garment image (affects fit/hem).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Omit for a fresh random result; pass a value to reproduce one.")
    parser.add_argument("--num-timesteps", type=int, default=50, help="FASHN steps.")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="FASHN guidance.")
    parser.add_argument("--flux-steps", type=int, default=28, help="FLUX steps.")
    parser.add_argument("--url", default=os.environ.get("ARCHIIVE_ENDPOINT"),
                        help="Endpoint base URL (or set ARCHIIVE_ENDPOINT).")
    parser.add_argument("--out", default=None,
                        help="Explicit output PNG path; overrides --output-dir "
                             "(default name: <output-dir>/<timestamp>_seed<seed>.png).")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open the result.")
    parser.add_argument("--no-face-swap", dest="face_swap", action="store_false",
                        help="Skip Stage 3 face swap (return FLUX's own face).")
    parser.add_argument("--debug", action="store_true",
                        help="Also save the raw Stage 1 (FASHN) and FLUX outputs next to the result.")
    args = parser.parse_args()

    if not args.url:
        sys.exit("No endpoint URL. Pass --url or set ARCHIIVE_ENDPOINT "
                 "(e.g. https://<workspace>--archiive-ml-fastapi-app.modal.run).")

    input_dir = Path(args.input_dir)
    person_path = resolve_input(args.person, input_dir)
    garment_path = resolve_input(args.garment, input_dir)

    endpoint = args.url.rstrip("/") + "/try-on"
    payload = {
        "person_image": encode(person_path),
        "garment_image": encode(garment_path),
        "category": args.category,
        "item_name": args.item_name,
        "item_brand": args.item_brand,
        "item_category": args.item_category,
        "item_color": args.item_color,
        "garment_photo_type": args.garment_photo_type,
        "num_timesteps": args.num_timesteps,
        "guidance_scale": args.guidance_scale,
        "flux_steps": args.flux_steps,
        "seed": args.seed,
        "face_swap": args.face_swap,
        "debug": args.debug,
    }

    print(f"POST {endpoint}  (seed={args.seed if args.seed is not None else 'random'})")
    start = time.time()
    try:
        resp = requests.post(endpoint, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Request failed: {e}")

    data = resp.json()
    if "image" not in data:
        sys.exit(f"Unexpected response (no image): {data}")

    seed = data.get("seed", args.seed)
    out_path = Path(args.out) if args.out else Path(args.output_dir) / f"{int(start)}_seed{seed}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(data["image"]))

    print(f"Done in {time.time() - start:.1f}s  ->  {out_path}  (seed={seed})")
    if "face_swap" in data:
        print(f"  face swap: {data['face_swap']}")

    # In debug mode, save the raw Stage 1 / FLUX images beside the result so we
    # can see which stage a hem/fit artifact comes from.
    for key, suffix in (("stage1_image", "stage1"), ("flux_image", "flux")):
        if key in data:
            side_path = out_path.with_name(f"{out_path.stem}_{suffix}.png")
            side_path.write_bytes(base64.b64decode(data[key]))
            print(f"  debug: {side_path}")

    if not args.no_open:
        try:
            os.startfile(out_path)  # Windows
        except AttributeError:
            import webbrowser
            webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
