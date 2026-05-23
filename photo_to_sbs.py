#!/usr/bin/env python3
"""
photo_to_sbs.py — Convert a folder of photos to Full-SBS 3D stereo photos.

Usage:
    python photo_to_sbs.py <input_folder> <output_folder>

Full-SBS images are written to <output_folder>.

The settings are hard-coded for a strong "pop-out" 3D feeling, based on the
"aggressive pop preset" documented in UserGuide.md and VisionDepth3D_Method.md.
"""

import argparse
import os
import sys
from pathlib import Path

# ── Make sure we can import VisionDepth3D modules regardless of cwd ──────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
os.chdir(_SCRIPT_DIR)

import numpy as np
import torch
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
#  Hard-coded settings — "aggressive pop" preset
#  (see UserGuide.md § Foreground/Midground/Background Shift)
# ═══════════════════════════════════════════════════════════════════════════════

# Depth model to use for AI depth estimation
MODEL_NAME       = "Depth Anything v2 Large"          # key in supported_models dict
INFERENCE_RES    = (518, 518)                         # inference resolution (W, H)

# Stereo shift values  (UserGuide.md "aggressive pop preset")
FG_SHIFT         = -10.0   # foreground: negative pulls subjects toward the viewer
MG_SHIFT         =  -2.0   # midground: slight negative for natural depth layering
BG_SHIFT         =  +4.5   # background: positive pushes distant areas deeper

# Global stereo strength controls
PARALLAX_BALANCE = 1.0     # 0.7 = gentle; 1.0 = showcase strength
IPD_FACTOR       = 1.2     # virtual eye distance multiplier (>1 = wider separation)
MAX_PIXEL_SHIFT  = 0.060   # maximum parallax as fraction of image width (6 %)

# Depth shaping — accentuates near/far separation
DEPTH_POP_GAMMA           = 0.60   # <1 = steeper near-end curve for more pop
DEPTH_POP_MID             = 0.45   # pivot point for the gamma curve
DEPTH_STRETCH_LO          = 0.02   # stretch depth histogram from this percentile
DEPTH_STRETCH_HI          = 0.98   # …to this percentile
FG_POP_MULTIPLIER         = 1.35   # extra boost to foreground region
BG_PUSH_MULTIPLIER        = 1.20   # extra push to background region
SUBJECT_LOCK_STRENGTH     = 0.15   # how strongly the subject is locked to z=0
FG_CURVATURE_STRENGTH     = 0.30   # foreground curvature (adds depth "rounding")

# Visual refinement
SHARPNESS        = 0.3     # mild sharpening of stereo edges
ZERO_PARALLAX    = 0.0     # fine-tune where the screen plane sits

# Output
OUTPUT_FORMAT    = "Full-SBS"        # each output image is left|right side-by-side
ASPECT_RATIO_KEY = "Default (16:9)"  # used only when preserve_original_aspect=False

# Supported input extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Aspect ratio table (mirrors RenderService._get_aspect_ratios)
ASPECT_RATIOS = {
    "Default (16:9)":           16 / 9,
    "Classic (4:3)":             4 / 3,
    "Square (1:1)":              1.0,
    "Vertical 9:16":             9 / 16,
    "CinemaScope (2.39:1)":      2.39,
    "Anamorphic (2.35:1)":       2.35,
    "Modern Cinema (2.40:1)":    2.40,
    "Academy Flat (1.85:1)":     1.85,
    "21:9 UltraWide":           21 / 9,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Depth model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_depth_model() -> None:
    """
    Download (if needed) and initialise the depth model.
    Sets the global `pipe` / `pipe_type` in core.render_depth so that
    _run_pipe_or_tile() works correctly.
    """
    import core.render_depth as rd
    from transformers import pipeline as hf_pipeline

    checkpoint = rd.supported_models.get(MODEL_NAME)
    if checkpoint is None:
        raise ValueError(
            f"Unknown model name: {MODEL_NAME!r}. "
            f"Available: {list(rd.supported_models.keys())}"
        )

    print(f"[depth] Loading model: {MODEL_NAME}  ({checkpoint})")
    model_callable, meta = rd.ensure_model_downloaded(
        checkpoint,
        use_fp16=torch.cuda.is_available(),
    )
    if model_callable is None:
        raise RuntimeError(f"Failed to download/load depth model: {MODEL_NAME}")

    # For standard HF models ensure_model_downloaded returns (model, processor).
    # Wrap them in a HuggingFace depth-estimation pipeline, just like update_pipeline().
    caps = meta if isinstance(meta, dict) else {}
    is_special = (
        caps.get("is_onnx")
        or caps.get("is_diffusion")
        or caps.get("kind") in ("vda", "da3", "vda_onnx")
        or getattr(model_callable, "_is_marigold", False)
        or callable(model_callable) and caps  # adapter already callable
    )

    if is_special:
        # Adapter is already a ready-to-call pipe; honour its type tag.
        rd.pipe = model_callable
        kind = caps.get("kind", "")
        if caps.get("is_onnx"):
            rd.pipe_type = "onnx"
        elif caps.get("is_diffusion"):
            rd.pipe_type = "diffusion_depth"
        elif kind == "vda":
            rd.pipe_type = "vda"
        elif kind == "da3":
            rd.pipe_type = "da3"
        else:
            rd.pipe_type = "hf"
    else:
        # Standard HuggingFace transformer: build a depth-estimation pipeline.
        processor = meta  # AutoImageProcessor / AutoProcessor
        device_idx: int | str
        if rd.torch_device.type == "cuda":
            device_idx = 0
        elif rd.torch_device.type == "mps":
            device_idx = "mps"
        else:
            device_idx = -1

        raw_pipe = hf_pipeline(
            "depth-estimation",
            model=model_callable,
            image_processor=processor,
            device=device_idx,
        )

        use_fp16 = torch.cuda.is_available()

        def _depth_pipe(images: list, inference_size=None, **_):
            if inference_size:
                images = [img.resize(inference_size, Image.BICUBIC) for img in images]
            with torch.inference_mode():
                if use_fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        result = raw_pipe(images)
                else:
                    result = raw_pipe(images)
            return result if isinstance(result, list) else [result]

        rd.pipe = _depth_pipe
        rd.pipe_type = "hf"

    print(f"[depth] Model ready on {rd.device_display_name()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-image depth inference
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_depth(pil_image: Image.Image) -> np.ndarray:
    """
    Run the loaded depth model on a single PIL image.
    Returns an 8-bit grayscale numpy array (H×W uint8).
    """
    import core.render_depth as rd

    predictions = rd._run_pipe_or_tile([pil_image], inference_size=INFERENCE_RES)
    raw_depth = predictions[0]["predicted_depth"]
    arr = rd._pred_to_np(raw_depth).squeeze()
    depth8 = rd.normalize_depth(arr, pil_image.size, invert=False, bit_depth=8)
    return depth8


# ═══════════════════════════════════════════════════════════════════════════════
#  Main conversion loop
# ═══════════════════════════════════════════════════════════════════════════════

def convert_folder(input_folder: Path, output_folder: Path) -> None:
    import cv2
    from core.render_3d import render_sbs_3d_image_from_arrays

    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in input_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    total = len(image_files)
    print(f"Found {total} image(s).  Output → {output_folder}\n")

    for idx, img_path in enumerate(image_files, start=1):
        print(f"[{idx}/{total}] {img_path.name}")

        # ── 1. Load source image ──────────────────────────────────────────────
        pil_img = Image.open(img_path).convert("RGB")

        # ── 2. Estimate depth (stays in memory — no temp files) ───────────────
        print("        estimating depth…")
        depth8 = estimate_depth(pil_img)  # H×W uint8 grayscale ndarray

        # ── 3. Load source as BGR for the renderer ────────────────────────────
        frame_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"        ⚠  could not read {img_path.name} with OpenCV, skipping")
            continue

        # ── 4. Render Full-SBS ────────────────────────────────────────────────
        print("        rendering Full-SBS…")
        # Keep the original file extension so JPEG photos stay JPEG.
        sbs_path = output_folder / f"{img_path.stem}_LR_Full_SBS{img_path.suffix}"

        result = render_sbs_3d_image_from_arrays(
            frame=frame_bgr,
            depth=depth8,           # grayscale 2D array — render_3d handles it
            output_image_path=str(sbs_path),

            # ── Stereo shift (aggressive pop) ────────────────────────────────
            fg_shift=FG_SHIFT,
            mg_shift=MG_SHIFT,
            bg_shift=BG_SHIFT,

            # ── Output format ────────────────────────────────────────────────
            output_format=OUTPUT_FORMAT,
            selected_aspect_ratio=ASPECT_RATIO_KEY,
            aspect_ratios=ASPECT_RATIOS,
            preserve_original_aspect=True,  # never crop/stretch source images

            # ── Depth shaping ────────────────────────────────────────────────
            depth_pop_gamma=DEPTH_POP_GAMMA,
            depth_pop_mid=DEPTH_POP_MID,
            depth_stretch_lo=DEPTH_STRETCH_LO,
            depth_stretch_hi=DEPTH_STRETCH_HI,
            fg_pop_multiplier=FG_POP_MULTIPLIER,
            bg_push_multiplier=BG_PUSH_MULTIPLIER,
            subject_lock_strength=SUBJECT_LOCK_STRENGTH,
            foreground_curvature_strength=FG_CURVATURE_STRENGTH,

            # ── Global stereo strength ────────────────────────────────────────
            parallax_balance=PARALLAX_BALANCE,
            ipd_factor=IPD_FACTOR,
            max_pixel_shift_percent=MAX_PIXEL_SHIFT,
            zero_parallax_strength=ZERO_PARALLAX,

            # ── Visual refinement ────────────────────────────────────────────
            sharpness_factor=SHARPNESS,
            enable_edge_masking=True,
            enable_feathering=True,
            feather_strength=0.0,
            blur_ksize=1,
            use_floating_window=True,   # cinematic edge protection for pop-out

            # ── Convergence (disable for stills — no temporal smoothing) ──────
            convergence_strength=0.0,
            enable_dynamic_convergence=False,

            # ── Miscellaneous ────────────────────────────────────────────────
            dof_strength=0.0,
            auto_crop_black_bars=False,
            use_subject_tracking=False,
            color_saturation=1.0,
            color_contrast=1.0,
            color_brightness=0.0,
            eye_mode="sbs",
            disable_shift_ema=True,     # no EMA smoothing needed for single images
        )

        if result:
            print(f"        saved → {Path(result).name}")
        else:
            print(f"        ⚠  render returned no output for {img_path.name}")

    print(f"\n✓ Done.  {total} image(s) converted.  SBS output: {output_folder}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a photo folder to Full-SBS 3D stereo (strong pop-out preset).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Depth model : Depth Anything v2 Large
Output format: Full-SBS (left|right side-by-side)
Preset       : Aggressive pop-out
  FG shift -10.0 / MG shift -2.0 / BG shift +4.5
  parallax_balance 1.0 / IPD 1.2 / max_pixel_shift 4.5%
        """,
    )
    parser.add_argument("input_folder",  help="Folder containing source photos")
    parser.add_argument("output_folder", help="Folder where SBS outputs will be saved")
    args = parser.parse_args()

    input_folder  = Path(args.input_folder).expanduser().resolve()
    output_folder = Path(args.output_folder).expanduser().resolve()

    if not input_folder.is_dir():
        parser.error(f"Input folder not found: {input_folder}")

    load_depth_model()
    convert_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
