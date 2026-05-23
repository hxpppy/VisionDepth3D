# core/image_utils.py

# 4K is the practical maximum for 3D headset hardware in 2026. Without this cap:
# - Generating stereo images from very large inputs causes out-of-memory errors.
# - 3D headsets have a texture size limit; images exceeding it cannot be displayed.
# This guard is applied at image load time so no downstream logic needs changing.
MAX_IMAGE_SIDE = 3840  # max pixels on any single side for output depth/stereo images


def clamp_image_to_max_side(img):
    """
    Resize *img* proportionally so its longest side does not exceed MAX_IMAGE_SIDE.

    Accepts:
      - PIL.Image.Image  (used by the depth-map pipeline)
      - numpy ndarray    (OpenCV BGR / grayscale, used by the 3D-stereo pipeline)

    Returns the image unchanged if it already fits, or a resized copy otherwise.
    """
    import numpy as np
    from PIL import Image

    if isinstance(img, Image.Image):
        w, h = img.size
        if max(w, h) <= MAX_IMAGE_SIDE:
            return img
        scale = MAX_IMAGE_SIDE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"⚠️  Input image {w}×{h} exceeds MAX_IMAGE_SIDE={MAX_IMAGE_SIDE}; resizing to {new_w}×{new_h}.")
        return img.resize((new_w, new_h), Image.LANCZOS)

    elif isinstance(img, np.ndarray):
        import cv2
        h, w = img.shape[:2]
        if max(w, h) <= MAX_IMAGE_SIDE:
            return img
        scale = MAX_IMAGE_SIDE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"⚠️  Input image {w}×{h} exceeds MAX_IMAGE_SIDE={MAX_IMAGE_SIDE}; resizing to {new_w}×{new_h}.")
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    else:
        raise TypeError(f"clamp_image_to_max_side: unsupported image type {type(img)}")
