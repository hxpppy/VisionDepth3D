Upstream:
https://github.com/VisionDepth/VisionDepth3D

Changes on top of upstream:
- Use Devbox for having Python venv
- Cap image inputs to 4K (MAX_IMAGE_SIDE = 3840 px per side) for both depth-map generation
  and 3D stereo rendering. 4K is the practical limit for 3D headset hardware in 2026;
  exceeding it causes out-of-memory errors and hits headset texture-size limits.
- Save 3D stereo output images as JPEG (.jpg, quality 95) instead of PNG to reduce file size.
- Add `photo_to_sbs.py` CLI that converts an photo folder to SBS directly,
  without saving temporary depth map files. Run with:
  `devbox run photo_to_sbs -- <input_folder> <output_folder>`
- Fix binary-looking depth maps for metric depth models (DepthPro, ZoeDepth,
  DA-v2 Metric). These models output absolute depth in metres; linear
  normalisation compressed the near-field into <2 % of the grey-level range.
  Apply `log1p` before the percentile stretch so that equal perceptual depth
  steps map to equal grey steps. Controlled by `DEPTH_IS_METRIC` (set
  automatically at model load time via `is_metric_depth_checkpoint()`).
  Affects image, folder, and video processing paths.
  `photo_to_sbs.py` also auto-inverts the depth map for metric models
  (`invert=DEPTH_IS_METRIC`) because metric depth (near=small) is the
  opposite convention to relative depth (near=large); without inversion
  the stereo renderer applies parallax backwards.
