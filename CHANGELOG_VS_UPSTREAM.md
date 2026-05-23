Upstream:
https://github.com/VisionDepth/VisionDepth3D

Changes on top of upstream:
- Use Devbox for having Python venv
- Cap image inputs to 4K (MAX_IMAGE_SIDE = 3840 px per side) for both depth-map generation
  and 3D stereo rendering. 4K is the practical limit for 3D headset hardware in 2026;
  exceeding it causes out-of-memory errors and hits headset texture-size limits.
