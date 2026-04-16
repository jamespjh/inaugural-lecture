#!/usr/bin/env python3
"""Convert UCL logo matte background to alpha-transparent PNG.

Default behavior:
- Input:  ucl-logo-original.png
- Output: ucl-logo.png
- Estimate matte color from image border median.
- Build alpha from distance to matte color.
- Decontaminate edge RGB from matte.
- Apply a small alpha choke.
- Force left/right side strips to alpha 0 to remove residual contamination.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def convert_logo(
    input_path: Path,
    output_path: Path,
    low: float = 16.0,
    high: float = 82.0,
    choke: float = 14.0,
    side_strip_px: int = 20,
) -> tuple[tuple[int, int, int], int, int, int]:
    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img).astype(np.float32)
    rgb = arr[:, :, :3]

    # Estimate matte from border pixels.
    top = rgb[0, :, :]
    bottom = rgb[-1, :, :]
    left = rgb[:, 0, :]
    right = rgb[:, -1, :]
    border = np.concatenate([top, bottom, left, right], axis=0)
    matte = np.median(border, axis=0)

    # Distance to matte color.
    dist = np.linalg.norm(rgb - matte, axis=2)

    # Alpha ramp.
    alpha = np.zeros_like(dist, dtype=np.float32)
    alpha[dist >= high] = 255.0
    mid = (dist > low) & (dist < high)
    alpha[mid] = (dist[mid] - low) * (255.0 / (high - low))

    # Decontaminate foreground color by inverting compositing against matte.
    a = alpha / 255.0
    safe_a = np.where(a > 1e-6, a, 1.0)
    fg = (rgb - (1.0 - a)[:, :, None] *
          matte[None, None, :]) / safe_a[:, :, None]
    fg = np.clip(fg, 0.0, 255.0)

    # Choke alpha slightly to suppress fringe.
    alpha = np.clip(alpha - choke, 0.0, 255.0)

    # Zero RGB where alpha is near zero to prevent dark halos.
    low_alpha_mask = alpha < 10.0
    fg[low_alpha_mask] = 0.0

    # Force side strips transparent.
    h, w = alpha.shape
    if side_strip_px > 0:
        strip = min(side_strip_px, w // 2)
        alpha[:, :strip] = 0.0
        alpha[:, w - strip:] = 0.0

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = fg.astype(np.uint8)
    out[:, :, 3] = alpha.astype(np.uint8)

    Image.fromarray(out, mode="RGBA").save(output_path)

    matte_rgb = tuple(int(round(x)) for x in matte.tolist())
    fully_transparent = int(np.sum(out[:, :, 3] == 0))
    low_alpha = int(np.sum((out[:, :, 3] > 0) & (out[:, :, 3] <= 40)))
    visible = int(np.sum(out[:, :, 3] > 40))
    return matte_rgb, fully_transparent, low_alpha, visible


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert UCL logo matte to alpha.")
    parser.add_argument(
        "--input",
        default="ucl-logo-original.png",
        help="Input logo path (default: ucl-logo-original.png)",
    )
    parser.add_argument(
        "--output",
        default="ucl-logo.png",
        help="Output logo path (default: ucl-logo.png)",
    )
    parser.add_argument("--low", type=float, default=16.0)
    parser.add_argument("--high", type=float, default=82.0)
    parser.add_argument("--choke", type=float, default=14.0)
    parser.add_argument("--side-strip", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    matte_rgb, fully_transparent, low_alpha, visible = convert_logo(
        input_path=input_path,
        output_path=output_path,
        low=args.low,
        high=args.high,
        choke=args.choke,
        side_strip_px=args.side_strip,
    )

    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print(f"matte color: {matte_rgb}")
    print(f"alpha==0: {fully_transparent}")
    print(f"alpha 1..40: {low_alpha}")
    print(f"alpha>40: {visible}")


if __name__ == "__main__":
    main()
