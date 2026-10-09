#!/usr/bin/env python3
"""使用随工具包提供的 MobileSeg 模型，对图像递归生成同尺寸二值前景 mask。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
sys.path.insert(0, str(MODEL_DIR))
from train_fp_mask_models import build_model  # noqa: E402

EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"无法写入：{path}")
    encoded.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "mobile_seg_best.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu；无 CUDA 时自动回退 CPU。")
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold 必须位于 (0, 1)")
    input_root, output_root, checkpoint = (args.input_root.resolve(), args.output_root.resolve(), args.checkpoint.resolve())
    if not input_root.is_dir() or not checkpoint.is_file():
        parser.error("输入目录或 checkpoint 不存在")
    if output_root.exists() and any(output_root.iterdir()):
        parser.error(f"输出目录非空，拒绝覆盖：{output_root}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint_data.get("args", {}) if isinstance(checkpoint_data, dict) else {}
    model = build_model(saved_args.get("model", "mobile_seg"), int(saved_args.get("base_ch", 16)))
    model.load_state_dict(checkpoint_data["model"] if isinstance(checkpoint_data, dict) and "model" in checkpoint_data else checkpoint_data)
    model.to(device).eval()

    files = sorted(p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        parser.error("未发现支持的输入图像")
    for index, path in enumerate(files, start=1):
        image = read_gray(path)
        resized = cv2.resize(image, (100, 110), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized[None, None]).to(device)
        with torch.inference_mode():
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        mask = cv2.resize((probability >= args.threshold).astype(np.uint8) * 255, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = output_root / path.relative_to(input_root)
        write_image(out, mask)
        print(f"[{index}/{len(files)}] {path.relative_to(input_root)}")
    print(f"mask 输出：{output_root}")


if __name__ == "__main__":
    main()
