#!/usr/bin/env python3
"""前端指纹处理流水线：衍射校正 -> V3 UNet 去噪 -> 可选分割/质量评分。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(stage: str, command: list[str]) -> None:
    print(f"\n{'=' * 72}\n{stage}\n{' '.join(map(str, command))}\n{'=' * 72}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="已完成 pair 编号的原始 CSV 根目录。")
    parser.add_argument("--output-root", type=Path, required=True, help="本次运行的新输出根目录。")
    parser.add_argument("--diffraction", choices=("v7", "v8", "v9"), default="v7")
    parser.add_argument("--skip-denoise", action="store_true")
    parser.add_argument("--run-segmentation", action="store_true")
    parser.add_argument("--run-quality", action="store_true")
    parser.add_argument("--device", default="cuda", help="分割使用的设备；去噪自动选择 CUDA。")
    parser.add_argument("--folder", action="append", default=[], help="仅处理指定一级手指目录；可重复指定。")
    parser.add_argument("--limit-pairs", type=int, default=0, help="每个手指最多处理前 N 个 pair；0 表示全部。")
    args = parser.parse_args()
    source, output = args.input_root.resolve(), args.output_root.resolve()
    if not source.is_dir():
        parser.error(f"输入目录不存在：{source}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"输出目录非空，拒绝覆盖：{output}")
    output.mkdir(parents=True, exist_ok=True)

    if args.diffraction == "v7":
        diff_root, pic_root, corrected = output / "01_diff", output / "02_pic", output / "03_diffraction_v7"
        selection = [item for folder in args.folder for item in ("--folder", folder)]
        run("1/4 原始 wo-wi 差分", [sys.executable, str(HERE / "00_preprocess/diff_final.py"), "--input-root", str(source), "--output-root", str(diff_root), *selection])
        run("2/4 差分 CSV 转五相位 BMP（pair 联合归一化）", [sys.executable, str(HERE / "00_preprocess/pic_final.py"), "--input-root", str(diff_root), "--output-root", str(pic_root), "--normalization", "per-pair", *selection])
        run("3/4 V7 衍射校正", [sys.executable, str(HERE / "01_diffraction/diffaractionv7.py"), "--input-root", str(pic_root), "--output-root", str(corrected), "--input-mode", "direct-diff", *selection])
    else:
        corrected = output / f"03_diffraction_{args.diffraction}"
        script = HERE / "01_diffraction" / f"diffaraction{args.diffraction}.py"
        selection = [item for folder in args.folder for item in ("--folder", folder)]
        if args.limit_pairs:
            selection.extend(["--limit-pairs", str(args.limit_pairs)])
        run(f"1/3 {args.diffraction.upper()} 衍射校正（直接读取原始 CSV）", [sys.executable, str(script), "--input-root", str(source), "--output-root", str(corrected), "--flat-output", *selection])

    final_images = corrected
    if not args.skip_denoise:
        final_images = output / "04_denoise"
        run("V3 UNet 去噪", [sys.executable, str(HERE / "04_denoising/infer_unet_v3.py"), "--input-root", str(corrected), "--output-root", str(final_images), "--model-path", str(HERE / "04_denoising/models/model_Unet_V3_best.pth")])

    masks = None
    if args.run_segmentation:
        masks = output / "05_mask"
        run("MobileSeg 指纹前景分割", [sys.executable, str(HERE / "02_segmentation/infer_mask.py"), "--input-root", str(final_images), "--output-root", str(masks), "--device", args.device])
    if args.run_quality:
        command = [sys.executable, str(HERE / "03_quality/quality_score.py"), "--input_dir", str(final_images), "--output_dir", str(output / "06_quality"), "--recursive", "--no_score_bins"]
        if masks is not None:
            command.extend(["--mask_dir", str(masks)])
        run("FFQ-Lite 质量评分", command)
    print(f"\n完成。最终图像目录：{final_images}")


if __name__ == "__main__":
    main()
