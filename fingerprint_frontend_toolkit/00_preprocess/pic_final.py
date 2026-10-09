"""预处理第 3 步：将 diff CSV 转为 V7 所需的五相位 BMP。

V7 推荐 ``--normalization per-pair``：同一 pair 的五个 Rgd 共用一个
强度范围，避免每张图单独拉伸而破坏五相位解调的相对幅值。V8/V9 直接读
原始 CSV，因此不经过此文件。
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import re

# =====================================
# 输入总目录
# =====================================
INPUT_ROOT = r"C:\Users\qwe\Desktop\wet_diff"

# =====================================
# 输出总目录
# =====================================
OUTPUT_ROOT = r"C:\Users\qwe\Desktop\wet_pic"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_to_uint8(img):

    img = img.astype(np.float32)

    mn = np.min(img)
    mx = np.max(img)

    img = (img - mn) / (mx - mn + 1e-8)

    return (img * 255).clip(0, 255).astype(np.uint8)


def read_csv_image(csv_path):
    try:
        df = pd.read_csv(csv_path, header=None, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, header=None, encoding="gbk", low_memory=False)
    return df.to_numpy(dtype=np.float32)


def normalize_with_range(img, mn, mx):
    return ((img.astype(np.float32) - mn) / (mx - mn + 1e-8) * 255).clip(0, 255).astype(np.uint8)


def process_folder(input_folder, output_folder, normalization):

    ensure_dir(output_folder)

    csv_files = sorted(
        Path(input_folder).glob("*.csv"),
        key=lambda x: x.name.lower()
    )

    print(f"\n处理文件夹: {os.path.basename(input_folder)}")
    print(f"找到 {len(csv_files)} 个CSV")

    success = 0

    arrays = {}
    for csv_path in csv_files:
        try:
            arrays[csv_path] = read_csv_image(csv_path)
        except Exception as e:
            print(f"失败读取: {csv_path.name}: {e}")

    # Five phase observations must share a scale before V7 demodulation.  The
    # legacy per-image branch remains the default for reproducibility.
    ranges = {}
    if normalization == "per-pair":
        groups = {}
        pattern = re.compile(r"diff-pair_(\d+)-Rgd=(1421|1423|1425|1427|1429)$", re.I)
        for path, image in arrays.items():
            match = pattern.match(path.stem)
            if match:
                groups.setdefault(match.group(1), []).append((path, image))
        for pair_id, items in groups.items():
            if len(items) == 5:
                combined = np.concatenate([image.ravel() for _, image in items])
                ranges.update({path: (float(combined.min()), float(combined.max())) for path, _ in items})
            else:
                print(f"pair_{pair_id} 相位图不完整，回退逐图归一化")

    for csv_path, pixel_array in arrays.items():
        try:
            if csv_path in ranges:
                bmp_array = normalize_with_range(pixel_array, *ranges[csv_path])
            else:
                bmp_array = normalize_to_uint8(pixel_array)

            bmp_name = (
                csv_path.stem + ".bmp"
            )

            bmp_path = os.path.join(
                output_folder,
                bmp_name
            )

            cv2.imwrite(
                bmp_path,
                bmp_array
            )

            print(f"完成: {bmp_name}")

            success += 1

        except Exception as e:
            print(
                f"失败: {csv_path.name}"
            )
            print(e)

    print(f"成功转换 {success} 个文件")


def main():

    parser = argparse.ArgumentParser(
        description="将差分 CSV 转为 BMP；V7 推荐同一五相位 pair 联合归一化。"
    )
    parser.add_argument("--input-root", default=INPUT_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--normalization", choices=("per-image", "per-pair"), default="per-image")
    parser.add_argument("--folder", action="append", default=[], help="仅处理指定一级数据文件夹；可重复指定。")
    args = parser.parse_args()

    ensure_dir(args.output_root)

    for folder_name in sorted(os.listdir(args.input_root)):

        if args.folder and folder_name not in args.folder:
            continue

        input_subfolder = os.path.join(
            args.input_root,
            folder_name
        )

        if not os.path.isdir(
                input_subfolder):
            continue

        output_subfolder = os.path.join(
            args.output_root,
            folder_name
        )

        process_folder(
            input_subfolder,
            output_subfolder,
            args.normalization,
        )

    print("\n全部完成")
    print("输出目录:")
    print(args.output_root)


if __name__ == "__main__":
    main()
