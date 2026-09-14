# -*- coding: utf-8 -*-

import os
import re
import cv2
import numpy as np

from pathlib import Path
from collections import defaultdict

from scipy.fft import fft2
from scipy import signal


# ============================================================
# 参数
# ============================================================

INPUT_DIR = r"C:\Users\qwe\Desktop\wet_pic"
OUTPUT_DIR = r"C:\Users\qwe\Desktop\finger\ysjz\wet_new"

FREQ = 11.364e6

SOUND_SPEED = 4000.0

DPI = 338.0

BASE_THICKNESS = 0.0010
D_OFFSET = 0.0
TOTAL_THICKNESS = BASE_THICKNESS + D_OFFSET

DELTAT = 62500000.0

SAMPNUM = 5

DO_BLOCK_CORRECTION = True

BLOCK_COL = 10
BLOCK_ROW = 10

PAD_MODE = "symmetric"

BOUNDARY_MODE = "fill"


# ============================================================
# 工具函数
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_rgd(path):

    m = re.search(r"Rgd=(\d+)", path)

    if not m:
        raise ValueError(f"Cannot extract Rgd from: {path}")

    return int(m.group(1))


def normalize_to_uint8(img):

    img = np.real(img).astype(np.float32)

    mn = np.min(img)
    mx = np.max(img)

    out = (img - mn) / (mx - mn + 1e-8)

    return (out * 255.0).clip(0, 255).astype(np.uint8)


def read_images(file_list):

    items = []

    for p in file_list:

        bmp_path = p if p.lower().endswith(".bmp") else p + ".bmp"

        img = cv2.imread(
            bmp_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            raise ValueError(f"读取失败: {bmp_path}")

        rgd = extract_rgd(p)

        img = img.astype(np.float32) / 255.0

        items.append(
            (rgd, img, bmp_path)
        )

    items.sort(key=lambda x: x[0])

    rgds = [x[0] for x in items]
    imgs = [x[1] for x in items]
    paths = [x[2] for x in items]

    return rgds, imgs, paths


# ============================================================
# 自动扫描 pair
# ============================================================

def collect_pairs(folder):

    pattern = re.compile(
        r"diff-pair_(\d+)-Rgd=(1421|1423|1425|1427|1429)"
    )

    groups = defaultdict(dict)

    for file in Path(folder).glob("*.bmp"):

        m = pattern.search(file.stem)

        if not m:
            continue

        pair_id = int(m.group(1))
        rgd = int(m.group(2))

        groups[pair_id][rgd] = str(file)

    return groups
# ============================================================
# DiffractionCalibrator
# ============================================================

class DiffractionCalibrator:

    def __init__(
            self,
            frequency=FREQ,
            sound_speed=SOUND_SPEED,
            dpi=DPI):

        self.f = frequency
        self.c = sound_speed
        self.pixel_pitch = 0.0254 / dpi
        self.dpi = dpi
        self.cached_kernel = None

        self.cached_thickness = None

        self.last_kernel_created = False


    def _save_image(self, img, path):

        cv2.imwrite(
            path,
            normalize_to_uint8(img)
        )

    # --------------------------------------------------------
    # Step1
    # --------------------------------------------------------

    def load_and_demodulate(
            self,
            imgs,
            deltaT,
            freq,
            sampNum):

        complex_img = np.zeros_like(
            imgs[0],
            dtype=np.complex64
        )

        for k in range(sampNum):

            stime = k * 1 / deltaT

            phase_shift = (
                -1j
                * 2
                * np.pi
                * freq
                * stime
            )

            complex_img += (
                imgs[k]
                * np.exp(phase_shift)
            )

        return complex_img

    # --------------------------------------------------------
    # Block correction
    # --------------------------------------------------------

    def block_column_correction(
            self,
            img,
            column=10):

        corrected_img = np.zeros_like(img)

        rows, cols = img.shape

        for i in range(0, cols, column):

            block = img[:, i:i + column]

            block_mean = np.mean(block)

            corrected_img[:, i:i + column] = (
                block - block_mean
            )

        return corrected_img

    def block_row_correction(
            self,
            img,
            rownum=10):

        corrected_img = np.zeros_like(img)

        rows, cols = img.shape

        for i in range(0, rows, rownum):

            block = img[i:i + rownum, :]

            block_mean = np.mean(block)

            corrected_img[i:i + rownum, :] = (
                block - block_mean
            )

        return corrected_img

    # --------------------------------------------------------
    # Global correction
    # --------------------------------------------------------

    def global_rowcol_correction(
            self,
            img):

        out = img.astype(
            np.complex64,
            copy=True
        )

        row_mean = np.mean(
            out,
            axis=1,
            keepdims=True
        )

        out = out - row_mean

        col_mean = np.mean(
            out,
            axis=0,
            keepdims=True
        )

        out = out - col_mean

        return out

    # --------------------------------------------------------
    # Step2
    # --------------------------------------------------------

    def build_inverse_kernel(
            self,
            thickness):

        PH = 49
        PW = 49

        y = (
            np.arange(PH)
            - PH // 2
        ) * self.pixel_pitch

        x = (
            np.arange(PW)
            - PW // 2
        ) * self.pixel_pitch

        XX, YY = np.meshgrid(
            x,
            y
        )

        R = (
            thickness
            + np.sqrt(
                XX ** 2
                + YY ** 2
                + thickness ** 2
            )
        )

        h1 = np.exp(
            -1j
            * 2
            * np.pi
            * self.f
            * R
            / self.c
        )

        F_h1 = fft2(
            h1
        )

        H_inv = (
            np.conj(F_h1)
            / (
                np.abs(F_h1) ** 2
                + 1e6
            )
        )

        kernel = np.fft.ifft2(
            H_inv
        )

        return kernel

    def deconvolve_diffraction_optimized(
            self,
            complex_img,
            thickness):

        H, W = complex_img.shape

        # 保留原来的处理逻辑
        H, W = 0, 0

        pad_h = H // 2
        pad_w = W // 2

        img_padded = np.pad(
            complex_img,
            (
                (pad_h, pad_h),
                (pad_w, pad_w)
            ),
            mode=PAD_MODE
        )

        self.last_kernel_created = False

        if (
            self.cached_kernel is None
            or self.cached_thickness != thickness
        ):

            self.cached_kernel = (
                self.build_inverse_kernel(
                    thickness
                )
            )

            self.cached_thickness = thickness

            self.last_kernel_created = True

        result_centered = signal.fftconvolve(
            img_padded,
            self.cached_kernel,
            mode="same"
        )

        result = result_centered[
                 pad_h:
                 pad_h + complex_img.shape[0],

                 pad_w:
                 pad_w + complex_img.shape[1]
                 ]

        return result

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    def optimize_phase(
            self,
            complex_img,
            save_path):

        S = np.sum(
            complex_img ** 2
        )

        if np.abs(S) < 1e-12:

            phase_corr = 1.0 + 0j

        else:

            phase_corr = np.sqrt(
                np.abs(S) / S
            )

        final = np.real(
            complex_img * phase_corr
        )

        if np.mean(final) > 0:
            final = -final

        self._save_image(
            final,
            save_path
        )

        return final
# ============================================================
# main
# ============================================================

def main():

    ensure_dir(OUTPUT_DIR)

    subfolders = [
        p for p in Path(INPUT_DIR).iterdir()
        if p.is_dir()
    ]

    print(f"发现 {len(subfolders)} 个数据文件夹")

    required_rgds = [
        1421,
        1423,
        1425,
        1427,
        1429
    ]

    for subfolder in sorted(subfolders):

        folder_name = subfolder.name

        print("\n" + "=" * 80)
        print("处理文件夹:", folder_name)
        print("=" * 80)

        pairs = collect_pairs(subfolder)

        print(f"发现 {len(pairs)} 个 pair")

        out_folder = os.path.join(
            OUTPUT_DIR,
            folder_name
        )

        ensure_dir(out_folder)

        calibrator = DiffractionCalibrator(
            frequency=FREQ,
            sound_speed=SOUND_SPEED,
            dpi=DPI
        )

        for pair_id in sorted(pairs.keys()):

            pair_files = pairs[pair_id]

            missing = [
                r
                for r in required_rgds
                if r not in pair_files
            ]

            if missing:

                print(
                    f"pair_{pair_id} 缺少文件: {missing}"
                )

                continue

            try:

                print(
                    f"\n处理 pair_{pair_id}"
                )

                file_list = [

                    pair_files[1421],

                    pair_files[1423],

                    pair_files[1425],

                    pair_files[1427],

                    pair_files[1429]

                ]

                rgds, imgs, paths = read_images(
                    file_list
                )

                complex_image_I1 = (
                    calibrator.load_and_demodulate(
                        imgs=imgs,
                        deltaT=DELTAT,
                        freq=FREQ,
                        sampNum=SAMPNUM
                    )
                )

                if DO_BLOCK_CORRECTION:

                    complex_image_I1 = (
                        calibrator.global_rowcol_correction(
                            complex_image_I1,
                        )
                    )


                else:

                    complex_image_I1 = (
                        calibrator.global_rowcol_correction(
                            complex_image_I1
                        )
                    )

                deblurred_complex_I2 = (
                    calibrator.deconvolve_diffraction_optimized(
                        complex_img=complex_image_I1,
                        thickness=TOTAL_THICKNESS
                    )
                )

                final_path = os.path.join(
                    out_folder,
                    f"pair_{pair_id}.bmp"
                )

                calibrator.optimize_phase(
                    complex_img=deblurred_complex_I2,
                    save_path=final_path
                )

                print(
                    f"完成 -> {folder_name}/pair_{pair_id}.bmp"
                )

            except Exception as e:

                print(
                    f"pair_{pair_id} 失败: {e}"
                )

    print("\n全部完成")
    print("输出目录:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()