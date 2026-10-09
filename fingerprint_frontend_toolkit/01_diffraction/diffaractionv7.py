# -*- coding: utf-8 -*-
"""V7：原始、已验证的衍射校正基线。

输入为 ``diff_final.py`` 生成并由 ``pic_final.py`` 转成 BMP 的五张
``diff-pair_N-Rgd=1421/1423/1425/1427/1429.bmp``。本版本直接对五张
``wo - wi`` 差分相位图执行：五相位复数解调 → 行列校正 → 衍射反卷积 →
全局输出相位优化。

它对应当前完整结果 ``pipeline_outputs/03_v7``，是 V8、V9 的对照基线。
建议通过 ``run_pipeline_v7.py`` 或命令行的 ``--input-mode direct-diff``
调用；不要修改这里的物理参数后覆盖已有 V7 输出。
"""

import os
import re
import argparse
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

FREQ = 12.0e6
SOUND_SPEED = 3800.0
DPI = 0.0254 / (75e-6)       # pitch = 75 um

BASE_THICKNESS = 700e-6
D_OFFSET = 0.0
TOTAL_THICKNESS = BASE_THICKNESS + D_OFFSET

DELTAT = 62500000.0
SAMPNUM = 5

# ============================================================
# 新增：Base / Raw 分别使用独立的初始相位
#
# 单位：弧度
#
# C_base = sum B_k * exp[-j(2*pi*f*t_k + BASE_INITIAL_PHASE)]
# C_raw  = sum R_k * exp[-j(2*pi*f*t_k + RAW_INITIAL_PHASE)]
# C_rgd  = C_base - C_raw
# ============================================================

BASE_INITIAL_PHASE = 0.0
RAW_INITIAL_PHASE = 0.0

# 是否根据 5 个多相位图像自动估计初始相位
USE_PHASE_ESTIMATION = True

# 参与相位估计的像素，按调制度从低到高筛选。
# 例如 30 表示保留调制度位于前 70% 的像素。
PHASE_MIN_MODULATION_PERCENTILE = 30.0

# 相位估计使用调制度加权。
# 调制度越高，该像素的相位估计可信度越高。
PHASE_WEIGHT_POWER = 2.0

# 如果自动估计失败，则使用下面的默认值
PHASE_FALLBACK_BASE = BASE_INITIAL_PHASE
PHASE_FALLBACK_RAW = RAW_INITIAL_PHASE

# 如果以后需要角度输入，可以使用：
# BASE_INITIAL_PHASE = np.deg2rad(10.0)
# RAW_INITIAL_PHASE = np.deg2rad(25.0)


# 行/列全局均值校正后的局部直流去除。每个非重叠 10×10 区域独立
# 减去复数均值；图像边缘不足一个块时，只使用该块内实际存在的像素。
DO_LOCAL_BLOCK_CORRECTION = True
LOCAL_BLOCK_ROW = 10
LOCAL_BLOCK_COL = 10

PAD_MODE = "symmetric"
BOUNDARY_MODE = "fill"

REQUIRED_RGDS = [1421, 1423, 1425, 1427, 1429]


# ============================================================
# 工具函数
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_rgd(path):
    m = re.search(r"Rgd=(\d+)", path, re.IGNORECASE)

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
    """
    按 Rgd 排序读取 5 张图。

    file_list:
        可以是：
        xxx.bmp
        或者不带 .bmp 的路径
    """

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
# 自动扫描 Base
#
# Base 文件格式：
# diff-pair_i-Rgd=1421.bmp
# diff-pair_i-Rgd=1423.bmp
# ...
#
# 每一个 pair 有自己的一组 Base。
# ============================================================

def collect_pairs(folder):

    pattern = re.compile(
        r"diff-pair_(\d+)-Rgd=(1421|1423|1425|1427|1429)",
        re.IGNORECASE
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
# 自动扫描 Raw
#
# 一个小文件夹中的所有 pair 共用同一组 Raw：
#
# raw-Rgd=1421.bmp
# raw-Rgd=1423.bmp
# raw-Rgd=1425.bmp
# raw-Rgd=1427.bmp
# raw-Rgd=1429.bmp
#
# 不再把 raw 和每个 pair 的 base 先做逐张相减。
# ============================================================

def collect_raw(folder):

    pattern = re.compile(
        r"raw-Rgd=(1421|1423|1425|1427|1429)",
        re.IGNORECASE
    )

    raw_files = {}

    for file in Path(folder).glob("*.bmp"):

        m = pattern.fullmatch(file.stem)

        if not m:
            continue

        rgd = int(m.group(1))

        raw_files[rgd] = str(file)

    return raw_files


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
    #
    # 现在不再直接对 Base-Raw 做逐张相减。
    #
    # 输入：
    #   imgs = 5 张 Base 或 5 张 Raw
    #
    # 输出：
    #   一个复数图像
    #
    # C = sum[
    #       I_k * exp(
    #           -j * (2*pi*f*t_k + initial_phase)
    #       )
    #     ]
    #
    # 也就是：
    #
    # Real(C) = sum[
    #     I_k * cos(theta_k)
    # ]
    #
    # Imag(C) = -sum[
    #     I_k * sin(theta_k)
    # ]
    # --------------------------------------------------------

    def estimate_initial_phase(
            self,
            imgs,
            deltaT,
            freq,
            modulation_percentile=30.0,
            weight_power=2.0):
        """
        根据 5 个多相位图像估计全局初始相位。

        假设每个像素满足：

            I_k = A + M*cos(delta*k + phi)

        其中：
            delta = 2*pi*freq/deltaT
            phi   = 待估计初始相位

        对每个像素做最小二乘：

            I_k = A + C*cos(delta*k) + S*sin(delta*k)

        则：

            C = M*cos(phi)
            S = -M*sin(phi)

        所以：

            phi = atan2(-S, C)

        最后对高调制度像素做加权圆均值，
        得到整个图像组的一个全局初始相位。
        """

        if len(imgs) != SAMPNUM:
            raise ValueError(
                f"相位估计需要 {SAMPNUM} 张图像，"
                f"实际得到 {len(imgs)} 张"
            )

        imgs_np = np.asarray(imgs, dtype=np.float64)

        if imgs_np.ndim != 3:
            raise ValueError(
                f"imgs 应为 [5,H,W]，实际 shape={imgs_np.shape}"
            )

        n, h, w = imgs_np.shape

        # 相邻采样之间的理论相位步进
        delta = 2.0 * np.pi * freq / deltaT

        k = np.arange(n, dtype=np.float64)

        cos_k = np.cos(delta * k)
        sin_k = np.sin(delta * k)

        # ----------------------------------------------------
        # 设计矩阵：
        #
        # [1, cos(delta*k), sin(delta*k)]
        #
        # 第 0 列用于吸收 DC / 平均亮度，
        # 避免 DC 项污染相位估计。
        # ----------------------------------------------------
        A = np.column_stack([
            np.ones(n, dtype=np.float64),
            cos_k,
            sin_k
        ])

        pinv_A = np.linalg.pinv(A)

        # [5, H*W]
        Y = imgs_np.reshape(n, -1)

        # [3, H*W]
        coeff = pinv_A @ Y

        # I = A + C*cos + S*sin
        C = coeff[1]
        S = coeff[2]

        # 调制度
        modulation = np.sqrt(C * C + S * S)

        # 按调制度去掉几乎没有相位信息的区域
        threshold = np.percentile(
            modulation,
            modulation_percentile
        )

        valid = (
            np.isfinite(modulation)
            & np.isfinite(C)
            & np.isfinite(S)
            & (modulation > threshold)
            & (modulation > 1e-8)
        )

        if np.count_nonzero(valid) < 10:
            raise ValueError(
                "有效相位像素太少，无法可靠估计初始相位"
            )

        # phi = atan2(-S, C)
        phase_map = np.arctan2(-S, C)

        # 调制度越高，权重越大
        weights = np.power(
            modulation[valid],
            weight_power
        )

        # 加权圆均值，避免 +pi / -pi 直接平均的问题
        complex_phase = np.sum(
            weights * np.exp(1j * phase_map[valid])
        )

        if np.abs(complex_phase) < 1e-12:
            raise ValueError(
                "相位圆均值接近 0，说明相位估计结果不稳定"
            )

        estimated_phase = np.angle(complex_phase)

        # ----------------------------------------------------
        # 计算相位一致性，作为诊断指标
        #
        # 1.0：所有有效像素相位非常一致
        # 接近 0：相位分布很分散
        # ----------------------------------------------------
        phase_coherence = (
            np.abs(complex_phase)
            / np.sum(weights)
        )

        return {
            "phase": float(estimated_phase),
            "phase_deg": float(np.degrees(estimated_phase)),
            "modulation_threshold": float(threshold),
            "valid_pixels": int(np.count_nonzero(valid)),
            "total_pixels": int(h * w),
            "phase_coherence": float(phase_coherence),
            "phase_map": phase_map.reshape(h, w),
            "modulation_map": modulation.reshape(h, w)
        }


    def load_and_demodulate(
            self,
            imgs,
            deltaT,
            freq,
            sampNum,
            initial_phase=0.0):

        if len(imgs) != sampNum:
            raise ValueError(
                f"图像数量错误：需要 {sampNum} 张，实际得到 {len(imgs)} 张"
            )

        complex_img = np.zeros_like(
            imgs[0],
            dtype=np.complex64
        )

        for k in range(sampNum):

            stime = k / deltaT

            theta = (
                2.0
                * np.pi
                * freq
                * stime
                + initial_phase
            )

            # 显式使用 cos / sin：
            #
            # exp(-j*theta)
            # = cos(theta) - j*sin(theta)
            #
            complex_img += (
                imgs[k]
                * (
                    np.cos(theta)
                    - 1j * np.sin(theta)
                )
            )

        return complex_img


    # --------------------------------------------------------
    # 新的 Base / Raw 差分流程
    #
    # Base 五张图：
    #   B1421, B1423, B1425, B1427, B1429
    #
    # Raw 五张图：
    #   R1421, R1423, R1425, R1427, R1429
    #
    # 分别进行五步复数累加：
    #
    # C_base =
    #   sum B_k * exp[-j(theta_k + phi_base)]
    #
    # C_raw =
    #   sum R_k * exp[-j(theta_k + phi_raw)]
    #
    # 最后：
    #
    # C_rgd = C_base - C_raw
    #
    # 注意：
    # 这里是在“复数域”做 Base - Raw，
    # 而不是先对每一个 Rgd 做：
    #
    # B_k - R_k
    #
    # 再统一做一次五步累加。
    # --------------------------------------------------------

    def demodulate_base_and_raw(
            self,
            base_imgs,
            raw_imgs,
            deltaT,
            freq,
            sampNum,
            base_initial_phase,
            raw_initial_phase,
            estimate_phase=True):

        # ----------------------------------------------------
        # Base 初始相位
        # ----------------------------------------------------
        if estimate_phase:
            try:
                base_phase_info = self.estimate_initial_phase(
                    imgs=base_imgs,
                    deltaT=deltaT,
                    freq=freq,
                    modulation_percentile=PHASE_MIN_MODULATION_PERCENTILE,
                    weight_power=PHASE_WEIGHT_POWER
                )

                base_initial_phase_used = base_phase_info["phase"]

            except Exception as e:
                print(f"Base 自动相位估计失败: {e}")
                print(
                    f"回退到 Base 默认初始相位: "
                    f"{base_initial_phase} rad"
                )

                base_phase_info = None
                base_initial_phase_used = base_initial_phase

        else:
            base_phase_info = None
            base_initial_phase_used = base_initial_phase

        # ----------------------------------------------------
        # Raw 初始相位
        # ----------------------------------------------------
        if estimate_phase:
            try:
                raw_phase_info = self.estimate_initial_phase(
                    imgs=raw_imgs,
                    deltaT=deltaT,
                    freq=freq,
                    modulation_percentile=PHASE_MIN_MODULATION_PERCENTILE,
                    weight_power=PHASE_WEIGHT_POWER
                )

                raw_initial_phase_used = raw_phase_info["phase"]

            except Exception as e:
                print(f"Raw 自动相位估计失败: {e}")
                print(
                    f"回退到 Raw 默认初始相位: "
                    f"{raw_initial_phase} rad"
                )

                raw_phase_info = None
                raw_initial_phase_used = raw_initial_phase

        else:
            raw_phase_info = None
            raw_initial_phase_used = raw_initial_phase

        # ----------------------------------------------------
        # Base 五步复数累加
        # ----------------------------------------------------
        base_complex = self.load_and_demodulate(
            imgs=base_imgs,
            deltaT=deltaT,
            freq=freq,
            sampNum=sampNum,
            initial_phase=base_initial_phase_used
        )

        # ----------------------------------------------------
        # Raw 五步复数累加
        # ----------------------------------------------------
        raw_complex = self.load_and_demodulate(
            imgs=raw_imgs,
            deltaT=deltaT,
            freq=freq,
            sampNum=sampNum,
            initial_phase=raw_initial_phase_used
        )

        # ----------------------------------------------------
        # 最终：
        #
        # C_Rgd = C_Base - C_Raw
        # ----------------------------------------------------
        rgd_complex = base_complex - raw_complex

        return (
            base_complex,
            raw_complex,
            rgd_complex,
            base_initial_phase_used,
            raw_initial_phase_used,
            base_phase_info,
            raw_phase_info
        )


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


    def local_block_mean_correction(
            self,
            img,
            block_height=10,
            block_width=10):
        """按非重叠局部块去除复数均值，不对边缘做零填充或镜像填充。"""
        if block_height <= 0 or block_width <= 0:
            raise ValueError("block_height 和 block_width 必须为正数")

        out = img.astype(np.complex64, copy=True)
        height, width = out.shape[:2]
        for row_start in range(0, height, block_height):
            row_end = min(row_start + block_height, height)
            for col_start in range(0, width, block_width):
                col_end = min(col_start + block_width, width)
                block = out[row_start:row_end, col_start:col_end]
                out[row_start:row_end, col_start:col_end] = block - np.mean(block)
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

        F_h1 = fft2(h1)

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

        # 原代码这里存在 H,W = 0,0 的覆盖问题。
        # 这里直接使用真实图像尺寸。

        H, W = complex_img.shape

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

    global INPUT_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="对五相位 Base/Raw BMP 执行 V7 相位估计与衍射校正。"
    )
    parser.add_argument("--input-root", default=INPUT_DIR)
    parser.add_argument("--output-root", default=OUTPUT_DIR)
    parser.add_argument("--folder", action="append", default=[], help="仅处理指定一级数据文件夹；可重复指定。")
    parser.add_argument(
        "--input-mode",
        choices=("direct-diff", "base-raw"),
        default="direct-diff",
        help=(
            "direct-diff: 直接处理 diff_final/pic_final 的五相位差分图；"
            "base-raw: 处理另一种需额外 raw-Rgd 文件的 Base/Raw 数据格式。"
        ),
    )
    args = parser.parse_args()
    INPUT_DIR = args.input_root
    OUTPUT_DIR = args.output_root

    ensure_dir(OUTPUT_DIR)

    subfolders = [
        p for p in Path(INPUT_DIR).iterdir()
        if p.is_dir() and (not args.folder or p.name in args.folder)
    ]

    print(f"发现 {len(subfolders)} 个数据文件夹")

    for subfolder in sorted(subfolders):

        folder_name = subfolder.name

        print("\n" + "=" * 80)
        print("处理文件夹:", folder_name)
        print("=" * 80)

        # ----------------------------------------------------
        # 1. 找这个小文件夹中的 Base
        # ----------------------------------------------------

        pairs = collect_pairs(subfolder)

        print(f"发现 {len(pairs)} 个 pair")

        raw_imgs = None
        if args.input_mode == "base-raw":
            raw_files = collect_raw(subfolder)
            missing_raw = [
                r for r in REQUIRED_RGDS
                if r not in raw_files
            ]
            if missing_raw:
                print(f"文件夹 {folder_name} 缺少 Raw 文件: {missing_raw}")
                continue
            raw_file_list = [raw_files[r] for r in REQUIRED_RGDS]
            raw_rgds, raw_imgs, raw_paths = read_images(raw_file_list)
            print("共用 Raw:")
            for rgd, path in zip(raw_rgds, raw_paths):
                print(f"  Rgd={rgd}: {path}")

        # ----------------------------------------------------
        # 输出目录
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 3. 逐个 pair 处理
        # ----------------------------------------------------

        for pair_id in sorted(pairs.keys()):

            pair_files = pairs[pair_id]

            missing_base = [
                r
                for r in REQUIRED_RGDS
                if r not in pair_files
            ]

            if missing_base:

                print(
                    f"pair_{pair_id} 缺少 Base 文件: "
                    f"{missing_base}"
                )

                continue

            try:

                print("\n" + "-" * 70)
                print(f"处理 pair_{pair_id}")
                print("-" * 70)

                # ------------------------------------------------
                # Base 五张图
                # ------------------------------------------------

                base_file_list = [
                    pair_files[1421],
                    pair_files[1423],
                    pair_files[1425],
                    pair_files[1427],
                    pair_files[1429]
                ]

                base_rgds, base_imgs, base_paths = read_images(
                    base_file_list
                )

                print("Base 五个 Rgd:")
                for rgd, path in zip(base_rgds, base_paths):
                    print(f"  Rgd={rgd}: {path}")

                if args.input_mode == "direct-diff":
                    phase_info = None
                    phase_used = PHASE_FALLBACK_BASE
                    if USE_PHASE_ESTIMATION:
                        try:
                            phase_info = calibrator.estimate_initial_phase(
                                imgs=base_imgs,
                                deltaT=DELTAT,
                                freq=FREQ,
                                modulation_percentile=PHASE_MIN_MODULATION_PERCENTILE,
                                weight_power=PHASE_WEIGHT_POWER,
                            )
                            phase_used = phase_info["phase"]
                        except Exception as e:
                            print(f"差分图自动相位估计失败，使用默认相位: {e}")
                    complex_image_I1 = calibrator.load_and_demodulate(
                        imgs=base_imgs,
                        deltaT=DELTAT,
                        freq=FREQ,
                        sampNum=SAMPNUM,
                        initial_phase=phase_used,
                    )
                    print(
                        "差分图五相位解调，初始相位 = "
                        f"{phase_used:.8f} rad ({np.degrees(phase_used):.4f} deg)"
                    )
                    if phase_info is not None:
                        print(
                            "相位一致性 = "
                            f"{phase_info['phase_coherence']:.4f}, 有效像素 = "
                            f"{phase_info['valid_pixels']}/{phase_info['total_pixels']}"
                        )
                else:
                    (
                        _base_complex,
                        _raw_complex,
                        complex_image_I1,
                        base_phase_used,
                        raw_phase_used,
                        _base_phase_info,
                        _raw_phase_info,
                    ) = calibrator.demodulate_base_and_raw(
                        base_imgs=base_imgs,
                        raw_imgs=raw_imgs,
                        deltaT=DELTAT,
                        freq=FREQ,
                        sampNum=SAMPNUM,
                        base_initial_phase=PHASE_FALLBACK_BASE,
                        raw_initial_phase=PHASE_FALLBACK_RAW,
                        estimate_phase=USE_PHASE_ESTIMATION,
                    )
                    print(
                        "Base/Raw 五步解调后在复数域相减；"
                        f"Base={base_phase_used:.8f}, Raw={raw_phase_used:.8f} rad"
                    )

                # ------------------------------------------------
                # 后续处理保持原流程
                # ------------------------------------------------

                complex_image_I1 = (
                    calibrator.global_rowcol_correction(complex_image_I1)
                )
                if DO_LOCAL_BLOCK_CORRECTION:
                    complex_image_I1 = calibrator.local_block_mean_correction(
                        complex_image_I1,
                        block_height=LOCAL_BLOCK_ROW,
                        block_width=LOCAL_BLOCK_COL,
                    )

                # ------------------------------------------------
                # Diffraction deconvolution
                # ------------------------------------------------

                deblurred_complex_I2 = (
                    calibrator.deconvolve_diffraction_optimized(
                        complex_img=complex_image_I1,
                        thickness=TOTAL_THICKNESS
                    )
                )

                # ------------------------------------------------
                # 最终输出
                # ------------------------------------------------

                final_path = os.path.join(
                    out_folder,
                    f"pair_{pair_id}.bmp"
                )

                calibrator.optimize_phase(
                    complex_img=deblurred_complex_I2,
                    save_path=final_path
                )

                print(
                    f"完成 -> "
                    f"{folder_name}/pair_{pair_id}.bmp"
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
