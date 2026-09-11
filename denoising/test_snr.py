import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# UNet
# ============================================================

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(
            64, 32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(
            32, 16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Conv2d(
            16,
            out_channels,
            3,
            padding=1
        )

        self.target_h = 128
        self.target_w = 128

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():

            if isinstance(
                m,
                (nn.Conv2d, nn.ConvTranspose2d)
            ):

                nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):

        _, _, h, w = x.shape

        pad_h = self.target_h - h
        pad_w = self.target_w - w

        x = F.pad(
            x,
            [
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2
            ],
            mode="reflect"
        )

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        d2 = self.up2(e3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        out = self.out(d1)

        # 裁剪回原尺寸
        out = out[
            :,
            :,
            pad_h // 2:pad_h // 2 + h,
            pad_w // 2:pad_w // 2 + w
        ]

        return out


# ============================================================
# SNR
# ============================================================

def calculate_snr(gt, img):
    """
    计算线性 SNR

    gt  : 干净图像
    img : noisy 或 denoised 图像

    SNR = signal_power / noise_power

    其中：

    signal_power = mean(gt^2)

    noise_power = mean((img - gt)^2)

    不使用 10*log10，因此返回的是线性 SNR。
    """

    gt = gt.astype(np.float64)
    img = img.astype(np.float64)

    # 信号功率
    signal_power = np.mean(gt ** 2)

    # 噪声功率
    noise = img - gt
    noise_power = np.mean(noise ** 2)

    # 防止除零
    if noise_power < 1e-12:
        return float("inf")

    snr = signal_power / noise_power

    return snr


# ============================================================
# 添加高斯噪声
# ============================================================

def add_gaussian_noise(img, sigma):
    """
    img   : [0,1]
    sigma : 高斯噪声标准差

    返回：
        noisy [0,1]
    """

    noise = np.random.normal(
        0,
        sigma,
        img.shape
    ).astype(np.float32)

    noisy = img + noise

    noisy = np.clip(
        noisy,
        0,
        1
    )

    return noisy


# ============================================================
# 测试
# ============================================================

def test_folder(
    input_dir,
    output_dir,
    model_path="model_Unet_V3_best.pth",
    sigma=0.05
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("使用设备:", device)
    print("噪声 sigma:", sigma)

    # ========================================================
    # 加载模型
    # ========================================================

    model = UNet(1, 1).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    # ========================================================
    # 输出目录
    # ========================================================

    gt_dir = os.path.join(
        output_dir,
        "GT"
    )

    noisy_dir = os.path.join(
        output_dir,
        "Noisy"
    )

    denoised_dir = os.path.join(
        output_dir,
        "Denoised"
    )

    os.makedirs(
        gt_dir,
        exist_ok=True
    )

    os.makedirs(
        noisy_dir,
        exist_ok=True
    )

    os.makedirs(
        denoised_dir,
        exist_ok=True
    )

    # ========================================================
    # 支持格式
    # ========================================================

    valid_ext = [
        ".bmp",
        ".jpg",
        ".png",
        ".jpeg"
    ]

    total = 0

    snr_noisy_list = []
    snr_denoised_list = []
    snr_ratio_list = []

    # ========================================================
    # 遍历数据
    # ========================================================

    for root, dirs, files in os.walk(input_dir):

        # 保持原来的目录结构
        rel_path = os.path.relpath(
            root,
            input_dir
        )

        gt_save_root = os.path.join(
            gt_dir,
            rel_path
        )

        noisy_save_root = os.path.join(
            noisy_dir,
            rel_path
        )

        denoised_save_root = os.path.join(
            denoised_dir,
            rel_path
        )

        os.makedirs(
            gt_save_root,
            exist_ok=True
        )

        os.makedirs(
            noisy_save_root,
            exist_ok=True
        )

        os.makedirs(
            denoised_save_root,
            exist_ok=True
        )

        for name in files:

            # =================================================
            # 判断图像格式
            # =================================================

            if os.path.splitext(name)[1].lower() not in valid_ext:
                continue

            input_path = os.path.join(
                root,
                name
            )

            # =================================================
            # 读取 GT
            # =================================================

            gt = cv2.imdecode(
                np.fromfile(
                    input_path,
                    dtype=np.uint8
                ),
                cv2.IMREAD_GRAYSCALE
            )

            if gt is None:
                print(
                    f"读取失败: {input_path}"
                )
                continue

            # =================================================
            # GT -> [0,1]
            # =================================================

            gt_float = (
                gt.astype(np.float32)
                / 255.0
            )

            # =================================================
            # 添加高斯噪声
            # =================================================

            noisy_float = add_gaussian_noise(
                gt_float,
                sigma
            )

            # =================================================
            # 计算加噪后的 SNR
            # =================================================

            snr_noisy = calculate_snr(
                gt_float,
                noisy_float
            )

            # =================================================
            # 转 Tensor
            # =================================================

            noisy_tensor = torch.from_numpy(
                noisy_float
            ).unsqueeze(0).unsqueeze(0).to(device)

            # =================================================
            # UNet 降噪
            # =================================================

            with torch.no_grad():

                pred_tensor = model(
                    noisy_tensor
                )

            # =================================================
            # Tensor -> numpy
            # =================================================

            pred = (
                pred_tensor
                .squeeze()
                .cpu()
                .numpy()
            )

            pred = np.clip(
                pred,
                0,
                1
            )

            # =================================================
            # 计算降噪后的 SNR
            # =================================================

            snr_denoised = calculate_snr(
                gt_float,
                pred
            )

            # =================================================
            # 计算 SNR 提升倍数
            #
            # ratio =
            # SNR_denoised / SNR_noisy
            # =================================================

            if np.isinf(snr_noisy):

                snr_ratio = float("inf")

            else:

                snr_ratio = (
                    snr_denoised
                    / snr_noisy
                )

            # =================================================
            # 保存图像
            # =================================================

            gt_save = gt

            noisy_save = (
                noisy_float * 255
            ).astype(np.uint8)

            denoised_save = (
                pred * 255
            ).astype(np.uint8)

            # GT
            cv2.imwrite(
                os.path.join(
                    gt_save_root,
                    name
                ),
                gt_save
            )

            # Noisy
            cv2.imwrite(
                os.path.join(
                    noisy_save_root,
                    name
                ),
                noisy_save
            )

            # Denoised
            cv2.imwrite(
                os.path.join(
                    denoised_save_root,
                    name
                ),
                denoised_save
            )

            # =================================================
            # 记录结果
            # =================================================

            snr_noisy_list.append(
                snr_noisy
            )

            snr_denoised_list.append(
                snr_denoised
            )

            snr_ratio_list.append(
                snr_ratio
            )

            total += 1

            # =================================================
            # 打印
            # =================================================

            print(
                f"{name} | "
                f"Noisy SNR: {snr_noisy:.6f} | "
                f"Denoised SNR: {snr_denoised:.6f} | "
                f"SNR提升倍数: {snr_ratio:.6f}x"
            )

    # ========================================================
    # 平均结果
    # ========================================================

    if total > 0:

        avg_noisy = np.mean(
            snr_noisy_list
        )

        avg_denoised = np.mean(
            snr_denoised_list
        )

        # 推荐：先计算每张图的比例，再求平均
        avg_ratio = np.mean(
            snr_ratio_list
        )

        # 另外计算整体 SNR 比值
        overall_ratio = (
            avg_denoised
            / avg_noisy
        )

        print("\n")
        print("=" * 70)
        print("测试完成")
        print("=" * 70)

        print(
            f"图像数量: {total}"
        )

        print(
            f"平均 Noisy SNR: "
            f"{avg_noisy:.6f}"
        )

        print(
            f"平均 Denoised SNR: "
            f"{avg_denoised:.6f}"
        )

        print(
            f"平均 SNR 提升倍数: "
            f"{avg_ratio:.6f}x"
        )

        print(
            f"整体 SNR 提升倍数: "
            f"{overall_ratio:.6f}x"
        )

        print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    input_dir = r"C:\Users\qwe\Desktop\finger\ysjz\select"

    output_dir = r"C:\Users\qwe\Desktop\finger\pre_deal\select_V3_SNR"

    test_folder(
        input_dir,
        output_dir,
        model_path="model_Unet_V3_best.pth",
        sigma=0.15
    )