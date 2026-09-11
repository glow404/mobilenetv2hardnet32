import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# 去掉报错的环境变量
# os.environ['OMP_NUM_THREADS'] = '1'

# ============ 修复完整的 UNet ============
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()


        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels,16,3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16,16,3,padding=1),
            nn.ReLU(inplace=True)
        )


        self.pool1 = nn.MaxPool2d(2)



        self.enc2 = nn.Sequential(
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU(inplace=True)
        )


        self.pool2 = nn.MaxPool2d(2)



        self.enc3 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(inplace=True)
        )


        # Decoder

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )


        self.dec2 = nn.Sequential(
            nn.Conv2d(64,32,3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU(inplace=True)
        )


        self.up1 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2
        )


        self.dec1 = nn.Sequential(
            nn.Conv2d(32,16,3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16,16,3,padding=1),
            nn.ReLU(inplace=True)
        )


        self.out = nn.Conv2d(
            16,
            out_channels,
            3,
            padding=1
        )


        self.target_h=128
        self.target_w=128



        self._init_weights()



    def _init_weights(self):

        for m in self.modules():

            if isinstance(
                m,
                (nn.Conv2d,
                 nn.ConvTranspose2d)
            ):

                nn.init.kaiming_normal_(
                    m.weight
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)



    def forward(self,x):

        _,_,h,w=x.shape


        pad_h=self.target_h-h
        pad_w=self.target_w-w


        x=F.pad(
            x,
            [
                pad_w//2,
                pad_w-pad_w//2,
                pad_h//2,
                pad_h-pad_h//2
            ],
            mode="reflect"
        )



        e1=self.enc1(x)

        e2=self.enc2(
            self.pool1(e1)
        )


        e3=self.enc3(
            self.pool2(e2)
        )



        d2=self.up2(e3)


        d2=torch.cat(
            [
                d2,
                e2
            ],
            dim=1
        )


        d2=self.dec2(d2)



        d1=self.up1(d2)


        d1=torch.cat(
            [
                d1,
                e1
            ],
            dim=1
        )


        d1=self.dec1(d1)



        out=self.out(d1)



        # 裁剪回100×110

        out=out[
            :,
            :,
            pad_h//2:pad_h//2+h,
            pad_w//2:pad_w//2+w
        ]


        return out




def test_folder(input_dir, output_dir, model_path="model_Unet_V6_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 加载模型 =====
    model = UNet(1, 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ===== 支持的格式 ====
    valid_ext = [".bmp", ".jpg", ".png", ".jpeg"]

    total = 0

    # ===== 递归遍历 =====
    for root, dirs, files in os.walk(input_dir):
        # 计算对应输出路径（保持目录结构）
        rel_path = os.path.relpath(root, input_dir)
        save_root = os.path.join(output_dir, rel_path)
        os.makedirs(save_root, exist_ok=True)

        for name in files:
            if os.path.splitext(name)[1].lower() not in valid_ext:
                continue

            input_path = os.path.join(root, name)

            # ---- 读取 ----
            img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"读取失败: {input_path}")
                continue

            # ---- 预处理 ----
            img_np = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

            # ---- 推理 ----
            with torch.no_grad():
                pred_tensor = model(img_tensor)

            # ---- 转回图像 ----
            pred_np = pred_tensor.squeeze().cpu().numpy()
            pred_np = np.clip(pred_np, 0, 1) * 255
            pred_np = pred_np.astype(np.uint8)

            # ---- 保存（保持子目录结构）----
            save_path = os.path.join(save_root, name)
            cv2.imwrite(save_path, pred_np)

            total += 1
            print(f"已处理: {input_path}")

    print(f"全部完成！共处理 {total} 张图像")


if __name__ == "__main__":
    input_dir = r"C:/Users/qwe/Desktop/new_data"
    output_dir = r"C:/Users/qwe/Desktop/new_data_V6"

    test_folder(input_dir, output_dir)