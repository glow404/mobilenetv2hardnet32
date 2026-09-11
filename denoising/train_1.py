import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)
import torch.optim as optim
from PIL import Image, ImageFilter

from denoising.HardNet import HardNet
#from pytorch_msssim import ssim
from runtime import (
    build_sift,
    detect_sift_keypoints,
    patchable_keypoints,
)
import HardNet as hardmodel




class GTFeatureExtractor:

    def __init__(self, config):
        self.config = config
        self.sift = build_sift(config)

    def detect_keypoints(self, image):
        """
        image:
            numpy uint8
            H×W

        return
            list[cv2.KeyPoint]
        """

        keypoints = detect_sift_keypoints(
            image,
            self.sift,
            self.config
        )

        max_keypoints = int(
            self.config.get(
                "keypoint_filter",
                {}
            ).get(
                "max_keypoints",
                400
            )
        )

        if max_keypoints > 0 and len(keypoints) > max_keypoints:

            order = np.argsort(
                [
                    kp.response
                    for kp in keypoints
                ]
            )[::-1][:max_keypoints]

            keypoints = [
                keypoints[int(i)]
                for i in order
            ]

        return keypoints

    def extract_patches(
            self,
            image,
            keypoints,
    ):
        """
        image:
            numpy uint8
            H×W

        keypoints:
            list[cv2.KeyPoint]

        return

            selected_keypoints

            patches
                N×32×32

            selected_indices
        """

        patch_cfg = self.config.get(
            "patch",
            {}
        )

        selected_keypoints, patches, selected_indices = patchable_keypoints(

            image,

            keypoints,

            {
                "patch": {

                    "crop_size":
                        int(
                            patch_cfg.get(
                                "crop_size",
                                64
                            )
                        ),

                    "out_size":
                        int(
                            patch_cfg.get(
                                "out_size",
                                32
                            )
                        ),

                    "min_overlap_ratio":
                        float(
                            patch_cfg.get(
                                "min_overlap_ratio",
                                0.55
                            )
                        ),

                    "normalize":
                        bool(
                            patch_cfg.get(
                                "normalize",
                                True
                            )
                        ),
                }
            }
        )

        return (
            selected_keypoints,
            patches,
            selected_indices
        )

# 解决 libgomp 警告
os.environ['OMP_NUM_THREADS'] = '1'

# ======================================
# 1. 合成退化函数
# ======================================
def synthetic_degradation(
    img_pil,
    add_noise=True,
    add_blur=True,
    add_downscale=False,
    add_jpeg=False,
    noise_type="gaussian",
    scale=2
):
    img = np.array(img_pil).astype(np.float32) / 255.
    h, w = img.shape[:2]

    if add_noise:
        if noise_type == "gaussian":
            sigma = random.uniform(0.01, 0.15)
            noise = np.random.normal(0, sigma, img.shape)
        elif noise_type == "uniform":
            noise = np.random.uniform(-0.05, 0.05, img.shape)
        elif noise_type == "poisson":
            noise = np.random.poisson(img.clip(0,1)*255)/255 - img
        img = (img + noise).clip(0, 1)

    if add_blur and random.random() > 0.3:
        img_pil = Image.fromarray((img*255).astype(np.uint8))
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.5)))
        img = np.array(img_pil).astype(np.float32)/255

    if add_downscale:
        h_lq, w_lq = h // scale, w // scale
        img_pil = Image.fromarray((img*255).astype(np.uint8))
        img_pil = img_pil.resize((w_lq, h_lq), Image.BICUBIC)
        img_pil = img_pil.resize((w, h), Image.BICUBIC)
        img = np.array(img_pil).astype(np.float32)/255

    if add_jpeg and random.random() > 0.4:
        quality = random.randint(20, 60)
        img_pil = Image.fromarray((img*255).astype(np.uint8))
        img_pil.save("tmp.jpg", quality=quality)
        img = np.array(Image.open("tmp.jpg")).astype(np.float32)/255

    img_lq = Image.fromarray((img.clip(0,1)*255).astype(np.uint8))
    return img_lq

# ======================================
# 2. UNet 模型（固定输入 100x110）
# ======================================
class TinyUNet(nn.Module):

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

# ======================================
# 3. 数据集（彻底禁止旋转，保证尺寸 100x110）
# ======================================
class BMPDataset(Dataset):

    def __init__(
            self,
            target_dir,
            feature_extractor,
            train=True
    ):

        self.target_dir = target_dir

        self.files = [
            f for f in os.listdir(target_dir)
            if f.endswith('.png')
        ]

        self.train = train


        # ==========================
        # 保存SIFT关键点
        # ==========================

        self.feature_extractor = feature_extractor

        self.keypoint_cache = {}


        print("正在提取GT关键点...")


        for filename in self.files:

            path = os.path.join(
                self.target_dir,
                filename
            )

            img = self.read_img(path)


            # cv2.KeyPoint列表
            keypoints = self.feature_extractor.detect_keypoints(
                img
            )


            # ======================
            # 转换成(x,y)
            # 防止DataLoader无法collate
            # ======================

            xy_points = []


            for kp in keypoints:

                xy_points.append(
                    (
                        kp.pt[0],
                        kp.pt[1]
                    )
                )


            self.keypoint_cache[filename] = xy_points



        print(
            "关键点提取完成"
        )



    def __len__(self):

        return len(self.files)



    def read_img(self, path):

        img = cv2.imdecode(
            np.fromfile(
                path,
                dtype=np.uint8
            ),
            cv2.IMREAD_GRAYSCALE
        )

        return img



    def __getitem__(self, idx):


        filename = self.files[idx]


        target_path = os.path.join(
            self.target_dir,
            filename
        )


        # ======================
        # GT
        # ======================

        img_gt = self.read_img(
            target_path
        )



        # ======================
        # synthetic degradation
        # ======================

        img_pil = Image.fromarray(
            img_gt
        )


        img_in_pil = synthetic_degradation(
            img_pil
        )


        img_in = np.array(
            img_in_pil
        )



        # ======================
        # normalize
        # ======================

        img_in = (
            img_in.astype(np.float32)
            /
            255.0
        )


        img_gt = (
            img_gt.astype(np.float32)
            /
            255.0
        )



        # ======================
        # channel维度
        # [1,H,W]
        # ======================

        img_in = np.expand_dims(
            img_in,
            axis=0
        )


        img_gt = np.expand_dims(
            img_gt,
            axis=0
        )



        # ======================
        # 返回关键点
        # ======================

        keypoints = self.keypoint_cache[
            filename
        ]



        return (
            torch.tensor(img_in),
            torch.tensor(img_gt),
            keypoints
        )

# ======================================
# 4. 损失函数
# ======================================
class RingL1Loss(nn.Module):

    def __init__(
            self,
            hardnet_checkpoint,
            border=8,
            template_weight=0.1,
            gradient_weight=0.2
    ):

        super().__init__()


        self.border = border

        self.template_weight = template_weight

        self.gradient_weight = gradient_weight



        # =========================
        # HardNet
        # =========================

        self.hardnet = hardmodel.HardNet()



        checkpoint = torch.load(
            hardnet_checkpoint,
            map_location="cpu"
        )


        if "model" in checkpoint:

            checkpoint = checkpoint["model"]



        self.hardnet.load_state_dict(
            checkpoint
        )



        # =========================
        # 冻结HardNet
        # =========================

        self.hardnet.eval()


        for p in self.hardnet.parameters():

            p.requires_grad = False



    # =====================================
    # 可微patch提取
    # =====================================

    def extract_patch_tensor(
            self,
            img,
            keypoints,
            size=32
    ):

        """
        img:
            [1,H,W]

        keypoints:
            [(x,y),...]

        return:
            [N,1,32,32]
        """


        patches = []


        _, H, W = img.shape



        for x, y in keypoints:


            # ---------------------
            # 坐标归一化
            # grid_sample要求:
            # [-1,1]
            # ---------------------

            x_norm = (
                x/(W-1)
                *2
                -1
            )


            y_norm = (
                y/(H-1)
                *2
                -1
            )



            theta = torch.tensor(
                [
                    [
                        [1,0,x_norm],
                        [0,1,y_norm]
                    ]
                ],
                dtype=torch.float32,
                device=img.device
            )



            grid = F.affine_grid(
                theta,
                size=(
                    1,
                    1,
                    size,
                    size
                ),
                align_corners=True
            )



            patch = F.grid_sample(
                img.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True
            )


            patches.append(
                patch.squeeze(0)
            )



        if len(patches)==0:

            return None



        return torch.stack(
            patches
        )



    # =====================================
    # HardNet descriptor loss
    # =====================================

    def hardnet_loss(
            self,
            pred,
            target,
            keypoints
    ):


        device = pred.device


        total_loss = 0.0

        valid_num = 0



        B = pred.shape[0]



        for i in range(B):


            # 当前图片关键点
            kp = keypoints[i]


            if len(kp)==0:

                continue



            # =====================
            # 可微提取patch
            # =====================

            gt_patches = self.extract_patch_tensor(
                target[i],
                kp
            )


            pred_patches = self.extract_patch_tensor(
                pred[i],
                kp
            )



            if gt_patches is None:

                continue



            # =====================
            # HardNet descriptor
            # =====================

            with torch.no_grad():

                desc_gt = self.hardnet(
                    gt_patches
                )



            desc_pred = self.hardnet(
                pred_patches
            )



            loss = F.mse_loss(
                desc_pred,
                desc_gt
            )



            total_loss += loss

            valid_num += 1




        if valid_num == 0:

            return torch.tensor(
                0.0,
                device=device,
                requires_grad=True
            )



        return total_loss / valid_num




    # =====================================
    # forward
    # =====================================

    def forward(
            self,
            pred,
            target,
            keypoints
    ):


        # ======================
        # 1. pixel loss
        # ======================

        l1_loss = F.l1_loss(
            pred,
            target
        )



        # ======================
        # 2. gradient loss
        # ======================

        def gradient(img):

            gx = (
                img[:,:,:,1:]
                -
                img[:,:,:,:-1]
            )


            gy = (
                img[:,:,1:,:]
                -
                img[:,:,:-1,:]
            )


            return gx, gy



        gx1, gy1 = gradient(pred)

        gx2, gy2 = gradient(target)



        grad_loss = (

            F.l1_loss(
                gx1,
                gx2
            )

            +

            F.l1_loss(
                gy1,
                gy2
            )

        )



        # ======================
        # 3. HardNet loss
        # ======================

        descriptor_loss = self.hardnet_loss(
            pred,
            target,
            keypoints
        )



        # ======================
        # total
        # ======================

        total_loss = (

            l1_loss

            +

            self.gradient_weight
            *
            grad_loss

            +

            0.15
            *
            descriptor_loss

        )


        return total_loss
    # ======================================
# ======================================
# 验证函数
# ======================================
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0


    with torch.no_grad():

        for img_in, img_gt, keypoints in loader:


            img_in = img_in.to(device)

            img_gt = img_gt.to(device)



            pred = model(
                img_in
            )


            loss = criterion(
                pred,
                img_gt,
                keypoints
            )


            total_loss += loss.item()



    return total_loss / len(loader)

# ======================================
if __name__ == "__main__":


    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)



    train_target = "C:/Users/qwe/Desktop/data4"



    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    print("device:", device)



    # ==================================
    # config
    # ==================================

    with open(
        "config_match_new.yaml",
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)



    # ==================================
    # 创建GT特征提取器
    # 只执行一次SIFT
    # ==================================

    feature_extractor = GTFeatureExtractor(
        config
    )



    # ==================================
    # Dataset
    # ==================================

    full_dataset = BMPDataset(
        train_target,
        feature_extractor
    )



    dataset_size = len(full_dataset)



    val_ratio = 0.2


    val_size = int(
        dataset_size * val_ratio
    )


    train_size = dataset_size - val_size



    train_dataset, val_dataset = random_split(

        full_dataset,

        [
            train_size,
            val_size
        ],

        generator=torch.Generator().manual_seed(42)

    )



    print(
        f"总样本数 : {dataset_size}"
    )


    print(
        f"训练集   : {train_size}"
    )


    print(
        f"验证集   : {val_size}"
    )



    # ==================================
    # DataLoader
    #
    # keypoints数量不固定
    # 使用自定义collate
    # ==================================

    def collate_fn(batch):

        img_in = []

        img_gt = []

        keypoints = []


        for item in batch:

            img_in.append(
                item[0]
            )

            img_gt.append(
                item[1]
            )

            keypoints.append(
                item[2]
            )


        return (

            torch.stack(img_in),

            torch.stack(img_gt),

            keypoints

        )



    train_loader = DataLoader(

        train_dataset,

        batch_size=16,

        shuffle=True,

        collate_fn=collate_fn

    )



    val_loader = DataLoader(

        val_dataset,

        batch_size=16,

        shuffle=False,

        collate_fn=collate_fn

    )



    # ==================================
    # UNet
    # ==================================

    model = TinyUNet(

        in_channels=1,

        out_channels=1

    ).to(device)



    # ==================================
    # Loss
    # ==================================

    criterion = RingL1Loss(

        hardnet_checkpoint="hardnet.pt",

        border=8,

        template_weight=0.1,

        gradient_weight=0.2

    ).to(device)



    # ==================================
    # optimizer
    # ==================================

    optimizer = optim.Adam(

        model.parameters(),

        lr=1e-3

    )



    epochs = 200



    best_loss = float("inf")

    best_epoch = 0



    best_model = (
        "model_Unet_V6_best.pth"
    )



    # ==================================
    # Training
    # ==================================

    for epoch in range(epochs):


        model.train()


        train_loss = 0.0



        for img_in, img_gt, keypoints in train_loader:



            img_in = img_in.to(device)

            img_gt = img_gt.to(device)



            # --------------------------
            # UNet
            # --------------------------

            pred = model(
                img_in
            )



            # --------------------------
            # Loss
            #
            # L1
            # Gradient
            # HardNet descriptor
            #
            # --------------------------

            loss = criterion(

                pred,

                img_gt,

                keypoints

            )



            optimizer.zero_grad()


            loss.backward()


            optimizer.step()



            train_loss += loss.item()



        train_loss /= len(train_loader)




        # ==================================
        # Validation
        # ==================================

        val_loss = validate(

            model,

            val_loader,

            criterion,

            device

        )



        print(

            f"Epoch {epoch+1:3d}/{epochs}"

            f" | Train:{train_loss:.6f}"

            f" | Val:{val_loss:.6f}"

        )



        # ==================================
        # Save best
        # ==================================

        if epoch > 40 and val_loss < best_loss:


            best_loss = val_loss


            best_epoch = epoch + 1



            torch.save(

                model.state_dict(),

                best_model

            )


            print(

                "✓ 保存最佳模型",

                f"Epoch={best_epoch}",

                f"Val={best_loss:.6f}"

            )




    print("="*60)


    print(
        "训练结束"
    )


    print(
        f"最佳Epoch : {best_epoch}"
    )


    print(
        f"最佳Val Loss : {best_loss:.6f}"
    )



    torch.save(

        model.state_dict(),

        "model_Unet_V6_final.pth"

    )


    print(
        "最终模型已保存"
    )