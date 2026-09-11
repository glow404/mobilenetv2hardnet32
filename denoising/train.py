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
            sigma = random.uniform(0.01, 0.20)
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
    def __init__(self, target_dir, train=True):
        self.target_dir = target_dir
        self.files = [f for f in os.listdir(target_dir) if f.endswith('.bmp')]
        self.train = train

    def __len__(self):
        return len(self.files)

    def read_img(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        return img

    def __getitem__(self, idx):
        filename = self.files[idx]
        target_path = os.path.join(self.target_dir, filename)
        img_gt = self.read_img(target_path)

        # 合成退化
        img_pil = Image.fromarray(img_gt)
        img_in_pil = synthetic_degradation(img_pil)
        img_in = np.array(img_in_pil)

        # 归一化
        img_in = img_in.astype(np.float32) / 255.0
        img_gt = img_gt.astype(np.float32) / 255.0

        # ✅ 完全关闭旋转，杜绝尺寸变化
        # 不做任何旋转

        # 维度
        img_in = np.expand_dims(img_in, axis=0)  # [1, 100, 110]
        img_gt = np.expand_dims(img_gt, axis=0)  # [1, 100, 110]

        return torch.tensor(img_in), torch.tensor(img_gt)

# ======================================
# 4. 损失函数
# ======================================
class RingL1Loss(nn.Module):

    def __init__(
            self,
            config,
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
        # GT 特征提取器
        # =========================

        self.feature_extractor = GTFeatureExtractor(
            config
        )


        # =========================
        # HardNet
        # =========================

        self.hardnet =hardmodel.HardNet()


        checkpoint = torch.load(
            hardnet_checkpoint,
            map_location="cpu"
        )


        if "model" in checkpoint:

            checkpoint = checkpoint["model"]


        self.hardnet.load_state_dict(
            checkpoint
        )


        # 冻结HardNet

        self.hardnet.eval()


        for p in self.hardnet.parameters():

            p.requires_grad=False




    # =====================================
    # descriptor loss
    # =====================================

    def hardnet_loss(
            self,
            pred,
            target
    ):


        device = pred.device


        total_loss = 0.0

        valid_num = 0



        B = pred.shape[0]


        for i in range(B):


            # -------------------------
            # GT图像
            # -------------------------

            gt_img = (
                target[i]
                .detach()
                .cpu()
                .numpy()
                .squeeze()
                *255
            ).astype(
                np.uint8
            )


            # -------------------------
            # GT检测关键点
            # -------------------------

            keypoints = self.feature_extractor.detect_keypoints(
                gt_img
            )


            if len(keypoints)==0:

                continue



            # -------------------------
            # GT patch
            # -------------------------

            _, gt_patches, _ = self.feature_extractor.extract_patches(

                gt_img,

                keypoints

            )



            if len(gt_patches)==0:

                continue



            # -------------------------
            # Denoised图像
            # -------------------------

            pred_img = (

                pred[i]
                .detach()
                .cpu()
                .numpy()
                .squeeze()
                *255

            ).astype(
                np.uint8
            )



            _, pred_patches, _ = self.feature_extractor.extract_patches(

                pred_img,

                keypoints

            )



            if len(pred_patches)==0:

                continue



            # ======================
            # numpy -> tensor
            # ======================

            gt_patches = torch.from_numpy(
                gt_patches
            ).float()


            pred_patches=torch.from_numpy(
                pred_patches
            ).float()



            # HardNet输入:

            # [N,1,32,32]

            gt_patches = gt_patches.unsqueeze(1).to(device)

            pred_patches = pred_patches.unsqueeze(1).to(device)



            # ======================
            # descriptor
            # ======================


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

            valid_num +=1




        if valid_num==0:

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
            target
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


            return gx,gy



        gx1,gy1 = gradient(pred)

        gx2,gy2 = gradient(target)



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
        # 3. HardNet descriptor loss
        # ======================

        descriptor_loss = self.hardnet_loss(

            pred,

            target

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

            self.template_weight
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

        for img_in, img_gt in loader:

            img_in = img_in.to(device)
            img_gt = img_gt.to(device)

            pred = model(img_in)

            loss = criterion(pred, img_gt)

            total_loss += loss.item()

    return total_loss / len(loader)

# ======================================
if __name__ == "__main__":


    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


    train_target = "./target_bmp"


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    print("device:", device)



    # ==================================
    # Dataset
    # ==================================

    full_dataset = BMPDataset(
        train_target
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



    train_loader = DataLoader(

        train_dataset,

        batch_size=16,

        shuffle=True

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=16,

        shuffle=False

    )



    # ==================================
    # UNet
    # ==================================

    model = TinyUNet(

        in_channels=1,

        out_channels=1

    ).to(device)



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
    # Loss
    # ==================================
    #
    # 内部:
    #
    # GT
    # |
    # SIFT
    # |
    # patch
    # |
    # HardNet
    #
    # pred
    # |
    # patch
    # |
    # HardNet
    #
    # descriptor loss
    #
    # ==================================

    criterion = RingL1Loss(

        config=config,

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
        "model_Unet_V3_best.pth"
    )



    # ==================================
    # Training
    # ==================================

    for epoch in range(epochs):


        model.train()


        train_loss = 0.0



        for img_in, img_gt in train_loader:


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

                img_gt

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

        "model_Unet_V12_final.pth"

    )


    print(
        "最终模型已保存"
    )