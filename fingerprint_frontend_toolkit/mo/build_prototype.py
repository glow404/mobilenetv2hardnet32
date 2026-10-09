# import os
# import numpy as np
# import torch
#
# from dataset import MembraneDataset
# from model import Encoder3D
# from torch.utils.data import DataLoader
#
#
# ROOT = r"C:\Users\qwe\Desktop\finger\mo"
#
# dataset = MembraneDataset(ROOT)
#
#
# loader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=False
# )
#
#
# device = torch.device(
#     "cuda" if torch.cuda.is_available()
#     else "cpu"
# )
#
#
# # ============================
# # 创建Encoder
# # ============================
#
# model = Encoder3D().to(device)
#
#
#
# # ============================
# # 加载checkpoint
# # ============================
#
# checkpoint = torch.load(
#     "best_model.pth",
#     map_location=device
# )
#
#
# # 只加载Encoder参数
# model.load_state_dict(
#     checkpoint["encoder"]
# )
#
#
# model.eval()
#
#
#
# # ============================
# # 提取特征
# # ============================
#
# features = {}
#
#
# with torch.no_grad():
#
#     for x, y in loader:
#
#
#         x = x.to(device)
#
#
#         # 输出128维特征
#         feat = model(x)
#
#
#         label = y.item()
#
#
#         if label not in features:
#             features[label] = []
#
#
#         features[label].append(
#             feat.cpu().numpy()[0]
#         )
#
#
#
# # ============================
# # 保存prototype
# # ============================
#
# os.makedirs(
#     "prototype_db",
#     exist_ok=True
# )
#
#
#
# for label in features:
#
#
#     proto = np.mean(
#         features[label],
#         axis=0
#     )
#
#
#     np.save(
#         f"prototype_db/{label}.npy",
#         proto
#     )
#
#
#
# print("完成")

import os
import re
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from model import Encoder3D


ROOT = r"C:\Users\qwe\Desktop\finger\mo\train"

MODEL_PATH = "best_model.pth"

SAVE_DIR = "prototype_db"

NUM_FRAMES = 16



device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



# =========================
# 加载Encoder
# =========================

encoder = Encoder3D().to(device)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


encoder.load_state_dict(
    checkpoint["encoder"]
)


encoder.eval()



# =========================
# 文件匹配
# =========================

pattern = re.compile(
    r"pair_(\d+).*Rgd=(\d+)"
)



def load_csv(path):

    arr = pd.read_csv(
        path,
        header=None
    ).values.astype(
        np.float32
    )


    # 与训练一致
    arr = arr[
        0::10,
        0::10
    ]

    return arr



os.makedirs(
    SAVE_DIR,
    exist_ok=True
)



classes = [
    c for c in sorted(os.listdir(ROOT))
    if os.path.isdir(
        os.path.join(ROOT,c)
    )
]



class_to_idx = {
    c:i
    for i,c in enumerate(classes)
}



# =========================
# 每个类别取一个pair
# =========================

for cls in classes:


    cls_dir=os.path.join(
        ROOT,
        cls
    )


    pair_groups=defaultdict(list)



    for file in os.listdir(cls_dir):


        if not file.endswith(".csv"):
            continue


        m=pattern.search(file)


        if m is None:
            continue


        pair_id=int(
            m.group(1)
        )

        rgd=int(
            m.group(2)
        )


        pair_groups[pair_id].append(
            (
                rgd,
                os.path.join(
                    cls_dir,
                    file
                )
            )
        )



    # =========================
    # 选择第一个完整pair
    # =========================

    template_pair=None


    for pair_id,files in pair_groups.items():

        if len(files)==61:

            template_pair=files
            print(
                cls,
                "选择pair:",
                pair_id
            )

            break



    if template_pair is None:

        print(
            cls,
            "没有完整pair"
        )

        continue



    # Rgd排序

    template_pair.sort(
        key=lambda x:x[0]
    )


    # 取16帧

    template_pair=template_pair[:NUM_FRAMES]


    imgs=[]


    for rgd,path in template_pair:

        imgs.append(
            load_csv(path)
        )



    x=np.stack(
        imgs
    )



    # 标准化

    x=(
        x-x.mean()
    )/(
        x.std()+1e-6
    )


    x=torch.tensor(
        x,
        dtype=torch.float32
    )


    # B,C,D,H,W

    x=x.unsqueeze(0).unsqueeze(0)

    x=x.to(device)



    with torch.no_grad():

        feat=encoder(x)



    # L2归一化

    feat=torch.nn.functional.normalize(
        feat,
        p=2,
        dim=1
    )



    prototype=feat.cpu().numpy()[0]



    np.save(
        os.path.join(
            SAVE_DIR,
            f"{class_to_idx[cls]}.npy"
        ),
        prototype
    )


    print(
        "保存:",
        f"{class_to_idx[cls]}.npy"
    )



print("单pair模板构建完成")