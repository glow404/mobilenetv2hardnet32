#
# import os
# import re
# import random
# import numpy as np
# import pandas as pd
# import torch
#
# from collections import defaultdict
# from torch.utils.data import Dataset
#
#
# class MembraneDataset(Dataset):
#
#     def __init__(self, root, num_frames=16):
#
#         self.samples = []
#
#         self.num_frames = num_frames
#
#
#         # 只保留文件夹
#         classes = [
#             c for c in sorted(os.listdir(root))
#             if os.path.isdir(
#                 os.path.join(root, c)
#             )
#         ]
#
#
#         self.class_to_idx = {
#             c: i
#             for i, c in enumerate(classes)
#         }
#
#
#         pattern = re.compile(
#             r"pair_(\d+).*Rgd=(\d+)"
#         )
#
#
#         for cls in classes:
#
#             cls_dir = os.path.join(root, cls)
#
#
#             pair_groups = defaultdict(list)
#
#
#             for file in os.listdir(cls_dir):
#
#                 if not file.endswith(".csv"):
#                     continue
#
#
#                 m = pattern.search(file)
#
#                 if m is None:
#                     continue
#
#
#                 pair_id = int(m.group(1))
#                 rgd = int(m.group(2))
#
#
#                 pair_groups[pair_id].append(
#                     (
#                         rgd,
#                         os.path.join(
#                             cls_dir,
#                             file
#                         )
#                     )
#                 )
#
#
#             for pair_id, files in pair_groups.items():
#
#                 # 按 Rgd 排序
#                 files.sort(
#                     key=lambda x:x[0]
#                 )
#
#
#                 # 原始必须61个
#                 if len(files) != 61:
#                     continue
#
#
#                 self.samples.append(
#                     (
#                         files,
#                         self.class_to_idx[cls]
#                     )
#                 )
#
#
#         print(
#             "有效pair数量:",
#             len(self.samples)
#         )
#
#
#
#     def __len__(self):
#
#         return len(self.samples)
#
#
#
#     def __getitem__(self, idx):
#
#         files, label = self.samples[idx]
#
#
#         # =========================
#         # 随机选择16个csv
#         # =========================
#
#         selected = random.sample(
#             files,
#             self.num_frames
#         )
#
#
#         # 保持Rgd顺序
#         selected.sort(
#             key=lambda x:x[0]
#         )
#
#
#         imgs = []
#
#
#         for rgd, path in selected:
#
#
#             try:
#
#                 arr = pd.read_csv(
#                     path,
#                     header=None
#                 ).values.astype(
#                     np.float32
#                 )
#
#
#             except Exception as e:
#
#                 print("\n读取失败文件:")
#                 print(path)
#
#                 raise e
#
#
#
#             imgs.append(arr)
#
#
#
#         # (16,H,W)
#         x = np.stack(imgs)
#
#
#
#         # =========================
#         # 标准化
#         # =========================
#
#         x = (
#             x - x.mean()
#         ) / (
#             x.std() + 1e-6
#         )
#
#
#
#         x = torch.tensor(
#             x,
#             dtype=torch.float32
#         )
#
#
#         # (16,H,W)
#         # ->
#         # (1,16,H,W)
#         #
#         # Conv3d输入:
#         # B,C,D,H,W
#
#         x = x.unsqueeze(0)
#
#
#
#         return x, label
# import os
# import re
# import numpy as np
# import pandas as pd
# import torch
#
# from collections import defaultdict
# from torch.utils.data import Dataset
#
#
# class MembraneDataset(Dataset):
#
#     def __init__(self, root, num_frames=16):
#
#         self.samples = []
#
#         self.num_frames = num_frames
#
#
#         # 只保留文件夹作为类别
#         classes = [
#             c for c in sorted(os.listdir(root))
#             if os.path.isdir(
#                 os.path.join(root, c)
#             )
#         ]
#
#
#         self.class_to_idx = {
#             c: i
#             for i, c in enumerate(classes)
#         }
#
#
#         pattern = re.compile(
#             r"pair_(\d+).*Rgd=(\d+)"
#         )
#
#
#         for cls in classes:
#
#             cls_dir = os.path.join(root, cls)
#
#
#             pair_groups = defaultdict(list)
#
#
#             for file in os.listdir(cls_dir):
#
#                 if not file.endswith(".csv"):
#                     continue
#
#
#                 m = pattern.search(file)
#
#                 if m is None:
#                     continue
#
#
#                 pair_id = int(m.group(1))
#                 rgd = int(m.group(2))
#
#
#                 pair_groups[pair_id].append(
#                     (
#                         rgd,
#                         os.path.join(
#                             cls_dir,
#                             file
#                         )
#                     )
#                 )
#
#
#             for pair_id, files in pair_groups.items():
#
#                 # 按Rgd排序
#                 files.sort(
#                     key=lambda x:x[0]
#                 )
#
#
#                 # 必须61个
#                 if len(files) != 61:
#                     continue
#
#
#                 self.samples.append(
#                     (
#                         files,
#                         self.class_to_idx[cls]
#                     )
#                 )
#
#
#         print(
#             "有效pair数量:",
#             len(self.samples)
#         )
#
#
#
#     def __len__(self):
#
#         return len(self.samples)
#
#
#
#     def __getitem__(self, idx):
#
#         files, label = self.samples[idx]
#
#
#         # ============================
#         # 固定取前16个
#         # Rgd排序后:
#         # 1360,1361,...取16个
#         # ============================
#
#         files = files[:self.num_frames]
#
#
#         imgs = []
#
#
#         for rgd, path in files:
#
#             try:
#
#                 arr = pd.read_csv(
#                     path,
#                     header=None
#                 ).values.astype(
#                     np.float32
#                 )
#
#
#             except Exception as e:
#
#                 print("\n读取失败文件:")
#                 print(path)
#
#                 raise e
#
#
#             imgs.append(arr)
#
#
#
#         # (16,H,W)
#         x = np.stack(imgs)
#
#
#
#         # 标准化
#         x = (
#             x - x.mean()
#         ) / (
#             x.std() + 1e-6
#         )
#
#
#
#         x = torch.tensor(
#             x,
#             dtype=torch.float32
#         )
#
#
#         # Conv3D:
#         # C,D,H,W
#         #
#         # 1,16,H,W
#
#         x = x.unsqueeze(0)
#
#
#
#         return x, label

#扰动
# import os
# import re
# import random
# import numpy as np
# import pandas as pd
# import torch
#
# from collections import defaultdict
# from torch.utils.data import Dataset
#
#
#
# class MembraneDataset(Dataset):
#
#     def __init__(self, root, num_frames=16):
#
#         self.samples = []
#
#         self.num_frames = num_frames
#
#
#         classes = [
#             c for c in sorted(os.listdir(root))
#             if os.path.isdir(
#                 os.path.join(root,c)
#             )
#         ]
#
#
#         self.class_to_idx = {
#             c:i
#             for i,c in enumerate(classes)
#         }
#
#
#         pattern = re.compile(
#             r"pair_(\d+).*Rgd=(\d+)"
#         )
#
#
#         for cls in classes:
#
#             cls_dir = os.path.join(
#                 root,
#                 cls
#             )
#
#
#             pair_groups = defaultdict(list)
#
#
#             for file in os.listdir(cls_dir):
#
#                 if not file.endswith(".csv"):
#                     continue
#
#
#                 m = pattern.search(file)
#
#                 if m is None:
#                     continue
#
#
#                 pair_id = int(m.group(1))
#                 rgd = int(m.group(2))
#
#
#                 pair_groups[pair_id].append(
#                     (
#                         rgd,
#                         os.path.join(
#                             cls_dir,
#                             file
#                         )
#                     )
#                 )
#
#
#
#             for pair_id, files in pair_groups.items():
#
#                 files.sort(
#                     key=lambda x:x[0]
#                 )
#
#
#                 if len(files)!=61:
#                     continue
#
#
#                 self.samples.append(
#                     (
#                         files,
#                         self.class_to_idx[cls]
#                     )
#                 )
#
#
#         print(
#             "有效pair数量:",
#             len(self.samples)
#         )
#
#
#
#     def __len__(self):
#
#         return len(self.samples)
#
#
#
#     def __getitem__(self, idx):
#
#         files,label = self.samples[idx]
#
#
#         # =================================
#         # 固定15个 + 随机1个
#         #
#         # 固定:
#         # 0~14
#         #
#         # 随机:
#         # 15~60
#         # =================================
#
#
#         fixed = files[:15]
#
#
#         random_one = random.choice(
#             files[15:]
#         )
#
#
#         selected = fixed + [random_one]
#
#
#         # 保证Rgd顺序
#         selected.sort(
#             key=lambda x:x[0]
#         )
#
#
#         imgs=[]
#
#
#         for rgd,path in selected:
#
#             try:
#
#                 arr = pd.read_csv(
#                     path,
#                     header=None
#                 ).values.astype(
#                     np.float32
#                 )
#
#
#             except Exception as e:
#
#                 print(
#                     "\n读取失败文件:"
#                 )
#
#                 print(path)
#
#                 raise e
#
#
#             imgs.append(arr)
#
#
#
#         # (16,H,W)
#         x=np.stack(imgs)
#
#
#
#         # 标准化
#
#         x = (
#             x-x.mean()
#         ) / (
#             x.std()+1e-6
#         )
#
#
#
#         x=torch.tensor(
#             x,
#             dtype=torch.float32
#         )
#
#
#         # (C,D,H,W)
#         # (1,16,H,W)
#
#         x=x.unsqueeze(0)
#
#
#
#         return x,label

import os
import re
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from torch.utils.data import Dataset


class MembraneDataset(Dataset):

    def __init__(self, root, num_frames=16):

        self.samples = []

        self.num_frames = num_frames


        # ============================
        # 类别文件夹
        # ============================

        classes = [
            c for c in sorted(os.listdir(root))
            if os.path.isdir(
                os.path.join(root, c)
            )
        ]


        self.class_to_idx = {
            c: i
            for i, c in enumerate(classes)
        }


        # 文件名匹配
        # pair_数字 ... Rgd=数字
        pattern = re.compile(
            r"pair_(\d+).*Rgd=(\d+)"
        )


        # ============================
        # 遍历类别
        # ============================

        for cls in classes:

            cls_dir = os.path.join(
                root,
                cls
            )


            pair_groups = defaultdict(list)


            for file in os.listdir(cls_dir):

                if not file.endswith(".csv"):
                    continue


                m = pattern.search(file)


                if m is None:
                    continue


                pair_id = int(
                    m.group(1)
                )

                rgd = int(
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


            # ============================
            # 每个pair
            # ============================

            for pair_id, files in pair_groups.items():


                # 按Rgd排序
                files.sort(
                    key=lambda x:x[0]
                )


                # 必须61帧
                if len(files) != 61:
                    continue



                self.samples.append(
                    (
                        files,
                        self.class_to_idx[cls]
                    )
                )


        print("=" * 50)
        print(
            "有效pair数量:",
            len(self.samples)
        )
        print(
            "类别数量:",
            len(self.class_to_idx)
        )
        print("=" * 50)



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, idx):


        files, label = self.samples[idx]


        # ============================
        # 取前16帧
        # ============================

        files = files[:self.num_frames]


        imgs = []


        for rgd, path in files:


            try:

                arr = pd.read_csv(
                    path,
                    header=None
                ).values.astype(
                    np.float32
                )


                # =================================
                # CSV降采样
                #
                # 奇数行:
                # 第1、3、5...
                # Python:
                # 0,2,4...
                #
                # 偶数列:
                # 第2、4、6...
                # Python:
                # 1,3,5...
                #
                # 例如:
                # 100×110
                # ->
                # 50×55
                # =================================

                arr = arr[
                    0::10,
                    0::10
                ]


            except Exception as e:


                print("\n读取失败:")
                print(path)

                raise e



            imgs.append(arr)



        # ============================
        #
        # x:
        # (16,H,W)
        #
        # ============================

        x = np.stack(
            imgs
        )



        # ============================
        # 标准化
        # ============================

        x = (
            x - x.mean()
        ) / (
            x.std() + 1e-6
        )



        x = torch.tensor(
            x,
            dtype=torch.float32
        )



        # ============================
        # Conv3D输入
        #
        # 原:
        # (16,H,W)
        #
        # 增加channel:
        #
        # (1,16,H,W)
        #
        # DataLoader后:
        #
        # (B,1,16,H,W)
        #
        # ============================

        x = x.unsqueeze(0)



        return x, label