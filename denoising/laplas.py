import os
import cv2
import numpy as np

# ===============================
# 路径
# ===============================
input_root = r"C:/Users/qwe/Desktop/finger/pre_deal/new_data_V3"
output_root =  r"C:/Users/qwe/Desktop/finger/pre_deal/new_data_V3_lp"

# ===============================
# 拉普拉斯锐化
# ===============================
def laplacian_sharpen(img, alpha=0.1):
    """
    拉普拉斯锐化

    alpha:
        锐化强度
        建议：
        0.3~0.5  较弱
        0.8~1.0  常用
        >1       较强
    """

    # 计算拉普拉斯
    lap = cv2.Laplacian(
        img,
        cv2.CV_32F,
        ksize=1
    )

    # 锐化
    sharpen = img.astype(np.float32) - alpha * lap

    # 限制到0~255
    sharpen = np.clip(
        sharpen,
        0,
        255
    ).astype(np.uint8)

    return sharpen


# ===============================
# 主程序
# ===============================
count = 0

for root, dirs, files in os.walk(input_root):

    relative_path = os.path.relpath(root, input_root)

    save_dir = os.path.join(
        output_root,
        relative_path
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    for file in files:

        if not file.lower().endswith(".bmp"):
            continue

        input_path = os.path.join(
            root,
            file
        )

        output_path = os.path.join(
            save_dir,
            file
        )

        # =====================
        # 读取图像
        # =====================
        img = cv2.imdecode(
            np.fromfile(
                input_path,
                dtype=np.uint8
            ),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            print("读取失败:", input_path)
            continue

        # =====================
        # 拉普拉斯锐化
        # =====================
        img_sharp = laplacian_sharpen(
            img,
            alpha=0.2
        )

        # =====================
        # 保存图像
        # =====================
        cv2.imencode(
            ".bmp",
            img_sharp
        )[1].tofile(
            output_path
        )

        count += 1

        print("完成:", input_path)

print("========================")
print("全部完成，共处理:", count, "张")