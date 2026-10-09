import os
import re

folder = r"C:\\Users\\qwe\\Desktop\\data\\GHM_ID5"  # 修改为你的文件夹路径

# 获取文件夹名
folder_name = os.path.basename(folder)

pattern = re.compile(
    r".*?pair_(\d+).*?Rgd=(\d+).*\.csv$",
    re.IGNORECASE
)

for filename in os.listdir(folder):
    if not filename.endswith(".csv"):
        continue

    match = pattern.match(filename)
    if match:
        pair_num = match.group(1)
        rgd = match.group(2)

        new_name = f"{folder_name}-pair_{pair_num}-Rgd={rgd}.csv"

        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)

        print(f"{filename} -> {new_name}")

print("重命名完成")