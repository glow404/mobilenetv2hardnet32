import os

folder = r"C:\\Users\\qwe\\Desktop\\data\\GHM_ID5"

files = sorted(
    [f for f in os.listdir(folder) if f.endswith(".csv")]
)

n_pair = 300

for i, file in enumerate(files):

    pair_id = i % n_pair + 1

    old_path = os.path.join(folder, file)

    new_name = file.replace(
        "wo-",
        f"wo-pair_{pair_id}-",
        1
    )

    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)

print("重命名完成")