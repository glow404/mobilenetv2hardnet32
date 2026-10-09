import os
import pandas as pd

root = r"C:\Users\qwe\Desktop\data"

for dirpath, _, filenames in os.walk(root):
    for f in filenames:
        if f.endswith(".csv"):

            path = os.path.join(dirpath, f)

            try:
                df = pd.read_csv(path)

                if len(df) == 0:
                    print("空数据:", path)

            except Exception:
                print("无法读取:", path)