"""预处理第 2 步：将同 pair、同 Rgd 的 wo 与 wi 原始 CSV 做 ``wo - wi``。

此文件不做衍射反卷积；输出的五张差分 CSV 交给 ``pic_final.py`` 联合
归一化，再交给 ``diffaractionv7.py``。V8/V9 不使用本文件，因为它们需要
保留 wo、wi 各自的原始幅值以便在复数域完成相位操作。
"""

import os
import re
import argparse
import pandas as pd

# ===========================
# 输入总目录（里面是很多小文件夹）
# ===========================
ROOT_FOLDER =r"C:\Users\qwe\Desktop\wet"

# ===========================
# 输出总目录
# ===========================
GLOBAL_OUT = r"C:\Users\qwe\Desktop\wet_diff"

pattern = re.compile(
    r"(w[io])-pair_(\d+).*?Rgd=(1421|1423|1425|1427|1429)"
)


def process_folder(folder, global_out, skip_existing=False):

    folder_name = os.path.basename(folder)

    # 输出到总目录下对应同名文件夹
    out_folder = os.path.join(
        global_out,
        folder_name
    )

    os.makedirs(out_folder, exist_ok=True)

    print("\n" + "=" * 80)
    print("处理文件夹:", folder_name)
    print("=" * 80)

    wi_files = {}
    wo_files = {}

    # ------------------------------------------------
    # 搜索当前小文件夹里的csv
    # ------------------------------------------------
    for f in os.listdir(folder):

        if not f.endswith(".csv"):
            continue

        m = pattern.search(f)

        if not m:
            continue

        prefix = m.group(1)
        pair_id = m.group(2)
        rgd = m.group(3)

        key = (pair_id, rgd)

        if prefix == "wi":
            wi_files[key] = f
        else:
            wo_files[key] = f

    common_keys = sorted(
        set(wi_files.keys()) &
        set(wo_files.keys())
    )

    print(f"找到 {len(common_keys)} 对文件\n")

    success = 0

    # ------------------------------------------------
    # wi - wo
    # ------------------------------------------------
    for pair_id, rgd in common_keys:

        wi_name = wi_files[(pair_id, rgd)]
        wo_name = wo_files[(pair_id, rgd)]

        print(f"pair={pair_id}, Rgd={rgd}")
        print(f"  wi: {wi_name}")
        print(f"  wo: {wo_name}")

        wi_path = os.path.join(folder, wi_name)
        wo_path = os.path.join(folder, wo_name)

        try:

            # 第一行也是数据
            wi_df = pd.read_csv(
                wi_path,
                header=None
            )

            wo_df = pd.read_csv(
                wo_path,
                header=None
            )

            if wi_df.shape != wo_df.shape:

                print(
                    f"  尺寸不一致: "
                    f"{wi_df.shape} vs {wo_df.shape}"
                )
                continue

            diff_df = wo_df - wi_df

            out_name = (
                f"diff-pair_{pair_id}"
                f"-Rgd={rgd}.csv"
            )

            out_path = os.path.join(
                out_folder,
                out_name
            )

            if skip_existing and os.path.exists(out_path):
                success += 1
                continue

            diff_df.to_csv(
                out_path,
                header=False,
                index=False
            )

            print(f"  -> {out_name}")

            success += 1

        except Exception as e:

            print(f"  失败: {e}")

    print(
        f"\n完成，共生成 {success} 个文件"
    )


def main():
    parser = argparse.ArgumentParser(
        description="按 pair/Rgd 对原始 wi、wo CSV 做 wo - wi 差分。"
    )
    parser.add_argument("--input-root", default=ROOT_FOLDER)
    parser.add_argument("--output-root", default=GLOBAL_OUT)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="保留已有差分 CSV，适用于中断后的续跑。",
    )
    parser.add_argument("--folder", action="append", default=[], help="仅处理指定一级数据文件夹；可重复指定。")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    for name in sorted(os.listdir(args.input_root)):
        if args.folder and name not in args.folder:
            continue
        subfolder = os.path.join(args.input_root, name)
        if os.path.isdir(subfolder):
            process_folder(
                subfolder,
                args.output_root,
                skip_existing=args.skip_existing,
            )

    print("\n" + "=" * 80)
    print("全部完成")
    print("结果保存到：")
    print(args.output_root)
    print("=" * 80)


if __name__ == "__main__":
    main()
