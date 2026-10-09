#!/usr/bin/env python3
"""为原始 wi/wo CSV 安全补充 pair 编号。

同一子目录中，按 Rgd=1421/1423/1425/1427/1429 分组并排序；每个 Rgd 的第
i 个文件组成 pair_i。默认仅预览，传入 --apply 才会真正重命名。
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

RGDS = ("1421", "1423", "1425", "1427", "1429")
PATTERN = re.compile(r"^(wi|wo)-(?!pair_)(.*Rgd=(1421|1423|1425|1427|1429).*\.csv)$", re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="执行重命名；默认仅显示计划。")
    args = parser.parse_args()
    root = args.input_root.resolve()
    if not root.is_dir():
        parser.error(f"输入目录不存在：{root}")

    plans: list[tuple[Path, Path]] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
        for path in folder.glob("*.csv"):
            match = PATTERN.match(path.name)
            if match:
                side, _, rgd = match.groups()
                groups[side][rgd].append(path)
        for side in ("wi", "wo"):
            if not groups[side]:
                continue
            missing = [r for r in RGDS if not groups[side][r]]
            if missing:
                print(f"跳过 {folder.name}/{side}：缺少 Rgd={','.join(missing)}")
                continue
            count = min(len(groups[side][r]) for r in RGDS)
            for rgd in RGDS:
                for index, old in enumerate(sorted(groups[side][rgd])[:count], start=1):
                    new = old.with_name(f"{side}-pair_{index}-{old.name[len(side) + 1:]}")
                    if new.exists():
                        raise FileExistsError(f"目标已存在，拒绝覆盖：{new}")
                    plans.append((old, new))

    print(f"计划重命名 {len(plans)} 个文件（{'执行' if args.apply else '预览'}模式）")
    for old, new in plans[:20]:
        print(f"{old} -> {new.name}")
    if len(plans) > 20:
        print(f"... 其余 {len(plans) - 20} 个省略")
    if args.apply:
        # 每个旧/新文件名均唯一，先验证完所有目标后再执行，避免半途覆盖。
        for old, new in plans:
            old.rename(new)


if __name__ == "__main__":
    main()
