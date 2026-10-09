#!/usr/bin/env python3
"""V9：公共初相 + Base/Raw 复数全局相位对齐的衍射校正。

针对原始 ``normal/<手指>/wi/wo-pair_*-Rgd=*.csv`` 直接运行。wo、wi 先以
相同初始相位完成五相位复数解调；随后用高幅值像素的加权复相关，估计一个
全局相位旋转，将 wi 对齐到 wo 后再相减。最后沿用 V7 的行列校正、衍射
反卷积与显示相位优化。

这就是“两个复数做一次相位对齐”的方案。输出不会覆盖 V7 或 V8。
"""
from diffaraction_phase_common import main


if __name__ == '__main__':
    main(default_scheme='same-align')
