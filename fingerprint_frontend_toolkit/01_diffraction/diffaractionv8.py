#!/usr/bin/env python3
"""V8：Base/Raw 分别估计初始相位的衍射校正。

针对原始 ``normal/<手指>/wi/wo-pair_*-Rgd=*.csv`` 直接运行。每个 pair
有五张 wo 与五张 wi：V8 分别从各自五相位观测估计初相，再在复数域相减，
最后复用 V7 的行列校正、衍射反卷积和显示相位优化。

这就是“Base 和 Raw 在 sin/cos 累加时使用不同初始相位”的方案。
输出为 ``<output-root>/<手指>/pair_N.bmp``，不会覆盖 V7。
"""
from diffaraction_phase_common import main


if __name__ == '__main__':
    main(default_scheme='independent')
