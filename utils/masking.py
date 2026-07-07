import torch
import numpy as np
import math

class TriangularCausalMask():
    """标准因果注意力 mask。

    生成形状为 [B, 1, L, L] 的布尔矩阵，其中 True 表示该位置会被 mask 掉。
    对第 i 个 query 来说，j > i 的 key 属于未来位置，需要屏蔽，防止模型提前看到未来信息。
    """
    def __init__(self, B, L, device="cpu"):
        # 多出来的维度 1 用于和多头注意力的 head 维广播：[B, H, L, L]。
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # torch.triu(..., diagonal=1) 取主对角线以上区域，即未来时间步。
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        # 返回给 attention 层使用，通常配合 scores.masked_fill(mask, -inf)。
        return self._mask

class ProbMask():
    """ProbSparse 注意力中用于被采样 query 的因果 mask。

    Full attention 可以直接构造 [B, 1, L, L] 的完整因果 mask；
    ProbSparse attention 只计算部分 query 的 score，因此需要根据 index 取出这些 query 对应的 mask 行。
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # 基础因果 mask，形状 [L, S]；S 由 scores 的最后一维决定，通常是 key 的长度。
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        # 扩展到 batch 和 head 维，得到 [B, H, L, S]，便于按每个 batch/head 的 index 取行。
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        # index 指出实际参与计算的 query 位置；这里取出这些 query 对应的因果 mask。
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        # 注意：当前代码没有把 indicator 赋给 self._mask，下面的 mask 属性会读不到 _mask。

    @property
    def mask(self):
        return self._mask

class LocalMask():
    """局部因果注意力 mask。

    该 mask 同时屏蔽未来位置和过远的历史位置：
    对第 i 个 query，只允许关注 [i - ceil(log2(L)), i] 这个局部时间窗口。
    """
    def __init__(self, B, L, S, device="cpu"):
        mask_shape = [B, 1, L, S]
        with torch.no_grad():
            # 根据序列长度动态确定局部窗口大小；L 越长，可看的历史范围越大。
            self.len = math.ceil(np.log2(L))
            # _mask1 屏蔽未来位置：key 的时间步大于当前 query 的时间步。
            self._mask1 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            # _mask2 屏蔽太久以前的位置：key 的时间步小于 i - self.len。
            self._mask2 = ~torch.triu(torch.ones(mask_shape,dtype=torch.bool),diagonal=-self.len).to(device)
            # 合并两个条件：未来位置或超出局部窗口的历史位置都会被 mask。
            self._mask = self._mask1+self._mask2

    @property
    def mask(self):
        return self._mask
