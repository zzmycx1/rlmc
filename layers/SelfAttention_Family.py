import torch
import torch.nn as nn

import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask


"""
本文件实现了两种注意力机制，以及一个把注意力机制包装成多头注意力层的封装类。

常用维度约定：
B: batch size，批大小
L / L_Q: query 序列长度
S / L_K / L_V: key/value 序列长度
H: 注意力头数
E / D: 每个注意力头内部的特征维度

AttentionLayer 接收普通的 [B, L, d_model] 张量；
FullAttention / ProbAttention 接收已经拆成多头后的 [B, L, H, D] 张量。
"""


class FullAttention(nn.Module):
    """标准 scaled dot-product attention。

    它会计算每一个 query 和每一个 key 的相似度，因此时间和空间复杂度都是 O(L*S)。
    如果 mask_flag=True，则使用因果 mask，防止当前位置看到未来时间步。
    """

    def __init__(self,
                 mask_flag=True,
                 scale=None,
                 attention_dropout=0.1,
                 output_attention=False):
        super().__init__()
        # scale 用于缩放 QK^T，默认是 1 / sqrt(每个头的维度)。
        self.scale = scale
        # mask_flag=True 时使用因果 mask，常用于时间序列预测或自回归任务。
        self.mask_flag = mask_flag
        # output_attention=True 时额外返回注意力权重矩阵，便于可视化或分析。
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # queries: [B, L, H, E]，L 是 query 长度。
        # keys:    [B, S, H, E]，S 是 key 长度。
        # values:  [B, S, H, D]。
        B, L, _, E = queries.shape
        scale = self.scale or 1. / sqrt(E)

        # 计算 Q 和 K 的点积相似度。
        # einsum 结果 scores: [B, H, L, S]，
        # 表示每个 batch、每个 head 下，每个 query 对所有 key 的得分。
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                # 默认构造上三角因果 mask：第 i 个位置不能关注 i 之后的位置。
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            # 被 mask 的位置填成 -inf，这样 softmax 后对应权重接近 0。
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 对最后一维 S 做 softmax，得到每个 query 对所有 key 的注意力分布。
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # 用注意力权重加权 value。
        # A:      [B, H, L, S]
        # values: [B, S, H, D]
        # V:      [B, L, H, D]
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    """ProbSparse attention，常见于 Informer 一类长序列模型。

    核心思想：
    1. 不是给所有 query 都完整计算 attention；
    2. 先随机采样一部分 key，估计哪些 query 的注意力分布更“尖锐”；
    3. 只对 top-u 个重要 query 做完整 QK^T；
    4. 其他 query 使用一个近似的初始 context。

    这样可以降低长序列上的计算量。
    """

    # 注意：这里函数名是 __init 而不是 Python 构造函数要求的 __init__。
    # 这意味着直接 ProbAttention(...) 传参时，这个方法不会作为构造函数被调用。
    # 下面保持原始代码不改逻辑，只对其内部意图进行注释。
    def __init__(self,
                mask_flag=True,
                factor=5,
                scale=None,
                attention_dropout=0.1,
                output_attention=False):
        super().__init__()
        # factor 控制采样 key 数和 top query 数，越大越接近 full attention，但越耗时。
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """采样 key 来估计 query 的重要性，再选出最重要的 n_top 个 query。

        Q: [B, H, L_Q, E]
        K: [B, H, L_K, E]
        sample_k: 每个 query 随机采样多少个 key 用于粗略估计
        n_top: 最终选出多少个 query 做完整 attention
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 为每个 query 准备一份 K 的视图，方便后面按随机索引抽样。
        # K_expand: [B, H, L_Q, L_K, E]
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)

        # 对每个 query，随机抽 sample_k 个 key。
        # index_sample: [L_Q, sample_k]
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q

        # 取出每个 query 对应的被采样 key。
        # K_sample: [B, H, L_Q, sample_k, E]
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # 粗略计算每个 query 与采样 key 的点积得分。
        # Q_K_sample: [B, H, L_Q, sample_k]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # 稀疏性度量 M：
        # max(score) - mean(score) 越大，说明该 query 对少数 key 更敏感，
        # 注意力分布更尖锐，更值得进行完整 attention 计算。
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)

        # 选出每个 batch、每个 head 下 M 最大的 n_top 个 query 的位置。
        # M_top: [B, H, n_top]
        M_top = M.topk(n_top, sorted=False)[1]

        # 根据 M_top 取出重要 query。
        # Q_reduce: [B, H, n_top, E]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)

        # 对这些重要 query 和所有 key 做完整点积。
        # Q_K: [B, H, n_top, L_K]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """为所有 query 准备初始 context，之后只更新 top query 的 context。"""
        B, H, L_V, _ = V.shape
        if not self.mask_flag:
            # 非因果注意力下，未被选中的 query 直接用所有 value 的均值作为近似 context。
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # 因果自注意力要求 query 长度等于 value 长度。
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only

            # 对于第 i 个位置，初始 context 是 value[0:i] 的累计和，
            # 保证不会使用未来信息。
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """用完整 attention 结果更新被选中的 top query 的 context。"""
        B, H, L_V, _ = V.shape

        if self.mask_flag:
            # ProbMask 只为被选中的 query 构造 mask 行。
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # scores: [B, H, n_top, L_V]，在 key/value 维度做 softmax。
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        # 只更新 index 指向的 top query 位置，其余位置保持初始 context。
        # torch.matmul(attn, V): [B, H, n_top, D]
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            # 如果需要返回 attention，先给所有 query 一个均匀分布占位，
            # 再把 top query 的真实 attention 写回去。
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        # 输入来自 AttentionLayer，形状仍是 [B, L, H, D]。
        _, L_Q, _, D = queries.shape
        _, L_K, _, _ = keys.shape

        # ProbAttention 内部更习惯使用 [B, H, L, D]，
        # 所以把 head 维和序列维交换。
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # U_part: 每个 query 采样多少个 key，用于估计 query 的重要性。
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)

        # u: 最终选多少个 query 做完整 attention。
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        # 采样数不能超过真实序列长度。
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # scores_top: top query 与所有 key 的完整得分。
        # index: top query 在原序列中的位置。
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # 缩放 QK^T，和 full attention 中的 1/sqrt(D) 作用一致。
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # 为所有 query 初始化 context。
        context = self._get_initial_context(values, L_Q)

        # 用精确 attention 结果更新 top query 的 context。
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """多头注意力的外层封装。

    这个类负责三件事：
    1. 用线性层把输入投影成 query/key/value；
    2. 把 d_model 拆成 n_heads 个注意力头；
    3. 调用具体 attention 实现后，再把多个头合并回 d_model。
    """

    def __init__(self,
                 attention,
                 d_model,
                 n_heads,
                 d_keys=None,
                 d_values=None):
        super().__init__()

        # 每个 head 的 key/query 维度，默认平均拆分 d_model。
        d_keys = d_keys or (d_model // n_heads)
        # 每个 head 的 value 维度，默认也平均拆分 d_model。
        d_values = d_values or (d_model // n_heads)

        # inner_attention 可以是 FullAttention，也可以是 ProbAttention。
        self.inner_attention = attention

        # 三个线性层分别生成 Q、K、V。
        # 输出维度是 每个头的维度 * 头数，后面再 reshape 成多头格式。
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        # 多头输出拼接后，再投影回 d_model。
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        # queries: [B, L, d_model]
        # keys:    [B, S, d_model]
        # values:  [B, S, d_model]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 线性投影后拆成多头：
        # queries: [B, L, H, d_keys]
        # keys:    [B, S, H, d_keys]
        # values:  [B, S, H, d_values]
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 调用具体注意力实现，输出 out: [B, L, H, d_values]。
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        # 把多个 head 拼回一个向量：[B, L, H*d_values]。
        out = out.view(B, L, -1)

        # 最终投影回 d_model，并按需返回 attention 权重。
        return self.out_projection(out), attn
