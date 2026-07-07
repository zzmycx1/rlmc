import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """Encoder 中的可选降采样层。

    输入/输出张量约定为 [B, L, D]：
    B 是 batch size，L 是序列长度，D 是特征维度。Conv1d 需要通道维在中间，
    因此前向传播里会先转成 [B, D, L]，卷积和池化后再转回 [B, L, D]。
    """
    def __init__(self, c_in):
        super().__init__()
        # 用 1D 卷积在时间维上提取局部上下文；padding_mode='circular' 用循环填充处理序列边界。
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channel=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')

        # BatchNorm + ELU 用于稳定卷积输出并引入非线性，MaxPool 将序列长度约减半。
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # [B, L, D] -> [B, D, L]，让 D 作为 Conv1d 的 channel 维。
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # [B, D, L'] -> [B, L', D]，恢复 Transformer 层常用的序列格式。
        x = x.transpose(1, 2)
        return x

class EncoderLayer(nn.Module):
    """单个 Encoder block：自注意力 + 前馈网络。

    结构与 Transformer Encoder 类似：
    1. 多头自注意力建模序列内部依赖；
    2. 残差连接和 LayerNorm 稳定训练；
    3. 两个 1x1 Conv1d 等价于逐位置的前馈网络，将 D -> d_ff -> D。
    """
    def __init__(self,
                 attention,
                 d_model,
                 d_ff=None,
                 dropout=0.1,
                 activation="relu"
                 ):
        super().__init__()
        # 若未指定隐藏维度，使用 Transformer 常见设置：前馈层宽度为 4 * d_model。
        d_ff = d_ff or 4 * d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # 自注意力：query/key/value 都来自 x，输出 new_x 仍为 [B, L, D]。
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # 注意力子层残差连接。
        x = x + self.dropout(new_x)

        # 前馈子层。Conv1d 要求 [B, D, L]，所以进入前后都需要转置。
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 返回编码后的表示，以及注意力权重，方便可视化或调试。
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    """由多个 EncoderLayer 组成的编码器。

    conv_layers 不为空时，每个注意力层后接一个 ConvLayer 做序列降采样；
    这类设计常见于 Informer，用于减少长序列的计算量。
    """
    def __init__(self,
                 attn_layers,
                 conv_layers=None,
                 norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            # 前 len(conv_layers) 层：先做注意力，再用卷积池化压缩序列长度。
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            # 最后一层只做注意力，不再降采样，保留最终编码表示。
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            # 标准 Encoder：逐层堆叠注意力 + 前馈网络。
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        # x 是最终编码结果；attns 收集每层注意力矩阵。
        return x, attns

class DecoderLayer(nn.Module):
    """单个 Decoder block：自注意力 + 编码器-解码器交叉注意力 + 前馈网络。"""
    def __init__(self,
                 self_attention,
                 cross_attention,
                 d_model,
                 d_ff=None,
                 dropout=0.1,
                 activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Decoder 自注意力：只在目标序列 x 内部建模，通常会配合 mask 防止看到未来位置。
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        # 交叉注意力：query 来自 Decoder，key/value 来自 Encoder 输出 cross。
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask = cross_mask
        )[0])

        # 前馈子层：逐位置做 D -> d_ff -> D 的非线性变换。
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 第三次 LayerNorm 作用在前馈子层残差之后。
        return self.norm3(x + y)

class Decoder(nn.Module):
    """由多个 DecoderLayer 组成的解码器，可选最终归一化和输出投影。"""
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        super.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 逐层使用 DecoderLayer：x 是目标序列表示，cross 是 Encoder 的输出表示。
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        # projection 常用于把 d_model 映射到最终预测维度，例如时间序列预测值或词表 logits。
        if self.projection is not None:
            x = self.projection(x)

        return x
 