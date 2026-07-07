"""Embedding blocks used by time-series forecasting models.

Most layers in this file expect batch-first time-series tensors:
    x:      [batch, seq_len, input_channels]
    x_mark: [batch, seq_len, time_feature_channels]
and return tensors projected to:
    [batch, seq_len, d_model]
"""

import torch
import torch.nn as nn
import math


class ETSEmbedding(nn.Module):
    """Value embedding used by ETS-style models.

    The Conv1d works on [batch, channels, seq_len], so forward() temporarily
    permutes the tensor, applies a length-preserving causal-style convolution,
    then switches back to [batch, seq_len, d_model].
    """

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        # padding=2 with kernel_size=3 adds two future positions; slicing them
        # off in forward() keeps the original sequence length.
        self.conv = nn.Conv1d(in_channels=c_in,
                              out_channels=d_model,
                              kernel_size=3,
                              padding=2,
                              bias=False)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        # [B, L, C] -> [B, C, L] -> Conv1d -> remove padded tail -> [B, L, D]
        x = self.conv(x.permute(0, 2, 1))[..., :-2]
        return self.dropout(x.transpose(1, 2))
    
class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal position encoding from the Transformer paper."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Precompute positional encodings once. register_buffer keeps them in
        # the state_dict and moves them with .to(device), but they are not trainable.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is only used to know the current sequence length.
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    """Projects raw input variables into d_model with a local circular Conv1d."""

    def __init__(self, c_in, d_model):
        super().__init__()
        # Circular padding lets the first and last time steps be neighbors for
        # the convolution. The version check preserves older PyTorch behavior.
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # Conv1d expects channels before length: [B, L, C] -> [B, C, L].
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class FixedEmbedding(nn.Module):
    """Non-trainable sinusoidal lookup table for discrete time fields."""

    def __init__(self, c_in, d_model):
        super().__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        # detach() makes the fixed table explicit even if this module is reused.
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    """Embeds calendar fields and sums them into one time feature tensor.

    Expected x_mark column order:
        month, day, weekday, hour[, minute]
    The minute column is used only when freq == 't'.
    """

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super().__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        # Calendar indices must be integer tensors for nn.Embedding.
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    """Linear projection for continuous/engineered time features."""

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super().__init__()

        # Number of time features produced by the data loader for each frequency.
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    """Full embedding: value + temporal/calendar + positional."""

    def __init__(self,
                 c_in,
                 d_model,
                 embed_type='fixed',
                 freq='h',
                 dropout=0.1
    ):
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, x, x_mark):
        # All three components produce [B, L, D] and are combined by addition.
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_onlypos(nn.Module):
    """Embedding variant that uses only values and positional encodings."""

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self,x, x_mark):
        # x_mark is accepted for interface compatibility but intentionally unused.
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
    """Embedding variant that removes positional encoding."""

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, x_mark):
        # Value and temporal features already have shape [B, L, D].
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
