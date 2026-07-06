import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.

    Input/Output shape:
        x: [B, L, C]
        return: [B, L, C]

    where:
        B = batch size
        L = input sequence length
        C = channel/variable count
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x):
        # x: [B, L, C]
        # Repeat the first and last time steps so AvgPool1d keeps the sequence length.
        # front/end: [B, (kernel_size - 1) // 2, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)

        # After padding: [B, L + kernel_size - 1, C]
        x = torch.cat([front, x, end], dim=1)

        # AvgPool1d expects [B, C, L], so move channel before time.
        # Pooled shape: [B, C, L]
        x = self.avg(x.permute(0, 2, 1))
        # Move back to the project's common time-series format: [B, L, C]
        return x.permute(0, 2, 1)  

class series_decomp(nn.Module):
    """
    Series decomposition block.

    It decomposes a time series into:
        residual/seasonal = original - moving average
        trend = moving average
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # x: [B, L, C]
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        # res: [B, L, C], moving_mean: [B, L, C]
        return res, moving_mean

class Model(nn.Module):
    """
    DLinear model: decompose to `trend` & `seasonal` parts,
    then adds a linear layer.

    Expected input:
        x: [B, seq_len, enc_in]

    Output:
        y: [B, pred_len, enc_in]
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomposition kernel size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            # Each channel has its own pair of Linear layers.
            # Every Linear maps the time dimension: seq_len -> pred_len.
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Decoder.append(nn.Linear(self.seq_len, self.pred_len))

        else:
            # All channels share the same Linear layers.
            # nn.Linear applies to the last dimension, so the forward pass will
            # first permute tensors from [B, L, C] to [B, C, L].
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Decoder = nn.Linear(self.seq_len, self.pred_len)

            # Initialize linear weights as an averaging projection over seq_len.
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        # x: [B, seq_len, C]
        # seasonal_init: residual/seasonal component, [B, seq_len, C]
        # trend_init: moving-average trend component, [B, seq_len, C]
        seasonal_init, trend_init = self.decompsition(x)

        # Linear layers map the last dimension, so put time at the end:
        # [B, seq_len, C] -> [B, C, seq_len]
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            # Allocate output tensors after time projection:
            # [B, C, pred_len]
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                # For channel i:
                # seasonal_init[:, i, :]: [B, seq_len]
                # Linear_Seasonal[i](...): [B, pred_len]
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # Shared Linear layers operate on the last dimension:
            # [B, C, seq_len] -> [B, C, pred_len]
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # Add seasonal and trend predictions: [B, C, pred_len]
        x = seasonal_output + trend_output
        # Return to [B, pred_len, C].
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
