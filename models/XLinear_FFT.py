import torch
from torch import nn
import math
from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.channel = configs.enc_in
        self.t_ff = configs.t_ff
        self.c_ff = configs.c_ff
        self.norm = configs.usenorm
        self.embed_dropout = configs.embed_dropout
        self.head_dropout = configs.head_dropout
        self.t_dropout = configs.t_dropout
        self.c_dropout = configs.c_dropout
        self.feature = configs.features

        # Frequency-specific params
        self.top_k = configs.top_k
        self.threshold = configs.threshold

        # DataEmbedding: [B, L, C] → [B, L, d_model] with pos + temporal embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        if self.feature == 'M':
            # Backbone 直接作用于原始通道 [B, C, L]（不使用 embedding 升维）
            self.backbone = Forcast_multi_Freq(
                self.seq_len, self.d_model, self.channel,
                self.top_k, self.threshold
            )
        else:
            self.backbone = Forcast_with_exogenous(
                self.seq_len, self.d_model, self.channel, self.embed_dropout
            )

        # Head: backbone 输出 [B, C, d_model]，将 d_model 投影到 pred_len
        self.head = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

    def forcast_multi(self, x_enc, x_mark=None):
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # 不使用 DataEmbedding，直接在原始通道上做频域去噪
        # [B, L, C] → [B, C, L]
        x_emb = x_enc.permute(0, 2, 1)
        en = self.backbone(x_emb)               # [B, C, d_model]
        dec_out = self.head(en).permute(0, 2, 1)  # [B, pred_len, C]

        if self.norm:
            dec_out = dec_out * \
                (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forcast_exogenous(self, x_enc):
        """Same norm/denorm pipeline, uses self.backbone for feature != 'M'."""
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        en = self.backbone(x_enc)
        dec_out = self.head(en).permute(0, 2, 1)

        if self.norm:
            dec_out = dec_out * \
                (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark=None):
        if self.feature == 'M':
            return self.forcast_multi(x_enc, x_mark)
        else:
            return self.forcast_exogenous(x_enc)


class Forcast_multi_Freq(nn.Module):
    """
    Multi-variate forecasting backbone with a frequency-domain pipeline:

    FFT → DC-zero → threshold(high-freq replaced by glob_token) → top-K(period)
    → IFFT → period-patch → Inception → aggregate
    """

    def __init__(self, seq_len, d_model, channel, top_k, threshold):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.channel = channel
        self.top_k = top_k
        self.threshold = threshold
        # 用于可视化：保存 batch 0 的中间结果（输入 / IFFT 输出）
        self.input_b0 = None
        self.irfft_b0 = None
        # rfft output dimension (based on raw seq_len)
        self.freq_dim = seq_len // 2 + 1
        # threshold 不能超过 freq_dim，否则切片会被静默截断
        self.threshold = min(self.threshold, self.freq_dim)

        if self.threshold < self.freq_dim:
            # 需被替代的高频个数 = freq_dim - threshold
            self.glob_dim = self.freq_dim - self.threshold
            # Learnable glob_token in the frequency domain (complex-valued)
            self.glob_token_real = nn.Parameter(
                torch.zeros([1, channel, self.glob_dim]))
            self.glob_token_imag = nn.Parameter(
                torch.zeros([1, channel, self.glob_dim]))
            nn.init.xavier_uniform_(self.glob_token_real)
            nn.init.xavier_uniform_(self.glob_token_imag)
        else:
            # threshold == freq_dim: 无需替换任何频率，不创建 glob_token
            self.glob_dim = 0

        # 频域维度 = threshold(低频保留) + glob_dim(高频去噪) = freq_dim
        self.freq_concat_dim = self.threshold + self.glob_dim
        # irfft 输出长度 = 2*(freq_concat_dim - 1) = seq_len
        self.irfft_out_len = 2 * (self.freq_concat_dim - 1)
        # Project the irfft result to d_model for head
        self.time_proj = nn.Linear(self.irfft_out_len, d_model)

        # Inception block for intra/inter-period interaction
        self.inception = Inception_Block_V1(channel, channel, num_kernels=6)

    def forward(self, x):
        """
        Input:  x — [B, channel, seq_len]
        Output:     [B, channel, d_model]
        """
        B, C, L = x.shape

        #TODO 1
        # 保存 batch 0 的输入序列用于可视化 [C, L]
        self.input_b0 = x[0].detach()

        # ---- 1. FFT directly on raw seq_len dimension ----
        fft = torch.fft.rfft(x, dim=-1)  # [B, C, F],  F = seq_len//2 + 1

        # ---- 2. Zero the DC component (index 0) — removes mean ----
        fft[:, :, 0] = 0

        # ---- 3. Threshold: 用 glob_token 替代高频噪声部分----
        # 低频前 threshold 个保留，高频 glob_dim 个用可学习 token 替代
        if self.glob_dim > 0:
            glob_token = torch.complex(
                self.glob_token_real, self.glob_token_imag)  # [1, C, glob_dim]
            fft[:, :, self.threshold:]=0
            fft_replaced = fft
        else:
            # threshold == freq_dim，保留全部频率，无需替换
            fft_replaced = fft[:, :, :self.threshold]
        # fft_replaced 长度 = threshold + glob_dim = freq_dim

        # ---- 4. Compute amplitude on the replaced freq-domain for period selection ----
        amp = torch.abs(fft_replaced)       # [B, C, F]
        amp_global = amp.mean(0).mean(-1)   # [F] (对 batch 和 channel 求均值)

        # ---- 5. Top-K: select K highest-amplitude frequencies to get periods ----
        K = min(self.top_k, self.freq_dim)
        if K <= 0:
            K = 1
        _, topk_indices = torch.topk(amp_global, k=K, dim=-1)  # [B, K]

        # ---- 6. IFFT back to time domain ----
        time_out = torch.fft.irfft(
            fft_replaced, dim=-1)  # [B, C, irfft_out_len]
        
        #TODO 2
        # 保存 batch 0 的 IFFT 重建序列用于可视化 [C, irfft_out_len]
        self.irfft_b0 = time_out[0].detach()

        # ---- 7. Period patch + Inception for each top-K frequency ----
        freq_idxs = topk_indices                               # [K]

        period_results = []
        for i in range(K):
            freq_idx = freq_idxs[i].item()
            if freq_idx <= 0:
                continue

            # Period based on irfft output length
            P = self.irfft_out_len // freq_idx
            if P < 2:
                P = 2

            # Pad irfft output to be divisible by P
            L_out = self.irfft_out_len
            n_frames = math.ceil(L_out / P)
            pad_len = n_frames * P - L_out
            if pad_len > 0:
                padded = torch.nn.functional.pad(time_out, (0, pad_len))
            else:
                padded = time_out

            # Reshape to [B, C, P, n_frames]
            patched = padded.reshape(B, C, P, n_frames)

            # Inception_Block_V1: [B, C, H, W] -> [B, C, H, W]
            conv_out = self.inception(patched)

            # Flatten back and project to d_model
            flat = conv_out.reshape(B, C, n_frames * P)
            flat = flat[:, :, :L_out]
            # [B, C, d_model]
            flat = self.time_proj(flat)

            period_results.append(flat)

        if len(period_results) == 0:
            return self.time_proj(time_out)

        # ---- 11. Aggregate results across periods (mean) ----
        out = torch.stack(period_results, dim=0).mean(
            dim=0)   # [B, C, d_model]

        return out


class Forcast_with_exogenous(nn.Module):
    """
    Exogenous forecasting path (for features == 'S' or 'MS').
    Simplified version: projects to d_model and extracts the target channel.
    """

    def __init__(self, seq_len, d_model, channel, embed_dropout):
        super().__init__()
        self.d_model = d_model
        self.channel = channel
        self.projection = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(embed_dropout)
        )

    def forward(self, x):
        # x: [B, C, seq_len]
        embed = self.projection(x)          # [B, C, d_model]
        # Use the last channel as the target
        en = embed[:, -1:, :]               # [B, 1, d_model]
        return en
