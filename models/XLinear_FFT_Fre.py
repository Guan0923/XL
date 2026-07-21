import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    """整体模型: Embed → Freq-domain Gating (FreTS) → Head 输出预测"""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.channel = configs.enc_in
        self.t_ff = configs.t_ff
        self.norm = configs.usenorm
        self.t_dropout = configs.t_dropout
        self.feature = configs.features
        self.channel_model = configs.channel_model
        self.hidden_size = configs.hidden_size

        self.backbone = FFT_FreqGating(
            self.seq_len,
            self.d_model,
            self.channel_model,
            self.channel,
            self.t_ff,
            self.t_dropout,
        )

        self.projection = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

    def forcast_multi(self, x_enc):
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev
        B, L, C = x_enc.shape
        x_emb = x_enc.permute(0, 2, 1)
        en = self.backbone(x_emb)
        en_flat = en.reshape(B, C, -1)
        dec_out = self.projection(en_flat)
        dec_out = dec_out.permute(0, 2, 1)

        if self.norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out

    def forward(self, x_enc, x_mark=None):
        return self.forcast_multi(x_enc)


class FFT_FreqGating(nn.Module):
    """
    FreTS + XLinear 风格频域门控模块:
      1. tokenEmb: [B, C, L] -> [B, C, L, d_model]
      2. 时间维频域门控（含 freq_glob_token）: rfft -> concat -> gate -> split -> irfft
      3. 通道维频域门控（含 glob 传入）: rfft -> concat -> gate -> split -> irfft
      4. 残差连接 + x_emb
    """

    def __init__(
        self, seq_len, d_model, channel_model, channel, t_ff, t_dropout
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.channel = channel
        self.channel_model = channel_model
        self.freq_dim = seq_len // 2 + 1

        self.token_emb = nn.Parameter(torch.randn(1, d_model))

        # ---- 频域可学习全局 token (类似 XLinear, 但在频域) ----
        scale = 0.02
        # 时间维频域: [1, channel, freq_dim, 1]
        self.freq_glob_real = nn.Parameter(
            scale * torch.randn(1, self.channel, self.freq_dim, 1)
        )
        self.freq_glob_imag = nn.Parameter(
            scale * torch.randn(1, self.channel, self.freq_dim, 1)
        )

        # ---- 门控模块 ----
        # en_attention (时间维): 输入 [B, C, F, d] + glob [B, C, F, d] -> 拼接后 2d
        self.freq_gating = Gating_Block(2 * self.d_model, t_ff, t_dropout)
        # ex_attention (通道维): 拼接 siganl + glob -> 2d
        self.freq_gating_channel = Gating_Block(2 * self.d_model, t_ff, t_dropout)

    def embedding(self, x):
        x = x.unsqueeze(3)
        return x @ self.token_emb

    def fft_gating_time(self, x):
        # x: [B, C, L, d]
        B, C, L, D = x.shape
        fft = torch.fft.rfft(x, dim=2, norm="ortho")  # [B, C, F, d]

        # 频域可学习模板, 沿 d 维拼接 (同 XLinear: concat(emb, glob_token))
        freq_glob = torch.complex(self.freq_glob_real, self.freq_glob_imag)
        freq_glob = freq_glob.expand(B, -1, -1, D)  # [B, C, F, d]

        fft_concat = torch.cat([fft, freq_glob], dim=-1)  # [B, C, F, 2d]
        fft_gated = self.freq_gating(fft_concat)           # [B, C, F, 2d]

        # 拆分: signal + glob (同 XLinear: origin_atten + glob_atten)
        fft_signal = fft_gated[:, :, :, :self.d_model]     # [B, C, F, d]
        fft_glob = fft_gated[:, :, :, self.d_model:]       # [B, C, F, d]

        # 都 irfft 回时域, glob 传给通道门控
        x_signal = torch.fft.irfft(fft_signal, n=self.seq_len, dim=2, norm="ortho")
        x_glob = torch.fft.irfft(fft_glob, n=self.seq_len, dim=2, norm="ortho")
        return x_signal, x_glob  # 均 [B, C, L, d]

    def fft_gating_channel(self, x_signal, x_glob):
        # x_signal, x_glob: [B, C, L, d]
        B, C, L, D = x_signal.shape

        # 信号 + glob 各自做通道维 FFT
        def _rfft_ch(x):
            x = x.permute(0, 2, 1, 3)  # [B, L, C, d]
            return torch.fft.rfft(x, dim=2, norm="ortho")  # [B, L, Cf, d]

        fft_sig = _rfft_ch(x_signal)  # [B, L, Cf, d]
        fft_glob = _rfft_ch(x_glob)   # [B, L, Cf, d]

        # 沿 d 维拼接 (同 fft_gating_time)
        fft_concat = torch.cat([fft_sig, fft_glob], dim=-1)  # [B, L, Cf, 2d]
        fft_gated = self.freq_gating_channel(fft_concat)      # [B, L, Cf, 2d]

        fft_sig_out = fft_gated[:, :, :, :self.d_model]      # [B, L, Cf, d]
        fft_glob_out = fft_gated[:, :, :, self.d_model:]     # [B, L, Cf, d]

        # 输出使用 signal + glob (同 XLinear: concat(origin_atten, glob))
        fft_out = fft_sig_out + fft_glob_out                  # [B, L, Cf, d]

        x_t_gated = torch.fft.irfft(
            fft_out, n=self.channel, dim=2, norm="ortho"
        )
        return x_t_gated.permute(0, 2, 1, 3)  # [B, C, L, d]

    def forward(self, x):
        B, C, L = x.shape

        x_emb = self.embedding(x)

        # Step 1: 时间维频域门控 -> 信号 + glob
        x_signal, x_glob = self.fft_gating_time(x_emb)

        # Step 2: 通道维频域门控, 使用 glob 做跨通道信息传递
        x_out = self.fft_gating_channel(x_signal, x_glob)

        # Step 3: 残差连接
        return x_out + x_emb


class Gating_Block(nn.Module):
    """
    复数域全连接门控 (FreTS 复数乘法 + XLinear 展宽融合).
    第1层: dim -> t_ff (全连接, 复数乘法保持相位信息).
    第2层: t_ff -> dim (全连接, Sigmoid 门控).
    """

    def __init__(self, dim, t_ff, dropout=0.0):
        super().__init__()
        scale = 0.02

        # 第1层: dim -> t_ff (复数全连接)
        self.W_r_1 = nn.Parameter(scale * torch.randn(dim, t_ff))
        self.W_i_1 = nn.Parameter(scale * torch.randn(dim, t_ff))
        self.B_r_1 = nn.Parameter(scale * torch.randn(t_ff))
        self.B_i_1 = nn.Parameter(scale * torch.randn(t_ff))

        # 第2层: t_ff -> dim (复数全连接)
        self.W_r_2 = nn.Parameter(scale * torch.randn(t_ff, dim))
        self.W_i_2 = nn.Parameter(scale * torch.randn(t_ff, dim))
        self.B_r_2 = nn.Parameter(scale * torch.randn(dim))
        self.B_i_2 = nn.Parameter(scale * torch.randn(dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        is_real = not torch.is_complex(x)
        if is_real:
            x = torch.complex(x, torch.zeros_like(x))

        # 复数全连接: 第1层 dim -> t_ff, 复数乘法保持相位
        r1 = F.relu(
            torch.einsum('...d,dh->...h', x.real, self.W_r_1) -
            torch.einsum('...d,dh->...h', x.imag, self.W_i_1) + self.B_r_1
        )
        i1 = F.relu(
            torch.einsum('...d,dh->...h', x.imag, self.W_r_1) +
            torch.einsum('...d,dh->...h', x.real, self.W_i_1) + self.B_i_1
        )
        r1, i1 = self.dropout(r1), self.dropout(i1)

        # 复数全连接: 第2层 t_ff -> dim, Sigmoid 门控
        r2 = torch.sigmoid(
            torch.einsum('...h,hd->...d', r1, self.W_r_2) -
            torch.einsum('...h,hd->...d', i1, self.W_i_2) + self.B_r_2
        )
        i2 = torch.sigmoid(
            torch.einsum('...h,hd->...d', i1, self.W_r_2) +
            torch.einsum('...h,hd->...d', r1, self.W_i_2) + self.B_i_2
        )

        out = x * torch.complex(r2, i2)
        return out.real if is_real else out
