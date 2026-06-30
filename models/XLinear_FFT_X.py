import torch
from torch import nn


class Model(nn.Module):
    """
    FFT denoising (DC-zero + threshold replace) → XLinear multi-series backbone.

    Pipeline:
      Input [B, L, C]
        → Norm/denorm (optional)
        → Permute to [B, C, L]
        → FFT_Denoise_XLinear backbone:
            Phase 1: FFT → DC-zero → high-freq replaced by learnable glob_token → IFFT
            Phase 2: XLinear-style projection + Gating_Block attention (endo/exo)
        → [B, C, 2*d_model]
        → Head: Linear(2*d_model, pred_len)
        → [B, pred_len, C]
    """

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
        self.threshold = configs.threshold

        if self.feature == 'M':
            self.backbone = FFT_Denoise_XLinear(
                self.seq_len, self.d_model, self.channel,
                self.threshold,
                self.t_ff, self.c_ff, self.t_dropout, self.c_dropout, self.embed_dropout
            )
        else:
            self.backbone = Forcast_with_exogenous(
                self.seq_len, self.d_model, self.channel, self.embed_dropout
            )

        # Head: backbone outputs [B, C, 2*d_model], project to pred_len
        self.head = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.Linear(2 * self.d_model, self.pred_len)
        )

    def forcast_multi(self, x_enc):
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # [B, L, C] → [B, C, L] for backbone (operates on channels)
        x_emb = x_enc.permute(0, 2, 1)
        en = self.backbone(x_emb)               # [B, C, 2*d_model]
        dec_out = self.head(en).permute(0, 2, 1)  # [B, pred_len, C]

        if self.norm:
            dec_out = dec_out * \
                (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forcast_exogenous(self, x_enc):
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
            return self.forcast_multi(x_enc)
        else:
            return self.forcast_exogenous(x_enc)


class FFT_Denoise_XLinear(nn.Module):
    """
    Phase 1: FFT denoising (DC-zero + threshold replace with learnable glob_token)
    Phase 2: XLinear multi-series backbone (projection + Gating_Block attention)

    Input:  [B, channel, seq_len]
    Output: [B, channel, 2*d_model]
    """

    def __init__(self, seq_len, d_model, channel, threshold,
                 t_ff, c_ff, t_dropout, c_dropout, embed_dropout):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.channel = channel
        self.threshold = threshold

        # ---- Phase 1: FFT Denoising params ----
        self.freq_dim = seq_len // 2 + 1
        self.threshold = min(self.threshold, self.freq_dim)

        if self.threshold < self.freq_dim:
            self.glob_dim = self.freq_dim - self.threshold
            self.glob_token_real = nn.Parameter(
                torch.zeros([1, channel, self.glob_dim]))
            self.glob_token_imag = nn.Parameter(
                torch.zeros([1, channel, self.glob_dim]))
            nn.init.xavier_uniform_(self.glob_token_real)
            nn.init.xavier_uniform_(self.glob_token_imag)
        else:
            self.glob_dim = 0

        # ---- Phase 2: XLinear multi-series backbone ----
        # Project denoised time-domain signal to d_model
        self.projection = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(embed_dropout)
        )

        # Learnable global token in the time-domain (XLinear style)
        self.glob_token = nn.Parameter(
            torch.ones([1, channel, d_model])
        )

        # Endogenous attention (channel-wise)
        self.en_attention = Gating_Block(2 * d_model, t_ff, t_dropout)
        # Exogenous / cross-channel attention
        self.ex_attention = Gating_Block(2 * channel, c_ff, c_dropout)

    def forward(self, x):
        """
        Input:  x — [B, channel, seq_len]
        Output:     [B, channel, 2*d_model]
        """
        B, C, L = x.shape

        # ===================== Phase 1: FFT Denoising =====================

        # ---- 1. FFT on raw seq_len dimension ----
        fft = torch.fft.rfft(x, dim=-1)  # [B, C, F], F = seq_len//2 + 1

        # ---- 2. Zero the DC component (index 0) — removes mean ----
        fft[:, :, 0] = 0

        # ---- 3. Low-freq保留 + high-freq用可学习 glob_token 替代 ----
        if self.glob_dim > 0:
            glob_token_fft = torch.complex(
                self.glob_token_real, self.glob_token_imag)  # [1, C, glob_dim]
            fft_replaced = torch.cat(
                [fft[:, :, :self.threshold], glob_token_fft.repeat(B, 1, 1)], dim=-1)
        else:
            # threshold == freq_dim，保留全部，无需替换
            fft_replaced = fft[:, :, :self.threshold]

        # ---- 4. IFFT back to time domain ----
        # Use n=self.seq_len to guarantee output length = seq_len
        x_denoised = torch.fft.irfft(fft_replaced, n=self.seq_len, dim=-1)
        # x_denoised: [B, C, seq_len] — cleaned signal

        # ===================== Phase 2: XLinear Backbone =====================

        # ---- 5. Project each channel's denoised seq to d_model ----
        emb = self.projection(x_denoised)  # [B, C, d_model]

        # ---- 6. Endogenous attention (channel-wise gating) ----
        glob_token_td = self.glob_token.repeat([B, 1, 1])  # [B, C, d_model]
        en_emb = torch.cat([emb, glob_token_td], dim=-1)   # [B, C, 2*d_model]

        en_atten = self.en_attention(en_emb)                # [B, C, 2*d_model]

        origin_atten = en_atten[:, :, :self.d_model]        # [B, C, d_model]
        glob_atten = en_atten[:, :, self.d_model:]          # [B, C, d_model]

        # ---- 7. Cross-channel exogenous attention ----
        ex_emb = torch.cat([emb, glob_atten], dim=1)        # [B, 2*C, d_model]
        ex_atten = self.ex_attention(ex_emb.permute(0, 2, 1))  # [B, d_model, 2*C]

        glob = ex_atten[:, :, self.channel:]                # [B, d_model, C]

        # ---- 8. Fuse endogenous + exogenous ----
        en = torch.cat(
            [origin_atten, glob.permute(0, 2, 1)], dim=-1
        )  # [B, C, 2*d_model]

        return en


class Forcast_with_exogenous(nn.Module):
    """
    Simplified exogenous path for features != 'M'.
    Projects to d_model, extracts the target channel, duplicates to 2*d_model
    to match the head's expected input dimension.
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
        embed = self.projection(x)              # [B, C, d_model]
        en = embed[:, -1:, :]                   # [B, 1, d_model]
        # Duplicate to match head's 2*d_model input
        en = torch.cat([en, en], dim=-1)        # [B, 1, 2*d_model]
        return en


class Gating_Block(nn.Module):
    """
    Gating block used in XLinear:
    y = x * sigmoid(Linear(ReLU(Linear(x))))
    """

    def __init__(self, d_model, hf, dropout=0.):
        super().__init__()
        self.d_model = d_model

        self.weight = nn.Sequential(
            nn.Linear(d_model, hf),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hf, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.weight(x)
        return x * weight
