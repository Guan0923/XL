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
            # Backbone operates on [B, d_model, L] (channel = d_model after embedding)
            self.backbone = Forcast_multi_Freq(
                self.seq_len, self.d_model, self.d_model,
                self.top_k, self.threshold
            )
        else:
            self.backbone = Forcast_with_exogenous(
                self.seq_len, self.d_model, self.channel,
                self.t_ff, self.c_ff,
                self.t_dropout, self.c_dropout, self.embed_dropout
            )

        # Head: backbone outputs [B, d_model, d_model]
        self.head = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

        # Project back from d_model → C for denorm
        self.final_proj = nn.Linear(self.d_model, self.channel)

    def forcast_multi(self, x_enc, x_mark=None):
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # DataEmbedding: [B, L, C] → [B, L, d_model]
        x_emb = self.enc_embedding(x_enc, x_mark)

        # Permute [B, d_model, L] for backbone
        x_emb = x_emb.permute(0, 2, 1)         # [B, d_model, L]
        en = self.backbone(x_emb)               # [B, d_model, d_model]
        dec_out = self.head(en).permute(0, 2, 1)  # [B, pred_len, d_model]

        # Project back to C channels for denorm
        dec_out = self.final_proj(dec_out)      # [B, pred_len, C]

        if self.norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forcast_exogenous(self, x_enc):
        """Same norm/denorm pipeline, uses self.backbone for feature != 'M'."""
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        en = self.backbone(x_enc)
        dec_out = self.head(en).permute(0, 2, 1)

        if self.norm:
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark=None):
        if self.feature == 'M':
            return self.forcast_multi(x_enc, x_mark)
        else:
            return self.forcast_exogenous(x_enc)


class Forcast_multi_Freq(nn.Module):
    """
    Multi-variate forecasting backbone with a frequency-domain pipeline:

    FFT → DC-zero → threshold → top-K → concat(glob_token, freq-domain)
    → IFFT → period-patch → Inception → aggregate
    """
    def __init__(self, seq_len, d_model, channel, top_k, threshold):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.channel = channel
        self.top_k = top_k
        self.threshold = threshold
        self.freq_dim = seq_len // 2 + 1  # rfft output dimension (based on raw seq_len)

        # Learnable glob_token in the frequency domain (complex-valued)
        self.glob_token_real = nn.Parameter(torch.zeros([1, channel, self.freq_dim]))
        self.glob_token_imag = nn.Parameter(torch.zeros([1, channel, self.freq_dim]))
        nn.init.xavier_uniform_(self.glob_token_real)
        nn.init.xavier_uniform_(self.glob_token_imag)

        # After irfft: input len = 2*(2*F-1) = 2*seq_len + 2
        self.irfft_out_len = 2 * (2 * self.freq_dim - 1)
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

        # ---- 1. FFT directly on raw seq_len dimension ----
        fft = torch.fft.rfft(x, dim=-1)  # [B, C, F],  F = seq_len//2 + 1

        # ---- 2. Zero the DC component (index 0) — removes mean ----
        fft[:, :, 0] = 0

        # ---- 3. Compute amplitude for frequency selection ----
        amp = torch.abs(fft)             # [B, C, F]

        # ---- 4. Average amplitude across channels (global ranking) ----
        amp_global = amp.mean(dim=1)     # [B, F]

        # ---- 5. Threshold: frequencies above threshold → discarded (zeroed) ----
        freq_range = torch.arange(self.freq_dim, device=x.device)
        below_threshold = freq_range <= self.threshold         # [F] bool

        masked_amp = amp_global.clone()
        masked_amp[:, ~below_threshold] = 0                   # zero out above-threshold

        # ---- 6. Top-K: select K highest-amplitude frequencies from the valid range ----
        K = min(self.top_k, int(below_threshold.sum().item()))
        if K <= 0:
            K = 1
        _, topk_indices = torch.topk(masked_amp, k=K, dim=-1)  # [B, K]

        # ---- 7. Keep only top-K frequencies, zero everything else ----
        fft_denoised = torch.zeros_like(fft)
        topk_idx = topk_indices.unsqueeze(1).expand(-1, C, -1)  # [B, C, K]
        fft_denoised.scatter_(
            dim=-1, index=topk_idx,
            src=fft.gather(dim=-1, index=topk_idx)
        )

        # ---- 8. Concat glob_token in the frequency domain (like original XLinear's concat) ----
        glob_token = torch.complex(self.glob_token_real, self.glob_token_imag)  # [1, C, F]
        freq_with_glob = torch.cat(
            [fft_denoised, glob_token.expand(B, -1, -1)], dim=-1
        )  # [B, C, 2*F]

        # ---- 9. IFFT directly from concat'd freq domain ----
        time_out = torch.fft.irfft(freq_with_glob, dim=-1)  # [B, C, L_out], L_out = 2*seq_len + 2

        # ---- 10. Period patch + Inception for each top-K frequency ----
        freq_idxs = topk_indices[0]                            # [K]

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
            flat = self.time_proj(flat)                         # [B, C, d_model]

            period_results.append(flat)

        if len(period_results) == 0:
            return self.time_proj(time_out)

        # ---- 11. Aggregate results across periods (mean) ----
        out = torch.stack(period_results, dim=0).mean(dim=0)   # [B, C, d_model]

        return out


class Forcast_with_exogenous(nn.Module):
    """
    Exogenous forecasting path (for features == 'S' or 'MS').
    Simplified version: projects to d_model and extracts the target channel.
    """
    def __init__(self, seq_len, d_model, channel, t_ff, c_ff, t_dropout, c_dropout, embed_dropout):
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