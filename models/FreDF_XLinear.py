import torch
from torch import nn

from layers.Embed import DataEmbedding
from models.FreDF import FFT_for_Decomp
from models.XLinear import Forcast_multi, Forcast_with_exogenous


class FreDFBranch(nn.Module):
    """FreDF forecasting path adapted to the two-argument experiment API."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel = configs.enc_in
        self.d_model = configs.d_model
        self.layers = max(1, configs.e_layers)

        self.history_projection = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
        )
        self.embedding = DataEmbedding(
            self.channel, self.d_model, configs.embed, configs.freq,
            configs.dropout,
        )
        self.frequency_layer = FFT_for_Decomp(
            (self.seq_len + self.pred_len) // 2 + 1,
            self.d_model,
        )
        self.frequency_dropout = nn.Dropout(configs.dropout)
        self.future_projection = nn.Sequential(
            nn.Linear(self.pred_len, self.pred_len),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
        )
        self.output_projection = nn.Linear(self.d_model, self.channel)

    def forward(self, x):
        # FreDF: history mapping -> zero future -> embedding -> residual FFT.
        history = self.history_projection(x.permute(0, 2, 1))
        history = history.permute(0, 2, 1)
        future = x.new_zeros(x.shape[0], self.pred_len, self.channel)
        representation = torch.cat([history, future], dim=1)

        # Future calendar marks are unavailable in the current experiment API;
        # value and positional embeddings preserve FreDF's compatible path.
        representation = self.embedding(representation, None)
        for _ in range(self.layers):
            residual = representation
            representation = self.frequency_layer(representation)
            representation = residual + self.frequency_dropout(representation)

        future_latent = representation[:, -self.pred_len:, :]
        future_latent = self.future_projection(
            future_latent.permute(0, 2, 1)
        ).permute(0, 2, 1)
        prediction = self.output_projection(future_latent)
        return prediction


class Model(nn.Module):
    """Full FreDF forecast fused dynamically with an XLinear forecast."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.channel = configs.enc_in
        self.norm = configs.usenorm
        self.feature = configs.features
        self.output_channels = self.channel if self.feature == 'M' else 1

        if self.feature == 'M':
            self.backbone = Forcast_multi(
                self.seq_len, self.d_model, self.channel, configs.t_ff,
                configs.c_ff, configs.t_dropout, configs.c_dropout,
                configs.embed_dropout,
            )
        else:
            self.backbone = Forcast_with_exogenous(
                self.seq_len, self.d_model, self.channel, configs.t_ff,
                configs.c_ff, configs.t_dropout, configs.c_dropout,
                configs.embed_dropout,
            )

        self.xlinear_head = nn.Sequential(
            nn.Dropout(configs.head_dropout),
            nn.Linear(2 * self.d_model, self.pred_len),
        )
        self.fredf = FreDFBranch(configs)

        # Convert the FreDF forecast into the same feature width as XLinear.
        self.fredf_feature = nn.Linear(self.pred_len, 2 * self.d_model)
        self.fusion_gate = nn.Sequential(
            nn.Linear(4 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )
        # Begin near the XLinear baseline while keeping a useful FreDF
        # gradient. The gate can then adapt per sample/channel/horizon.
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.constant_(self.fusion_gate[-1].bias, 2.0)
        self.last_fusion_gate = None

    def _select_output_channels(self, x):
        if self.feature == 'M':
            return x
        return x[:, :, -1:]

    def _forecast_normalized(self, x_enc):
        xlinear_features = self.backbone(x_enc.permute(0, 2, 1))
        xlinear_prediction = self.xlinear_head(xlinear_features)
        xlinear_prediction = xlinear_prediction.permute(0, 2, 1)

        fredf_prediction = self._select_output_channels(self.fredf(x_enc))
        fredf_features = self.fredf_feature(
            fredf_prediction.permute(0, 2, 1)
        )

        gate_input = torch.cat([xlinear_features, fredf_features], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(gate_input)).permute(0, 2, 1)
        self.last_fusion_gate = gate.detach()

        return (
            gate * xlinear_prediction
            + (1.0 - gate) * fredf_prediction
        )

    def forward(self, x_enc, x_mark=None):
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = centered / stdev

        dec_out = self._forecast_normalized(x_enc)

        if self.norm:
            stats_slice = slice(None) if self.feature == 'M' else slice(-1, None)
            scale = stdev[:, 0, stats_slice].unsqueeze(1)
            shift = means[:, 0, stats_slice].unsqueeze(1)
            dec_out = dec_out * scale + shift

        return dec_out
