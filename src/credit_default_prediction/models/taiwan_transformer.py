from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 64):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class TaiwanTransformer(nn.Module):
    def __init__(
        self,
        series_dim: int,
        feature_dim: int = 0,
        hidden_dim: int = 256,
        transformer_layers: int = 3,
        attention_heads: int = 16,
        feedforward_dim: int = 256,
        transformer_dropout: float = 0.05,
        feature_dropout: float = 0.01,
        output_dropout: float = 0.05,
        feature_hidden_layers: int = 2,
        positional_encoding: str = "none",
        use_padding_mask: bool = False,
        gru_pooling: str = "hidden",
    ):
        super().__init__()
        self.use_features = feature_dim > 0
        self.use_padding_mask = use_padding_mask
        if gru_pooling not in {"hidden", "last_output"}:
            raise ValueError(f"Unsupported gru_pooling: {gru_pooling}")
        self.gru_pooling = gru_pooling
        self.input_block = nn.Sequential(nn.Linear(series_dim, hidden_dim), nn.LayerNorm(hidden_dim))

        if positional_encoding == "sinusoidal":
            self.position = SinusoidalEncoding(hidden_dim)
        elif positional_encoding == "none":
            self.position = nn.Identity()
        else:
            raise ValueError(f"Unsupported positional_encoding: {positional_encoding}")

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.pooler = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        if self.use_features:
            blocks: list[nn.Module] = [
                nn.Linear(feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(feature_dropout),
                nn.LeakyReLU(),
            ]
            for _ in range(max(feature_hidden_layers - 1, 0)):
                blocks.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.Dropout(feature_dropout),
                        nn.LeakyReLU(),
                    ]
                )
            self.feature_block = nn.Sequential(*blocks)

        classifier_dim = hidden_dim * (3 if self.use_features else 2)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(classifier_dim),
            nn.Linear(classifier_dim, hidden_dim),
            nn.Dropout(output_dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        series = self.input_block(batch["series"])
        series = self.position(series)
        padding_mask = ~batch["mask"] if self.use_padding_mask else None
        encoded = self.encoder(series, src_key_padding_mask=padding_mask)

        lengths = batch["lengths"].detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            encoded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.pooler(packed)
        if self.gru_pooling == "last_output":
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            row_idx = torch.arange(output.shape[0], device=output.device)
            last_idx = batch["lengths"].to(output.device) - 1
            pooled = output[row_idx, last_idx]
        else:
            pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)

        if self.use_features:
            feature_repr = self.feature_block(batch["features"])
            pooled = torch.cat([pooled, feature_repr], dim=1)
        return self.classifier(pooled)
