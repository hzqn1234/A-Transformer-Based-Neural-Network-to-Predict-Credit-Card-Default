from __future__ import annotations

import math

import torch
import torch.nn as nn


class SequencePositionEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 25):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[0]]


class AmexTransformer(nn.Module):
    """AMEX Transformer/GRU model for sequence and optional tabular features."""

    def __init__(
        self,
        series_dim: int,
        feature_dim: int = 0,
        hidden_dim: int = 256,
        transformer_layers: int = 3,
        attention_heads: int = 32,
        feedforward_dim: int = 256,
        transformer_dropout: float = 0.05,
        feature_dropout: float = 0.01,
        output_dropout: float = 0.1,
        feature_hidden_layers: int = 3,
        positional_encoding: str = "sinusoidal",
    ):
        super().__init__()
        self.use_features = feature_dim > 0
        self.input_series_block = nn.Sequential(nn.Linear(series_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        if positional_encoding == "sinusoidal":
            self.position_encoding: nn.Module | None = SequencePositionEncoding(hidden_dim)
            encoded_dim = hidden_dim * 2
        elif positional_encoding == "none":
            self.position_encoding = None
            encoded_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported positional_encoding: {positional_encoding}")
        layer = nn.TransformerEncoderLayer(
            d_model=encoded_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=transformer_dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.gru_series = nn.GRU(encoded_dim, encoded_dim, batch_first=True, bidirectional=True)

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

        classifier_dim = encoded_dim * 2 + (hidden_dim if self.use_features else 0)
        self.output_block = nn.Sequential(
            nn.BatchNorm1d(classifier_dim),
            nn.Linear(classifier_dim, hidden_dim),
            nn.Dropout(output_dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _last_gru_output(self, encoded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            encoded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru_series(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        row_idx = torch.arange(output.shape[0], device=output.device)
        last_idx = lengths.to(output.device) - 1
        return output[row_idx, last_idx]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        series = self.input_series_block(batch["series"])
        encoded = series.permute(1, 0, 2)
        if self.position_encoding is not None:
            encoded = torch.cat([encoded, self.position_encoding(encoded)], dim=2)
        encoded = self.transformer_encoder(encoded).permute(1, 0, 2)
        pooled = self._last_gru_output(encoded, batch["lengths"])
        if self.use_features:
            feature_repr = self.feature_block(batch["features"])
            pooled = torch.cat([pooled, feature_repr], dim=1)
        return self.output_block(pooled)
