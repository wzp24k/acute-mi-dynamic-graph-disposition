"""Ordinal sequence models and temporal-structure experiments."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_CLASSES = 5


def cumulative_ordinal_probabilities(
    latent: torch.Tensor,
    thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cumulative = torch.sigmoid(latent[:, None] - thresholds[None, :])
    probabilities = torch.cat(
        [
            1.0 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ],
        dim=1,
    ).clamp_min(1e-7)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    return probabilities, cumulative


class CumulativeOrdinalHead(nn.Module):
    """Shared latent score with K-1 ordered cumulative logits."""

    def __init__(self, hidden: int, n_classes: int = N_CLASSES):
        super().__init__()
        if n_classes < 2:
            raise ValueError("Ordinal outcome requires at least two classes")
        self.n_classes = n_classes
        self.score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )
        self.base = nn.Parameter(torch.tensor(0.0))
        self.gaps = nn.Parameter(torch.zeros(n_classes - 2))

    def thresholds(self) -> torch.Tensor:
        if self.n_classes == 2:
            return self.base.reshape(1)
        gaps = F.softplus(self.gaps)
        return torch.cat(
            [self.base.reshape(1), self.base.reshape(1) + torch.cumsum(gaps, dim=0)],
            dim=0,
        )

    def forward(self, representation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.score(representation).squeeze(1)
        probabilities, cumulative = cumulative_ordinal_probabilities(
            latent, self.thresholds()
        )
        return probabilities, cumulative


class DynamicFeatureGraph(nn.Module):
    """Attention graph over physiological channels at each ICU hour."""

    def __init__(self, n_features: int, edge_dim: int = 16):
        super().__init__()
        self.edge_dim = edge_dim
        self.edge = nn.Parameter(torch.randn(n_features, edge_dim) * 0.05)
        self.query = nn.Linear(edge_dim + 2, edge_dim)
        self.key = nn.Linear(edge_dim + 2, edge_dim)
        self.update = nn.Linear(2, 1)

    def forward(
        self,
        values: torch.Tensor,
        observed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, features = values.shape
        observed = observed.to(dtype=values.dtype)
        nodes = values * observed
        edge = self.edge[None, None].expand(batch, steps, features, -1)
        token = torch.cat([edge, nodes.unsqueeze(-1), observed.unsqueeze(-1)], dim=-1)
        query = self.query(token)
        key = self.key(token)
        logits = query @ key.transpose(-2, -1) / math.sqrt(self.edge_dim)
        valid = observed > 0
        all_missing = ~valid.any(dim=-1, keepdim=True)
        allowed = valid | all_missing
        logits = logits.masked_fill(
            ~allowed.unsqueeze(-2), torch.finfo(logits.dtype).min
        )
        adjacency = torch.softmax(logits, dim=-1)
        adjacency = adjacency * (~all_missing).to(adjacency.dtype).unsqueeze(-1)
        messages = adjacency @ nodes.unsqueeze(-1)
        gate = torch.sigmoid(
            self.update(torch.cat([messages, nodes.unsqueeze(-1)], dim=-1))
        )
        return nodes + gate.squeeze(-1) * messages.squeeze(-1), adjacency


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        logits = self.score(hidden).squeeze(-1)
        valid = valid > 0
        no_valid = ~valid.any(dim=1)
        if no_valid.any():
            valid = valid.clone()
            valid[no_valid, -1] = True
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        return (weights.unsqueeze(-1) * hidden).sum(dim=1)


class StaticEncoder(nn.Module):
    def __init__(self, n_static: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_static),
            nn.Linear(n_static, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        return self.net(static)


class TabularOrdinalNet(nn.Module):
    """Tabular MLP comparator with the same cumulative ordinal head."""

    def __init__(
        self,
        n_features: int,
        *,
        hidden: int = 128,
        dropout: float = 0.2,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = CumulativeOrdinalHead(hidden, n_classes)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(features)
        return self.head(representation)


class TemporalOrdinalNet(nn.Module):
    """GRU, DG-GRU, or STGCN-lite with a cumulative ordinal head."""

    def __init__(
        self,
        n_sequence: int,
        n_static: int,
        *,
        architecture: str = "dggru",
        hidden: int = 64,
        edge_dim: int = 16,
        dropout: float = 0.2,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        if n_sequence % 2:
            raise ValueError("Sequence tensor must contain values and masks")
        self.architecture = architecture
        self.n_values = n_sequence // 2
        self.hidden_size = hidden
        self.graph = (
            DynamicFeatureGraph(self.n_values, edge_dim)
            if architecture == "dggru"
            else None
        )
        self.spatial = (
            nn.Conv1d(self.n_values, self.n_values, kernel_size=1)
            if architecture == "stgcn_lite"
            else None
        )
        self.gru = nn.GRU(
            input_size=n_sequence,
            hidden_size=hidden,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)
        self.projection = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.pool = MaskedAttentionPool(hidden)
        self.static = StaticEncoder(n_static, hidden, dropout)
        self.head = CumulativeOrdinalHead(hidden * 2, n_classes)
        self.trajectory = nn.Linear(hidden, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        static: torch.Tensor,
        *,
        return_temporal_cumulative: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        values = sequence[:, :, : self.n_values]
        observed = sequence[:, :, self.n_values :]
        if self.graph is not None:
            values, _ = self.graph(values, observed)
        elif self.spatial is not None:
            fixed = torch.roll(torch.eye(self.n_values, device=values.device), 1, 1)
            mixed = torch.einsum("ij,btj->bti", fixed, values)
            values = values + F.gelu(self.spatial(mixed.transpose(1, 2)).transpose(1, 2))
        model_input = torch.cat([values, observed], dim=-1)
        hidden, _ = self.gru(model_input)
        hidden = F.gelu(self.projection(self.norm(hidden)))
        valid = observed.sum(dim=-1) > 0
        pooled = self.pool(hidden, valid)
        static_encoded = self.static(static)
        representation = torch.cat([self.dropout(pooled), static_encoded], dim=1)
        probabilities, cumulative = self.head(representation)
        trajectory = self.trajectory(hidden).squeeze(-1)
        if return_temporal_cumulative:
            expanded_static = static_encoded[:, None, :].expand(
                -1, hidden.shape[1], -1
            )
            temporal_representation = torch.cat(
                [hidden, expanded_static], dim=-1
            )
            _, temporal_cumulative = self.head(
                temporal_representation.reshape(-1, hidden.shape[-1] * 2)
            )
            temporal_cumulative = temporal_cumulative.reshape(
                hidden.shape[0], hidden.shape[1], -1
            )
            return probabilities, trajectory, valid, temporal_cumulative
        return probabilities, trajectory, valid


class GRUDOrdinalNet(nn.Module):
    """GRU-D with learned input and hidden-state decay."""

    def __init__(
        self,
        n_sequence: int,
        n_static: int,
        *,
        hidden: int = 64,
        dropout: float = 0.2,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        if n_sequence % 2:
            raise ValueError("Sequence tensor must contain values and masks")
        self.n_values = n_sequence // 2
        self.hidden_size = hidden
        self.register_buffer("imputation_mean", torch.zeros(self.n_values))
        self.input_decay = nn.Linear(self.n_values, self.n_values)
        self.hidden_decay = nn.Linear(self.n_values, hidden)
        self.cell = nn.GRUCell(self.n_values * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.projection = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.pool = MaskedAttentionPool(hidden)
        self.static = StaticEncoder(n_static, hidden, dropout)
        self.head = CumulativeOrdinalHead(hidden * 2, n_classes)
        self.trajectory = nn.Linear(hidden, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = sequence[:, :, : self.n_values]
        observed = sequence[:, :, self.n_values :]
        batch, steps, _ = values.shape
        hidden = torch.zeros(batch, self.hidden_size, device=values.device)
        last = self.imputation_mean.unsqueeze(0).expand(batch, -1).clone()
        delta = torch.ones(batch, self.n_values, device=values.device)
        hidden_steps = []
        for hour in range(steps):
            obs = observed[:, hour]
            current = values[:, hour]
            gamma_x = torch.exp(-F.softplus(self.input_decay(delta)))
            decayed = gamma_x * last + (1.0 - gamma_x) * self.imputation_mean
            imputed = obs * current + (1.0 - obs) * decayed
            gamma_h = torch.exp(-F.softplus(self.hidden_decay(delta)))
            hidden = gamma_h * hidden
            hidden = self.cell(torch.cat([imputed, obs], dim=1), hidden)
            hidden_steps.append(hidden)
            last = obs * current + (1.0 - obs) * last
            delta = torch.where(obs > 0, torch.ones_like(delta), delta + 1.0)
        hidden_sequence = torch.stack(hidden_steps, dim=1)
        hidden_sequence = F.gelu(
            self.projection(self.norm(hidden_sequence))
        )
        valid = observed.sum(dim=-1) > 0
        representation = torch.cat(
            [self.dropout(self.pool(hidden_sequence, valid)), self.static(static)],
            dim=1,
        )
        probabilities, cumulative = self.head(representation)
        trajectory = self.trajectory(hidden_sequence).squeeze(-1)
        return probabilities, trajectory, valid


class TransformerOrdinalNet(nn.Module):
    def __init__(
        self,
        n_sequence: int,
        n_static: int,
        *,
        hidden: int = 64,
        dropout: float = 0.2,
        heads: int = 4,
        layers: int = 2,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.hidden_size = hidden
        self.n_values = n_sequence // 2
        self.input = nn.Linear(n_sequence, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.projection = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.pool = MaskedAttentionPool(hidden)
        self.static = StaticEncoder(n_static, hidden, dropout)
        self.head = CumulativeOrdinalHead(hidden * 2, n_classes)
        self.trajectory = nn.Linear(hidden, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed = sequence[:, :, self.n_values :]
        valid = observed.sum(dim=-1) > 0
        padding = ~valid.clone()
        padding[~valid.any(dim=1), -1] = False
        steps = sequence.shape[1]
        position = torch.arange(steps, device=sequence.device).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, self.hidden_size, 2, device=sequence.device)
            * (-math.log(10000.0) / self.hidden_size)
        )
        encoding = torch.zeros(
            steps, self.hidden_size, device=sequence.device
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        hidden = self.input(sequence) + encoding[None]
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        hidden = F.gelu(self.projection(self.norm(hidden)))
        representation = torch.cat(
            [self.dropout(self.pool(hidden, valid)), self.static(static)], dim=1
        )
        probabilities, cumulative = self.head(representation)
        trajectory = self.trajectory(hidden).squeeze(-1)
        return probabilities, trajectory, valid


def build_network(
    architecture: str,
    n_sequence: int,
    n_static: int,
    *,
    hidden: int = 64,
    edge_dim: int = 16,
    dropout: float = 0.2,
    n_classes: int = N_CLASSES,
) -> nn.Module:
    if architecture in {"gru", "dggru", "stgcn_lite"}:
        return TemporalOrdinalNet(
            n_sequence,
            n_static,
            architecture=architecture,
            hidden=hidden,
            edge_dim=edge_dim,
            dropout=dropout,
            n_classes=n_classes,
        )
    if architecture == "grud":
        return GRUDOrdinalNet(
            n_sequence,
            n_static,
            hidden=hidden,
            dropout=dropout,
            n_classes=n_classes,
        )
    if architecture == "transformer":
        return TransformerOrdinalNet(
            n_sequence,
            n_static,
            hidden=hidden,
            dropout=dropout,
            n_classes=n_classes,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def _first_last_valid(
    trajectory: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = valid > 0
    index = torch.arange(trajectory.shape[1], device=trajectory.device)[None]
    first_index = torch.where(mask, index, trajectory.shape[1]).min(dim=1).values
    last_index = torch.where(mask, index, -1).max(dim=1).values
    safe_first = first_index.clamp_max(trajectory.shape[1] - 1)
    safe_last = last_index.clamp_min(0)
    return (
        trajectory.gather(1, safe_first[:, None]).squeeze(1),
        trajectory.gather(1, safe_last[:, None]).squeeze(1),
        mask.any(dim=1),
    )


def temporal_regularization(
    trajectory: torch.Tensor,
    valid: torch.Tensor,
    *,
    lambda_smooth: float,
    lambda_drift: float,
    mode: str = "smooth_drift",
    drift_target: float = 0.0,
) -> torch.Tensor:
    """Direction-agnostic trajectory penalty with prespecified controls."""
    if mode == "permuted":
        batch, steps = trajectory.shape
        shifts = torch.randint(
            1, steps, (batch,), device=trajectory.device
        )
        indices = (
            torch.arange(steps, device=trajectory.device)[None]
            + shifts[:, None]
        ) % steps
        trajectory = torch.gather(trajectory, 1, indices)
    pair_valid = valid[:, 1:] * valid[:, :-1]
    difference = trajectory[:, 1:] - trajectory[:, :-1]
    smooth = (difference.square() * pair_valid).sum() / pair_valid.sum().clamp_min(1.0)
    first, last, any_valid = _first_last_valid(trajectory, valid)
    drift = ((last - first - drift_target).square() * any_valid).sum() / any_valid.sum().clamp_min(1.0)
    if mode in {"smooth_drift", "permuted"}:
        return lambda_smooth * smooth + lambda_drift * drift
    if mode == "smooth_only":
        return lambda_smooth * smooth
    if mode in {"monotone_decrease", "monotone_increase"}:
        direction = -1.0 if mode == "monotone_decrease" else 1.0
        monotone = (
            F.relu(-difference * direction) * pair_valid
        ).sum() / pair_valid.sum().clamp_min(1.0)
        return (
            lambda_smooth * smooth
            + lambda_drift * drift
            + lambda_smooth * monotone
        )
    raise ValueError(f"Unknown temporal regularization mode: {mode}")


def cumulative_curvature_regularization(
    cumulative_probabilities: torch.Tensor,
    valid: torch.Tensor,
    *,
    lambda_curvature: float,
) -> torch.Tensor:
    """Penalize jitter in outcome probabilities while permitting monotone trends."""
    if cumulative_probabilities.shape[1] < 3:
        return cumulative_probabilities.sum() * 0.0
    curvature = (
        cumulative_probabilities[:, 2:]
        - 2.0 * cumulative_probabilities[:, 1:-1]
        + cumulative_probabilities[:, :-2]
    )
    triple_valid = (
        valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]
    ).unsqueeze(-1)
    denominator = triple_valid.sum().clamp_min(1.0)
    penalty = (curvature.square() * triple_valid).sum() / denominator
    return lambda_curvature * penalty


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        static: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        self.sequences = torch.from_numpy(sequences.astype(np.float32))
        self.static = torch.from_numpy(static.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        if sample_weight is None:
            sample_weight = np.ones(len(labels), dtype=np.float32)
        self.sample_weight = torch.from_numpy(
            np.asarray(sample_weight, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.sequences[index],
            self.static[index],
            self.labels[index],
            self.sample_weight[index],
        )


@dataclass
class SequencePreprocessor:
    value_mean: np.ndarray
    value_std: np.ndarray
    static_median: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        masks: np.ndarray,
        static: np.ndarray,
    ) -> "SequencePreprocessor":
        mean = []
        std = []
        for index in range(values.shape[2]):
            observed = values[:, :, index][masks[:, :, index] > 0]
            observed = observed[np.isfinite(observed)]
            mean.append(float(observed.mean()) if len(observed) else 0.0)
            std.append(float(observed.std()) if len(observed) > 1 else 1.0)
        mean_array = np.asarray(mean, dtype=np.float32)
        std_array = np.asarray(std, dtype=np.float32)
        std_array[~np.isfinite(std_array) | (std_array < 1e-6)] = 1.0
        median = np.nanmedian(static, axis=0)
        median[~np.isfinite(median)] = 0.0
        filled = np.where(np.isfinite(static), static, median)
        static_mean = filled.mean(axis=0)
        static_std = filled.std(axis=0)
        static_std[static_std < 1e-6] = 1.0
        return cls(
            value_mean=mean_array,
            value_std=std_array,
            static_median=median.astype(np.float32),
            static_mean=static_mean.astype(np.float32),
            static_std=static_std.astype(np.float32),
        )

    def transform_sequence(
        self,
        values: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        standardized = (
            values - self.value_mean[None, None, :]
        ) / self.value_std[None, None, :]
        standardized = np.where(
            np.isfinite(standardized), standardized, 0.0
        ).astype(np.float32)
        return np.concatenate(
            [standardized, (masks > 0).astype(np.float32)], axis=2
        )

    def transform_static(self, static: np.ndarray) -> np.ndarray:
        filled = np.where(
            np.isfinite(static), static, self.static_median[None, :]
        )
        standardized = (filled - self.static_mean[None, :]) / self.static_std[
            None, :
        ]
        return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )


@dataclass
class TabularPreprocessor:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "TabularPreprocessor":
        median = np.nanmedian(features, axis=0)
        median[~np.isfinite(median)] = 0.0
        filled = np.where(np.isfinite(features), features, median)
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(
            median=median.astype(np.float32),
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        filled = np.where(
            np.isfinite(features), features, self.median[None, :]
        )
        standardized = (filled - self.mean[None, :]) / self.std[None, :]
        return np.nan_to_num(
            standardized, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)


def class_sample_weights(
    labels: np.ndarray,
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return weights[labels.astype(int)].astype(np.float32)


def train_temporal_model(
    sequences: np.ndarray,
    static: np.ndarray,
    labels: np.ndarray,
    *,
    architecture: str,
    use_regularization: bool = False,
    regularization_kind: str = "scalar_trajectory",
    regularization_mode: str = "smooth_drift",
    lambda_smooth: float = 0.0,
    lambda_drift: float = 0.0,
    lambda_curvature: float = 0.0,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: int = 64,
    edge_dim: int = 16,
    dropout: float = 0.2,
    seed: int = 42,
    n_classes: int | None = None,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if n_classes is None:
        n_classes = int(np.max(labels)) + 1
    model = build_network(
        architecture,
        n_sequence=sequences.shape[2],
        n_static=static.shape[1],
        hidden=hidden,
        edge_dim=edge_dim,
        dropout=dropout,
        n_classes=n_classes,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    weights = class_sample_weights(labels, n_classes)
    dataset = SequenceDataset(sequences, static, labels, weights)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    for _ in range(epochs):
        model.train()
        for sequence, static_batch, target, sample_weight in loader:
            cumulative_regularization = (
                use_regularization
                and regularization_kind == "cumulative_curvature"
            )
            if cumulative_regularization:
                output = model(
                    sequence,
                    static_batch,
                    return_temporal_cumulative=True,
                )
            else:
                output = model(sequence, static_batch)
            if cumulative_regularization:
                probabilities, trajectory, valid, temporal_cumulative = output
            else:
                probabilities, trajectory, valid = output
            losses = F.nll_loss(
                torch.log(probabilities), target, reduction="none"
            )
            loss = (losses * sample_weight).sum() / sample_weight.sum().clamp_min(
                1e-12
            )
            if cumulative_regularization:
                loss = loss + cumulative_curvature_regularization(
                    temporal_cumulative,
                    valid,
                    lambda_curvature=lambda_curvature,
                )
            elif use_regularization:
                loss = loss + temporal_regularization(
                    trajectory,
                    valid,
                    lambda_smooth=lambda_smooth,
                    lambda_drift=lambda_drift,
                    mode=regularization_mode,
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model


def train_tabular_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: int = 128,
    dropout: float = 0.2,
    seed: int = 42,
    n_classes: int | None = None,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if n_classes is None:
        n_classes = int(np.max(labels)) + 1
    model = TabularOrdinalNet(
        features.shape[1],
        hidden=hidden,
        dropout=dropout,
        n_classes=n_classes,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    weights = class_sample_weights(labels, n_classes)
    tensors = torch.utils.data.TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
        torch.from_numpy(weights.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    for _ in range(epochs):
        model.train()
        for feature_batch, target, sample_weight in loader:
            probabilities, _ = model(feature_batch)
            losses = F.nll_loss(
                torch.log(probabilities), target, reduction="none"
            )
            loss = (losses * sample_weight).sum() / sample_weight.sum().clamp_min(
                1e-12
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model


def predict_tabular_model(
    model: nn.Module,
    features: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            batch = torch.from_numpy(features[start:stop].astype(np.float32))
            probs, _ = model(batch)
            probabilities.append(probs.numpy())
    return np.concatenate(probabilities).astype(np.float32)


def predict_temporal_model(
    model: nn.Module,
    sequences: np.ndarray,
    static: np.ndarray,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    trajectories = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            stop = min(start + batch_size, len(sequences))
            sequence = torch.from_numpy(
                sequences[start:stop].astype(np.float32)
            )
            static_batch = torch.from_numpy(static[start:stop].astype(np.float32))
            probs, trajectory, _ = model(sequence, static_batch)
            probabilities.append(probs.numpy())
            trajectories.append(trajectory.numpy())
    return (
        np.concatenate(probabilities).astype(np.float32),
        np.concatenate(trajectories).astype(np.float32),
    )
