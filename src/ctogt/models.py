"""Continuous-time organ graph transformer and comparator architectures."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CLINICAL_PRIOR_EDGES, ORGAN_CHANNELS, ORGAN_GROUPS


def build_prior_adjacency(
    channels: tuple[str, ...],
    organ_edges: tuple[tuple[str, str], ...],
) -> np.ndarray:
    channel_to_organ = {
        "gcs_total": "neurologic",
        "pupil_reactivity": "neurologic",
        "hr": "circulatory",
        "sbp": "circulatory",
        "dbp": "circulatory",
        "mbp": "circulatory",
        "lactate": "circulatory",
        "rr": "respiratory",
        "spo2": "respiratory",
        "fio2": "respiratory",
        "creatinine": "renal",
        "urine_output": "renal",
        "platelets": "coagulation",
        "inr": "coagulation",
        "aptt": "coagulation",
        "wbc": "inflammation",
        "temp": "inflammation",
        "glucose": "metabolic",
        "ph": "metabolic",
        "bicarbonate": "metabolic",
    }
    n = len(channels)
    adjacency = np.eye(n, dtype=np.float32)
    edge_set = {
        tuple(sorted(edge)) for edge in organ_edges
    }
    for left in range(n):
        for right in range(left + 1, n):
            left_organ = channel_to_organ[channels[left]]
            right_organ = channel_to_organ[channels[right]]
            if left_organ == right_organ:
                adjacency[left, right] = adjacency[right, left] = 1.0
            elif tuple(sorted((left_organ, right_organ))) in edge_set:
                adjacency[left, right] = adjacency[right, left] = 0.5
    return adjacency


def build_organ_prior(
    organ_edges: tuple[tuple[str, str], ...],
) -> tuple[np.ndarray, list[list[int]]]:
    organ_names = tuple(ORGAN_GROUPS)
    channel_indices = {
        organ: [
            index
            for index, channel in enumerate(ORGAN_CHANNELS)
            if channel in channels
        ]
        for organ, channels in ORGAN_GROUPS.items()
    }
    adjacency = np.eye(len(organ_names), dtype=np.float32)
    positions = {name: index for index, name in enumerate(organ_names)}
    for left, right in organ_edges:
        adjacency[positions[left], positions[right]] = 1.0
        adjacency[positions[right], positions[left]] = 1.0
    return adjacency, [channel_indices[name] for name in organ_names]


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def gradient_reverse(values: torch.Tensor, strength: float) -> torch.Tensor:
    return GradientReverse.apply(values, strength)


class SequenceStandardizer:
    def __init__(
        self,
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        static_median: np.ndarray,
        static_mean: np.ndarray,
        static_std: np.ndarray,
    ):
        self.channel_mean = channel_mean
        self.channel_std = channel_std
        self.static_median = static_median
        self.static_mean = static_mean
        self.static_std = static_std

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        masks: np.ndarray,
        static: np.ndarray,
    ) -> "SequenceStandardizer":
        means = []
        stds = []
        for channel in range(values.shape[2]):
            observed = values[:, :, channel][masks[:, :, channel] > 0]
            observed = observed[np.isfinite(observed)]
            means.append(float(observed.mean()) if observed.size else 0.0)
            std = float(observed.std()) if observed.size > 1 else 1.0
            stds.append(std if np.isfinite(std) and std > 1e-6 else 1.0)
        median = np.nanmedian(static, axis=0)
        median[~np.isfinite(median)] = 0.0
        filled = np.where(np.isfinite(static), static, median)
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std[~np.isfinite(std) | (std < 1e-6)] = 1.0
        return cls(
            np.asarray(means, dtype=np.float32),
            np.asarray(stds, dtype=np.float32),
            median.astype(np.float32),
            mean.astype(np.float32),
            std.astype(np.float32),
        )

    def transform_sequence(
        self,
        values: np.ndarray,
        masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        standardized = (
            values - self.channel_mean[None, None, :]
        ) / self.channel_std[None, None, :]
        standardized = np.where(masks > 0, standardized, 0.0)
        standardized = np.nan_to_num(
            standardized, nan=0.0, posinf=0.0, neginf=0.0
        )
        return standardized.astype(np.float32), (masks > 0).astype(np.float32)

    def transform_static(self, static: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(static), static, self.static_median)
        standardized = (filled - self.static_mean) / self.static_std
        return np.nan_to_num(
            standardized, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)


def continuous_time_features(
    masks: torch.Tensor,
    *,
    include_continuous_time: bool = True,
) -> torch.Tensor:
    batch, steps, channels = masks.shape
    if not include_continuous_time:
        return torch.zeros(
            batch, steps, channels, device=masks.device, dtype=masks.dtype
        )
    since = torch.ones_like(masks)
    for step in range(steps):
        observed = masks[:, step] > 0
        if step == 0:
            since[:, step] = torch.where(
                observed,
                torch.zeros_like(since[:, step]),
                since[:, step],
            )
        else:
            since[:, step] = torch.where(
                observed,
                torch.zeros_like(since[:, step]),
                torch.clamp(since[:, step - 1] + 1.0 / steps, max=1.0),
            )
    return since


def time2vec(steps: int, dimension: int, device: torch.device) -> torch.Tensor:
    positions = torch.linspace(0.0, 1.0, steps, device=device).unsqueeze(1)
    frequencies = torch.exp(
        torch.linspace(
            -math.log(16.0),
            math.log(16.0),
            max(1, dimension - 1),
            device=device,
        )
    ).unsqueeze(0)
    return torch.cat(
        [positions, torch.sin(positions * frequencies)],
        dim=1,
    )


class OrganGraphBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
        prior: np.ndarray | None,
        graph_mode: str = "dynamic",
        edge_dim: int = 16,
    ):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.graph_mode = graph_mode
        self.node_input = nn.Linear(5, hidden)
        self.query = nn.Linear(hidden, edge_dim)
        self.key = nn.Linear(hidden, edge_dim)
        self.value = nn.Linear(hidden, hidden)
        self.message = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        if prior is None:
            prior = np.eye(channels, dtype=np.float32)
        self.register_buffer("prior", torch.from_numpy(prior.astype(np.float32)))
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        since: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, channels = values.shape
        hours = torch.linspace(
            0.0, 1.0, steps, device=values.device, dtype=values.dtype
        ).view(1, steps, 1).expand(batch, steps, channels)
        repeated_prior = self.prior.unsqueeze(0).expand(
            batch * steps, channels, channels
        )
        node = self.node_input(
            torch.stack([values, masks, since, hours, hours * since], dim=-1)
        )
        node_flat = node.reshape(batch * steps, channels, self.hidden)
        q = self.query(node_flat)
        k = self.key(node_flat)
        v = self.value(node_flat)
        if self.graph_mode == "dynamic":
            logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
            logits = logits + self.prior_scale * torch.log1p(repeated_prior)
            observed = masks.reshape(batch * steps, channels)
            valid_keys = observed > 0
            all_missing = ~valid_keys.any(dim=1, keepdim=True)
            allowed = valid_keys | all_missing
            logits = logits.masked_fill(
                ~allowed.unsqueeze(1), torch.finfo(logits.dtype).min
            )
            adjacency = torch.softmax(logits, dim=-1)
        else:
            adjacency = repeated_prior / repeated_prior.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
        aggregated = torch.bmm(adjacency, v)
        updated = self.norm(node_flat + self.message(aggregated))
        return updated.reshape(batch, steps, channels, self.hidden), adjacency.reshape(
            batch, steps, channels, channels
        )


class TemporalFusionLite(nn.Module):
    """Small variable-selection + GRU + attention-pooling TFT comparator."""

    def __init__(self, channels: int, static_size: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.variable_selection = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, channels * 2),
            nn.Sigmoid(),
        )
        self.gru = nn.GRU(
            channels * 2, hidden, batch_first=True, bidirectional=False
        )
        self.attention = nn.Linear(hidden, 1)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(static_size),
            nn.Linear(static_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        static: torch.Tensor,
        since: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del since
        raw = torch.cat([values, masks], dim=-1)
        weights = self.variable_selection(raw)
        selected = raw * weights
        hidden, _ = self.gru(selected)
        score = self.attention(hidden).squeeze(-1)
        score = score.masked_fill(temporal_mask <= 0, -1e8)
        attention = torch.softmax(score, dim=1)
        pooled = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        representation = torch.cat(
            [pooled, self.static_encoder(static)], dim=-1
        )
        logits = self.head(representation).squeeze(-1)
        trajectory = logits.unsqueeze(1).expand(-1, values.shape[1])
        return {
            "logits": logits,
            "trajectory": trajectory,
            "representation": representation,
            "graph_penalty": logits.new_zeros(()),
        }


class GRUComparator(nn.Module):
    def __init__(self, channels: int, static_size: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(channels * 2, hidden, batch_first=True)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(static_size),
            nn.Linear(static_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        static: torch.Tensor,
        since: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del since
        hidden, _ = self.gru(torch.cat([values, masks], dim=-1))
        lengths = temporal_mask.sum(dim=1).long()
        last = (lengths - 1).clamp_min(0)
        pooled = hidden[torch.arange(len(hidden)), last]
        representation = torch.cat(
            [pooled, self.static_encoder(static)], dim=-1
        )
        logits = self.head(representation).squeeze(-1)
        trajectory = logits.unsqueeze(1).expand(-1, hidden.shape[1])
        return {
            "logits": logits,
            "trajectory": trajectory,
            "representation": representation,
            "graph_penalty": logits.new_zeros(()),
        }


class TemporalTransformerComparator(nn.Module):
    def __init__(
        self,
        channels: int,
        static_size: int,
        hidden: int = 64,
        dropout: float = 0.2,
        layers: int = 2,
    ):
        super().__init__()
        self.input = nn.Linear(channels * 3 + hidden, hidden)
        self.register_buffer(
            "position", torch.zeros(256, hidden), persistent=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(static_size),
            nn.Linear(static_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        static: torch.Tensor,
        since: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        steps = values.shape[1]
        encoding = time2vec(steps, self.input.out_features, values.device)
        hidden = self.input(
            torch.cat([values, masks, since, encoding.unsqueeze(0).expand(values.shape[0], -1, -1)], dim=-1)
        )
        hidden = self.encoder(
            hidden, src_key_padding_mask=temporal_mask <= 0
        )
        score = hidden.mean(dim=-1, keepdim=True)
        logits_steps = self.head(
            torch.cat(
                [
                    hidden,
                    self.static_encoder(static).unsqueeze(1).expand(
                        -1, steps, -1
                    ),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        score = score.masked_fill(temporal_mask.unsqueeze(-1) <= 0, -1e8)
        attention = torch.softmax(score.squeeze(-1), dim=1)
        pooled = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        representation = torch.cat(
            [pooled, self.static_encoder(static)], dim=-1
        )
        logits = self.head(representation).squeeze(-1)
        return {
            "logits": logits,
            "trajectory": logits_steps,
            "representation": representation,
            "graph_penalty": logits.new_zeros(()),
        }


class ContinuousOrganGraphNet(nn.Module):
    def __init__(
        self,
        channels: int,
        static_size: int,
        prior: np.ndarray,
        *,
        hidden: int = 64,
        dropout: float = 0.2,
        graph_mode: str = "dynamic",
        temporal_layers: int = 2,
        include_continuous_time: bool = True,
        use_environment_head: bool = False,
    ):
        super().__init__()
        self.include_continuous_time = include_continuous_time
        self.graph = OrganGraphBlock(
            channels,
            hidden,
            prior,
            graph_mode=graph_mode,
        )
        self.organ_score = nn.Linear(hidden, 1)
        self.time_projection = nn.Linear(hidden, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(static_size),
            nn.Linear(static_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.domain_head = (
            nn.Linear(hidden * 2, 8) if use_environment_head else None
        )

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        static: torch.Tensor,
        since: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        nodes, adjacency = self.graph(values, masks, since)
        node_score = self.organ_score(nodes)
        node_score = node_score.masked_fill(masks.unsqueeze(-1) <= 0, -1e8)
        organ_attention = torch.softmax(node_score.squeeze(-1), dim=-1)
        token = (nodes * organ_attention.unsqueeze(-1)).sum(dim=2)
        steps = values.shape[1]
        token = self.time_projection(
            token + time2vec(steps, token.shape[-1], token.device).unsqueeze(0)
        )
        hidden = self.temporal(
            token, src_key_padding_mask=temporal_mask <= 0
        )
        static_hidden = self.static_encoder(static)
        trajectory = self.head(
            torch.cat(
                [
                    hidden,
                    static_hidden.unsqueeze(1).expand(-1, steps, -1),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        score = trajectory.masked_fill(temporal_mask <= 0, -1e8)
        attention = torch.softmax(score, dim=1)
        pooled = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        representation = torch.cat([pooled, static_hidden], dim=-1)
        logits = self.head(representation).squeeze(-1)
        graph_penalty = torch.log1p(adjacency + 1e-6).mean()
        domain_logits = (
            self.domain_head(gradient_reverse(representation, 0.1))
            if self.domain_head is not None
            else None
        )
        return {
            "logits": logits,
            "trajectory": trajectory,
            "representation": representation,
            "graph_penalty": graph_penalty,
            "organ_attention": organ_attention,
            "temporal_attention": attention,
            "domain_logits": domain_logits,
        }


class OrganTokenGraphNet(nn.Module):
    """Dynamic graph over seven organ tokens rather than twenty channel nodes."""

    def __init__(
        self,
        static_size: int,
        prior: np.ndarray,
        organ_channels: list[list[int]],
        *,
        hidden: int = 64,
        dropout: float = 0.2,
        graph_mode: str = "dynamic",
        temporal_layers: int = 2,
        use_organ_rates: bool = False,
    ):
        super().__init__()
        self.register_buffer(
            "organ_channel_mask",
            torch.zeros(len(organ_channels), len(ORGAN_CHANNELS)),
        )
        for organ_index, indices in enumerate(organ_channels):
            self.organ_channel_mask[organ_index, indices] = 1.0
        self.graph = OrganGraphBlock(
            len(organ_channels),
            hidden,
            prior,
            graph_mode=graph_mode,
        )
        self.organ_score = nn.Linear(hidden, 1)
        self.time_projection = nn.Linear(hidden, hidden)
        self.use_organ_rates = bool(use_organ_rates)
        self.rate_projection = nn.Linear(1, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(static_size),
            nn.Linear(static_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        static: torch.Tensor,
        since: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        organ_mask_matrix = self.organ_channel_mask.unsqueeze(0).unsqueeze(0)
        channel_masks = masks.unsqueeze(2) * organ_mask_matrix
        denominators = channel_masks.sum(dim=-1).clamp_min(1.0)
        organ_values = (
            values.unsqueeze(2) * channel_masks
        ).sum(dim=-1) / denominators
        organ_since = (
            since.unsqueeze(2) * channel_masks
        ).sum(dim=-1) / denominators
        organ_masks = (channel_masks.sum(dim=-1) > 0).to(values.dtype)
        organ_since = torch.where(
            organ_masks > 0, organ_since, torch.ones_like(organ_since)
        )
        nodes, adjacency = self.graph(organ_values, organ_masks, organ_since)
        if self.use_organ_rates:
            organ_rates = torch.zeros_like(organ_values)
            organ_rates[:, 1:] = organ_values[:, 1:] - organ_values[:, :-1]
            organ_rates = organ_rates * organ_masks
            nodes = nodes + self.rate_projection(
                organ_rates.unsqueeze(-1)
            )
        node_score = self.organ_score(nodes).masked_fill(
            organ_masks.unsqueeze(-1) <= 0, -1e8
        )
        organ_attention = torch.softmax(node_score.squeeze(-1), dim=-1)
        token = (nodes * organ_attention.unsqueeze(-1)).sum(dim=2)
        steps = values.shape[1]
        token = self.time_projection(
            token + time2vec(steps, token.shape[-1], token.device).unsqueeze(0)
        )
        hidden = self.temporal(
            token, src_key_padding_mask=temporal_mask <= 0
        )
        static_hidden = self.static_encoder(static)
        trajectory = self.head(
            torch.cat(
                [
                    hidden,
                    static_hidden.unsqueeze(1).expand(-1, steps, -1),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        score = trajectory.masked_fill(temporal_mask <= 0, -1e8)
        attention = torch.softmax(score, dim=1)
        pooled = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        representation = torch.cat([pooled, static_hidden], dim=-1)
        logits = self.head(representation).squeeze(-1)
        return {
            "logits": logits,
            "trajectory": trajectory,
            "representation": representation,
            "graph_penalty": torch.log1p(adjacency + 1e-6).mean(),
            "temporal_attention": attention,
        }


@dataclass(frozen=True)
class DeepConfig:
    architecture: str = "ctogt"
    graph_mode: str = "dynamic"
    include_continuous_time: bool = True
    use_environment_head: bool = False
    use_environment_penalty: bool = False
    time_mode: str = "forward"
    random_graph: bool = False
    use_organ_rates: bool = False


def build_deep_model(
    config: DeepConfig,
    channels: int,
    static_size: int,
    prior: np.ndarray,
    *,
    hidden: int,
    dropout: float,
) -> nn.Module:
    if config.architecture == "gru":
        return GRUComparator(channels, static_size, hidden, dropout)
    if config.architecture == "transformer":
        return TemporalTransformerComparator(
            channels, static_size, hidden, dropout
        )
    if config.architecture == "tft":
        return TemporalFusionLite(channels, static_size, hidden, dropout)
    if config.architecture == "organ_graph":
        organ_prior, organ_channels = build_organ_prior(CLINICAL_PRIOR_EDGES)
        return OrganTokenGraphNet(
            static_size,
            organ_prior,
            organ_channels,
            hidden=hidden,
            dropout=dropout,
            graph_mode=config.graph_mode,
            use_organ_rates=config.use_organ_rates,
        )
    return ContinuousOrganGraphNet(
        channels,
        static_size,
        prior,
        hidden=hidden,
        dropout=dropout,
        graph_mode=config.graph_mode,
        include_continuous_time=config.include_continuous_time,
        use_environment_head=config.use_environment_head,
    )


def apply_time_mode(
    values: torch.Tensor,
    masks: torch.Tensor,
    since: torch.Tensor,
    time_mode: str,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if time_mode == "forward":
        return values, masks, since
    if time_mode == "reverse":
        return (
            torch.flip(values, dims=[1]),
            torch.flip(masks, dims=[1]),
            torch.flip(since, dims=[1]),
        )
    if time_mode == "permute":
        generator = torch.Generator(device=values.device)
        generator.manual_seed(int(seed))
        order = torch.randperm(values.shape[1], generator=generator).to(values.device)
        return values[:, order], masks[:, order], since[:, order]
    raise ValueError(f"Unknown time mode: {time_mode}")


def temporal_penalty(
    trajectory: torch.Tensor,
    temporal_mask: torch.Tensor,
    lambda_smooth: float,
    lambda_drift: float,
) -> torch.Tensor:
    valid = temporal_mask > 0
    pair = valid[:, 1:] & valid[:, :-1]
    differences = trajectory[:, 1:] - trajectory[:, :-1]
    smooth = (
        differences.square() * pair
    ).sum() / pair.sum().clamp_min(1.0)
    first = trajectory[:, 0]
    last_index = (
        temporal_mask.shape[1]
        - 1
        - torch.flip(temporal_mask, dims=[1]).argmax(dim=1)
    )
    last = trajectory[torch.arange(len(trajectory)), last_index]
    drift = (last - first).square().mean()
    return float(lambda_smooth) * smooth + float(lambda_drift) * drift


def environment_penalty(
    logits: torch.Tensor,
    labels: torch.Tensor,
    environments: torch.Tensor,
    representation: torch.Tensor,
    domain_logits: torch.Tensor | None,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    means = []
    for environment in environments.unique():
        selected = environments == environment
        if int(selected.sum()) >= 4:
            means.append(losses[selected].mean())
    if len(means) < 2:
        risk_variance = logits.sum() * 0.0
    else:
        stacked = torch.stack(means)
        risk_variance = (stacked - stacked.mean()).square().mean()
    domain_loss = logits.sum() * 0.0
    if domain_logits is not None:
        domain_target = torch.remainder(environments, domain_logits.shape[1])
        domain_loss = F.cross_entropy(
            domain_logits,
            domain_target,
        )
    return risk_variance + 0.05 * domain_loss


def train_deep_model(
    train_values: np.ndarray,
    train_masks: np.ndarray,
    train_static: np.ndarray,
    train_labels: np.ndarray,
    train_env: np.ndarray,
    valid_values: np.ndarray,
    valid_masks: np.ndarray,
    valid_static: np.ndarray,
    valid_labels: np.ndarray,
    valid_env: np.ndarray,
    *,
    config: DeepConfig,
    prior: np.ndarray,
    hidden: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    lambda_smooth: float,
    lambda_drift: float,
    lambda_environment: float,
    lambda_graph_sparsity: float,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(14, int(torch.get_num_threads()))))
    device = torch.device("cpu")
    random_prior = prior
    if config.random_graph:
        rng = np.random.default_rng(seed)
        random_prior = rng.uniform(size=prior.shape).astype(np.float32)
        random_prior = np.triu(random_prior, 1)
        random_prior = random_prior + random_prior.T + np.eye(len(prior))
    model = build_deep_model(
        config,
        train_values.shape[2],
        train_static.shape[1],
        random_prior,
        hidden=hidden,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    tensor_values = torch.from_numpy(train_values.astype(np.float32))
    tensor_masks = torch.from_numpy(train_masks.astype(np.float32))
    tensor_static = torch.from_numpy(train_static.astype(np.float32))
    tensor_labels = torch.from_numpy(train_labels.astype(np.float32))
    tensor_env = torch.from_numpy(train_env.astype(np.int64))
    valid_values_t = torch.from_numpy(valid_values.astype(np.float32))
    valid_masks_t = torch.from_numpy(valid_masks.astype(np.float32))
    valid_static_t = torch.from_numpy(valid_static.astype(np.float32))
    valid_labels_t = torch.from_numpy(valid_labels.astype(np.float32))
    valid_env_t = torch.from_numpy(valid_env.astype(np.int64))

    def predict_tensor(
        values: torch.Tensor,
        masks: torch.Tensor,
        static: torch.Tensor,
        env: torch.Tensor,
        *,
        time_mode: str,
        seed_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        logits_parts = []
        loss_parts = []
        with torch.no_grad():
            for start in range(0, len(values), 512):
                stop = min(start + 512, len(values))
                values_batch = values[start:stop]
                masks_batch = masks[start:stop]
                if time_mode == "permute":
                    order = torch.from_numpy(
                        np.random.default_rng(seed_offset + start).permutation(
                            values.shape[1]
                        ).astype(np.int64)
                    )
                    values_batch = values_batch[:, order]
                    masks_batch = masks_batch[:, order]
                elif time_mode == "reverse":
                    values_batch = torch.flip(values_batch, dims=[1])
                    masks_batch = torch.flip(masks_batch, dims=[1])
                since_batch = continuous_time_features(
                    masks_batch,
                    include_continuous_time=config.include_continuous_time,
                )
                temporal_mask = (masks_batch.sum(dim=-1) > 0).float()
                output = model(
                    values_batch,
                    masks_batch,
                    static[start:stop],
                    since_batch,
                    temporal_mask,
                )
                logits_parts.append(output["logits"])
                loss_parts.append(
                    F.binary_cross_entropy_with_logits(
                        output["logits"],
                        valid_labels_t[start:stop],
                        reduction="none",
                    )
                )
        return torch.cat(logits_parts), torch.cat(loss_parts)

    best_loss = np.inf
    best_state = None
    stale = 0
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(int(epochs)):
        model.train()
        order = torch.randperm(len(tensor_values), generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, len(order), int(batch_size)):
            indexes = order[start : start + int(batch_size)]
            values = tensor_values[indexes]
            masks = tensor_masks[indexes]
            if config.time_mode == "permute":
                permutations = []
                for local, index in enumerate(indexes.tolist()):
                    rng = np.random.default_rng(seed * 100000 + index)
                    permutations.append(
                        torch.from_numpy(
                            rng.permutation(values.shape[1]).astype(np.int64)
                        )
                    )
                perm = torch.stack(permutations)
                values = torch.gather(
                    values,
                    1,
                    perm.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
                )
                masks = torch.gather(
                    masks,
                    1,
                    perm.unsqueeze(-1).expand(-1, -1, masks.shape[-1]),
                )
            elif config.time_mode == "reverse":
                values = torch.flip(values, dims=[1])
                masks = torch.flip(masks, dims=[1])
            since = continuous_time_features(
                masks,
                include_continuous_time=config.include_continuous_time,
            )
            temporal_mask = (masks.sum(dim=-1) > 0).float()
            output = model(
                values,
                masks,
                tensor_static[indexes],
                since,
                temporal_mask,
            )
            loss = F.binary_cross_entropy_with_logits(
                output["logits"], tensor_labels[indexes]
            )
            loss = loss + temporal_penalty(
                output["trajectory"],
                temporal_mask,
                lambda_smooth,
                lambda_drift,
            )
            if config.use_environment_penalty:
                loss = loss + float(lambda_environment) * environment_penalty(
                    output["logits"],
                    tensor_labels[indexes],
                    tensor_env[indexes],
                    output["representation"],
                    output.get("domain_logits"),
                )
            if config.graph_mode != "none":
                loss = loss + float(lambda_graph_sparsity) * output[
                    "graph_penalty"
                ]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            batches += 1
        valid_logits, valid_losses = predict_tensor(
            valid_values_t,
            valid_masks_t,
            valid_static_t,
            valid_env_t,
            time_mode=config.time_mode,
            seed_offset=7_000_000,
        )
        valid_loss = float(valid_losses.mean())
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_deep_model(
    model: nn.Module,
    values: np.ndarray,
    masks: np.ndarray,
    static: np.ndarray,
    *,
    config: DeepConfig,
    batch_size: int = 512,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    values_t = torch.from_numpy(values.astype(np.float32))
    masks_t = torch.from_numpy(masks.astype(np.float32))
    static_t = torch.from_numpy(static.astype(np.float32))
    model.eval()
    probability_parts = []
    organ_parts = []
    temporal_parts = []
    with torch.no_grad():
        for start in range(0, len(values_t), int(batch_size)):
            stop = min(start + int(batch_size), len(values_t))
            batch_values = values_t[start:stop]
            batch_masks = masks_t[start:stop]
            if config.time_mode == "permute":
                order = torch.from_numpy(
                    np.random.default_rng(seed + start).permutation(
                        values.shape[1]
                    ).astype(np.int64)
                )
                batch_values = batch_values[:, order]
                batch_masks = batch_masks[:, order]
            elif config.time_mode == "reverse":
                batch_values = torch.flip(batch_values, dims=[1])
                batch_masks = torch.flip(batch_masks, dims=[1])
            since = continuous_time_features(
                batch_masks,
                include_continuous_time=config.include_continuous_time,
            )
            temporal_mask = (batch_masks.sum(dim=-1) > 0).float()
            output = model(
                batch_values,
                batch_masks,
                static_t[start:stop],
                since,
                temporal_mask,
            )
            probability_parts.append(
                torch.sigmoid(output["logits"]).numpy()
            )
            if "organ_attention" in output:
                organ_parts.append(output["organ_attention"].mean(dim=1).numpy())
            if "temporal_attention" in output:
                temporal_parts.append(
                    output["temporal_attention"].numpy()
                )
    result = {"probability": np.concatenate(probability_parts)}
    if organ_parts:
        result["organ_attention"] = np.concatenate(organ_parts)
    if temporal_parts:
        result["temporal_attention"] = np.concatenate(temporal_parts)
    return result
