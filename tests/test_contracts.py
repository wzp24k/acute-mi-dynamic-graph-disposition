from __future__ import annotations

import numpy as np
import torch

from ctogt.common import CLINICAL_PRIOR_EDGES, ORGAN_CHANNELS
from ctogt.models import (
    DeepConfig,
    build_deep_model,
    build_prior_adjacency,
    continuous_time_features,
)


def test_prior_adjacency() -> None:
    adjacency = build_prior_adjacency(ORGAN_CHANNELS, CLINICAL_PRIOR_EDGES)
    assert adjacency.shape == (len(ORGAN_CHANNELS), len(ORGAN_CHANNELS))
    assert np.allclose(adjacency, adjacency.T)
    assert np.all(np.diag(adjacency) == 1)
    assert np.all(adjacency >= 0)


def test_continuous_time_features() -> None:
    masks = torch.tensor(
        [[[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]]
    )
    features = continuous_time_features(masks)
    assert features.shape == masks.shape
    assert features[0, 0, 0] == 0
    assert features[0, 2, 0] == 0
    assert features[0, 2, 1] == 0


def test_full_model_shapes() -> None:
    torch.manual_seed(0)
    channels = len(ORGAN_CHANNELS)
    prior = build_prior_adjacency(ORGAN_CHANNELS, CLINICAL_PRIOR_EDGES)
    model = build_deep_model(
        DeepConfig(
            architecture="ctogt",
            graph_mode="dynamic",
            use_environment_head=True,
        ),
        channels,
        static_size=12,
        prior=prior,
        hidden=32,
        dropout=0.1,
    )
    masks = (torch.rand(3, 24, channels) > 0.6).float()
    values = torch.randn(3, 24, channels) * masks
    static = torch.randn(3, 12)
    since = continuous_time_features(masks)
    temporal_mask = (masks.sum(dim=-1) > 0).float()
    output = model(values, masks, static, since, temporal_mask)
    assert output["logits"].shape == (3,)
    assert output["trajectory"].shape == (3, 24)
    assert output["organ_attention"].shape == (3, 24, channels)
    assert output["temporal_attention"].shape == (3, 24)


def test_organ_token_graph_shapes() -> None:
    torch.manual_seed(1)
    channels = len(ORGAN_CHANNELS)
    prior = build_prior_adjacency(ORGAN_CHANNELS, CLINICAL_PRIOR_EDGES)
    model = build_deep_model(
        DeepConfig(architecture="organ_graph", graph_mode="dynamic"),
        channels,
        static_size=12,
        prior=prior,
        hidden=32,
        dropout=0.1,
    )
    masks = (torch.rand(2, 24, channels) > 0.5).float()
    values = torch.randn(2, 24, channels) * masks
    static = torch.randn(2, 12)
    since = continuous_time_features(masks)
    temporal_mask = (masks.sum(dim=-1) > 0).float()
    output = model(values, masks, static, since, temporal_mask)
    assert output["logits"].shape == (2,)
    assert output["trajectory"].shape == (2, 24)


def test_organ_token_rate_graph_shapes() -> None:
    torch.manual_seed(2)
    channels = len(ORGAN_CHANNELS)
    prior = build_prior_adjacency(ORGAN_CHANNELS, CLINICAL_PRIOR_EDGES)
    model = build_deep_model(
        DeepConfig(
            architecture="organ_graph",
            graph_mode="dynamic",
            use_organ_rates=True,
        ),
        channels,
        static_size=12,
        prior=prior,
        hidden=32,
        dropout=0.1,
    )
    masks = (torch.rand(2, 24, channels) > 0.5).float()
    values = torch.randn(2, 24, channels) * masks
    static = torch.randn(2, 12)
    since = continuous_time_features(masks)
    temporal_mask = (masks.sum(dim=-1) > 0).float()
    output = model(values, masks, static, since, temporal_mask)
    assert output["logits"].shape == (2,)
    assert output["trajectory"].shape == (2, 24)
