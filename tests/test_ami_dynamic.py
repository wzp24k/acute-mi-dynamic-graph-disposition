from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from src.ami_dynamic.common import (
    CHANNELS,
    DEMOGRAPHIC_FEATURES,
    build_sequence_tensor,
    four_to_three,
    five_to_four,
    five_to_three,
    map_five_level,
    map_six_level,
)
from src.ami_dynamic.build_cohorts import _attach_outcomes
from src.ami_dynamic.evaluation import (
    _metric_scalar,
    delong_auc_test,
    point_metrics,
)
from src.ami_dynamic.models import (
    CumulativeOrdinalHead,
    TabularOrdinalNet,
    cumulative_curvature_regularization,
    cumulative_ordinal_probabilities,
    temporal_regularization,
    train_temporal_model,
)
from src.ami_dynamic.train_evaluate import (
    TEMPORAL_MODELS,
    _assert_feature_contract,
    _feature_columns,
)
from work.run_temporal_controls import _trajectory_summary
from work.run_external_calibration_common3 import patient_safe_hospital_clusters


def test_forward_carry_keeps_observation_mask_zero_and_never_backfills() -> None:
    hourly = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "hour": [1, 4],
            "channel": ["hr", "hr"],
            "value": [80.0, 90.0],
        }
    )
    values, masks, ids = build_sequence_tensor(
        hourly,
        [1],
        channels=CHANNELS,
        max_hour=8,
        carry_hours=2,
    )
    hr_index = CHANNELS.index("hr")
    assert ids == [1]
    assert np.isnan(values[0, 0, hr_index])
    assert values[0, 1, hr_index] == 80.0
    assert values[0, 2, hr_index] == 80.0
    assert values[0, 3, hr_index] == 80.0
    assert not np.isnan(values[0, 4, hr_index])
    assert values[0, 4, hr_index] == 90.0
    assert masks[0, 1, hr_index] == 1.0
    assert masks[0, 2:4, hr_index].tolist() == [0.0, 0.0]


def test_cumulative_ordinal_probabilities_are_monotone_and_normalized() -> None:
    latent = torch.tensor([-1.0, 0.2, 1.3], dtype=torch.float32)
    thresholds = torch.tensor([-0.8, 0.0, 0.7, 1.5], dtype=torch.float32)
    probabilities, cumulative = cumulative_ordinal_probabilities(
        latent, thresholds
    )
    assert torch.all(cumulative[:, 1:] <= cumulative[:, :-1])
    assert torch.all(probabilities >= 0)
    assert torch.allclose(
        probabilities.sum(dim=1), torch.ones(len(latent)), atol=1e-6
    )


def test_four_class_ordinal_head_supports_external_outcome() -> None:
    head = CumulativeOrdinalHead(hidden=8, n_classes=4)
    representation = torch.randn(6, 8)
    probabilities, cumulative = head(representation)
    assert probabilities.shape == (6, 4)
    assert cumulative.shape == (6, 3)
    assert torch.allclose(
        probabilities.sum(dim=1), torch.ones(6), atol=1e-6
    )


def test_temporal_regularization_is_direction_agnostic() -> None:
    positive = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    negative = -positive
    valid = torch.ones_like(positive)
    positive_loss = temporal_regularization(
        positive,
        valid,
        lambda_smooth=0.2,
        lambda_drift=0.05,
    )
    negative_loss = temporal_regularization(
        negative,
        valid,
        lambda_smooth=0.2,
        lambda_drift=0.05,
    )
    assert torch.isclose(positive_loss, negative_loss, atol=1e-7)


def test_cumulative_curvature_penalty_ignores_constant_drift() -> None:
    valid = torch.ones(2, 5)
    linear = torch.tensor(
        [
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.6, 0.5, 0.4, 0.3, 0.2],
        ]
    ).unsqueeze(-1)
    jittered = linear.clone()
    jittered[0, 1, 0] += 0.1
    linear_loss = cumulative_curvature_regularization(
        linear, valid, lambda_curvature=0.01
    )
    jittered_loss = cumulative_curvature_regularization(
        jittered, valid, lambda_curvature=0.01
    )
    assert linear_loss.item() == pytest.approx(0.0, abs=1e-10)
    assert jittered_loss > linear_loss


def test_dggru_static_matches_full_input_but_disables_raw_sequence() -> None:
    static_spec = TEMPORAL_MODELS["dggru_static"]
    sequence_spec = TEMPORAL_MODELS["dggru"]
    assert static_spec["architecture"] == sequence_spec["architecture"]
    assert static_spec["input_kind"] == sequence_spec["input_kind"] == "full"
    assert static_spec.get("use_sequence", True) is False
    assert sequence_spec.get("use_sequence", True) is True


def test_patient_safe_hospital_clusters_connect_shared_patients() -> None:
    patients = np.array(["a", "a", "b", "c"])
    hospitals = np.array(["h1", "h2", "h2", "h3"])
    clusters, metadata = patient_safe_hospital_clusters(patients, hospitals)

    assert metadata["n_hospitals"] == 3
    assert metadata["n_hospital_clusters"] == 2
    assert metadata["n_cross_hospital_patients"] == 1
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[3] != clusters[0]
    for patient in np.unique(patients):
        assert len(np.unique(clusters[patients == patient])) == 1


def test_full_feature_contract_retains_all_measurement_counts() -> None:
    frame = pd.DataFrame(
        {
            **{column: np.zeros(2) for column in DEMOGRAPHIC_FEATURES},
            "troponin_assay_t": np.zeros(2),
            "gcs_change": np.zeros(2),
            **{
                f"{channel}_count": np.ones(2)
                for channel in CHANNELS
            },
        }
    )
    _, full_columns = _feature_columns(frame)
    assert {
        f"{channel}_count" for channel in CHANNELS
    }.issubset(full_columns)


def test_tabular_ordinal_network_outputs_normalized_probabilities() -> None:
    model = TabularOrdinalNet(12, hidden=16, n_classes=5)
    probabilities, cumulative = model(torch.randn(4, 12))
    assert probabilities.shape == (4, 5)
    assert cumulative.shape == (4, 4)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-6)


@pytest.mark.parametrize("architecture", ["grud", "transformer"])
def test_non_cumulative_temporal_models_accept_standard_training_call(
    architecture: str,
) -> None:
    rng = np.random.default_rng(20261105)
    sequence = rng.normal(size=(8, 3, 4)).astype(np.float32)
    sequence[:, :, 2:] = 1.0
    static = rng.normal(size=(8, 5)).astype(np.float32)
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=int)
    model = train_temporal_model(
        sequence,
        static,
        labels,
        architecture=architecture,
        epochs=1,
        batch_size=4,
        hidden=8,
        n_classes=4,
    )
    with torch.no_grad():
        probabilities, trajectory, valid = model(
            torch.from_numpy(sequence),
            torch.from_numpy(static),
        )
    assert probabilities.shape == (8, 4)
    assert trajectory.shape == (8, 3)
    assert valid.shape == (8, 3)


def test_trajectory_summary_accepts_numpy_arrays() -> None:
    trajectory = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    masks = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    summary = _trajectory_summary(trajectory, masks)
    assert np.isclose(summary["mean_adjacent_squared_change"], 2.0)
    assert np.isclose(summary["mean_absolute_net_drift"], 2.0)


def test_outcome_mappings_preserve_ordinal_direction() -> None:
    assert map_five_level("HOME", "mimic4") == 0
    assert map_five_level("HOME HEALTH CARE", "mimic3") == 1
    assert map_five_level("REHAB/DISTINCT PART HOSP", "mimic3") == 2
    assert map_five_level("SKILLED NURSING FACILITY", "mimic4") == 3
    assert map_five_level("SNF", "mimic3") == 3
    assert map_five_level("DIED", "mimic4") == 4
    assert map_six_level("HOSPICE", "mimic4") == 4
    assert map_six_level("DIED", "mimic4") == 5
    assert map_five_level("NOT MAPPED", "mimic4") != map_five_level(
        "NOT MAPPED", "mimic4"
    )
    assert five_to_four(0) == 0
    assert five_to_four(2) == 1
    assert five_to_four(3) == 2
    assert five_to_four(4) == 3
    assert [five_to_three(value) for value in range(5)] == [0, 1, 1, 1, 2]
    assert [four_to_three(value) for value in range(4)] == [0, 1, 1, 2]


def test_eicu_hospital_death_is_mapped_even_without_a_location_string() -> None:
    frame = pd.DataFrame(
        {
            "discharge_location": ["HOME", ""],
            "hospital_expire_flag": [0, 1],
        }
    )
    mapped = _attach_outcomes(
        frame,
        "eicu",
        "discharge_location",
        "hospital_expire_flag",
    )
    assert mapped["outcome"].tolist() == [0.0, 3.0]
    assert mapped["death_at_hospital"].tolist() == [0.0, 1.0]


@pytest.mark.parametrize("dataset", ["mimic4", "mimic3"])
def test_mimic_hospital_death_overrides_discharge_location(
    dataset: str,
) -> None:
    hospice_label = "HOSPICE" if dataset == "mimic4" else "HOSPICE-HOME"
    frame = pd.DataFrame(
        {
            "discharge_location": [
                "HOME",
                "HOME HEALTH CARE",
                hospice_label,
                "",
            ],
            "hospital_expire_flag": [1, 0, 0, 1],
        }
    )
    mapped = _attach_outcomes(
        frame,
        dataset,
        "discharge_location",
        "hospital_expire_flag",
    )
    assert mapped.loc[0, "outcome"] == 4.0
    assert mapped.loc[0, "outcome_six"] == 5.0
    assert mapped.loc[0, "outcome4"] == 3.0
    assert mapped.loc[0, "death_at_hospital"] == 1.0
    assert mapped.loc[1, "outcome"] == 1.0
    assert mapped.loc[2, "outcome"] == 4.0
    assert mapped.loc[2, "outcome_six"] == 4.0
    assert pd.isna(mapped.loc[3, "outcome"])
    assert pd.isna(mapped.loc[3, "outcome_six"])


def test_outcome_fields_cannot_enter_predictors() -> None:
    _assert_feature_contract(["age", "lab_lactate_mean", "gcs_min"])
    for leaked in ("outcome", "outcome4", "disposition_raw", "hospital_expire_flag"):
        with pytest.raises(ValueError, match="outcome fields"):
            _assert_feature_contract(["age", leaked])


def test_point_metrics_accepts_float32_rounding_error_without_warning() -> None:
    labels = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    probabilities = np.full((len(labels), 5), 0.2, dtype=np.float32)
    probabilities += np.eye(5, dtype=np.float32)[labels] * 0.01
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        metrics = point_metrics(labels, probabilities)
    assert np.isfinite(metrics["macro_auc"])


def test_highest_class_target_does_not_shift_when_resample_omits_top_class() -> None:
    labels = np.array([0, 1, 2, 0, 1])
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.20, 0.70],
            [0.60, 0.25, 0.15],
            [0.25, 0.60, 0.15],
        ]
    )
    resampled = np.array([0, 1, 3, 4])

    assert labels[resampled].max() < probabilities.shape[1] - 1
    assert np.isnan(
        _metric_scalar(
            labels[resampled],
            probabilities[resampled],
            "highest_class_auc",
        )
    )


def test_delong_auc_matches_roc_auc_and_identical_scores_have_no_difference() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    scores = np.array(
        [0.05, 0.15, 0.35, 0.60, 0.20, 0.40, 0.55, 0.65, 0.70, 0.75, 0.85, 0.95]
    )
    result = delong_auc_test(labels, scores, scores)
    assert result["auc_a"] == pytest.approx(roc_auc_score(labels, scores))
    assert result["p_value"] == pytest.approx(1.0)
