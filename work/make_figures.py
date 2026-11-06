"""Create submission-grade figures from locked AMI study results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import PROCESSED, RESULTS  # noqa: E402


FIGURE_DIR = RESULTS / "figures"
SOURCE_DIR = FIGURE_DIR / "source_data"
MODEL_LABELS = {
    "xgb_static": "XGB-static",
    "xgb_static_ts": "XGB-static+TS",
    "mlp_static": "MLP-static",
    "mlp_static_ts": "MLP-static+TS",
    "gru": "SeqOrdinal GRU",
    "grud": "GRU-D",
    "transformer": "Transformer",
    "stgcn_lite": "STGCN-lite",
    "dggru_static": "DG-GRU-static",
    "dggru": "DG-GRU",
    "dggru_tr": "DG-GRU+TR",
    "dggru_ctr": "DG-GRU+CTR",
    "admission_severity_benchmark": "Admission-severity benchmark",
}
MODEL_ORDER = [
    "admission_severity_benchmark",
    "xgb_static",
    "xgb_static_ts",
    "mlp_static",
    "mlp_static_ts",
    "gru",
    "grud",
    "transformer",
    "stgcn_lite",
    "dggru_static",
    "dggru",
    "dggru_tr",
    "dggru_ctr",
]
MODEL_COLORS = {
    "admission_severity_benchmark": "#9A9A9A",
    "xgb_static": "#7A8DA6",
    "xgb_static_ts": "#3E6F9E",
    "mlp_static": "#A9B8C8",
    "mlp_static_ts": "#5A87AE",
    "gru": "#C28E4A",
    "grud": "#A86F2B",
    "transformer": "#7C9E63",
    "stgcn_lite": "#4D7C4A",
    "dggru_static": "#D7A2A6",
    "dggru": "#B05B62",
    "dggru_tr": "#8C2F39",
    "dggru_ctr": "#6E4A8A",
}
MODEL_MARKERS = {
    "admission_severity_benchmark": "P",
    "xgb_static": "s",
    "xgb_static_ts": "s",
    "mlp_static": "v",
    "mlp_static_ts": "v",
    "gru": "o",
    "grud": "o",
    "transformer": "^",
    "stgcn_lite": "^",
    "dggru_static": "D",
    "dggru": "D",
    "dggru_tr": "D",
    "dggru_ctr": "D",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _save(figure: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "tiff", "png"):
        figure.savefig(
            FIGURE_DIR / f"{stem}.{extension}",
            dpi=600 if extension == "tiff" else (300 if extension == "png" else None),
        )
    plt.close(figure)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def figure1() -> None:
    figure = plt.figure(figsize=(7.09, 4.15))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1.0, 0.95],
        wspace=0.55,
        hspace=0.6,
    )
    flow_axis = figure.add_subplot(grid[0, 0])
    outcome_axis = figure.add_subplot(grid[0, 1])
    model_axis = figure.add_subplot(grid[1, :])

    flow = pd.read_csv(PROCESSED / "mimic4" / "cohort_flow.csv")
    labels = [
        "AMI admissions",
        "First ICU stay",
        "In ICU at 24 h",
        "Mapped outcome",
        "Analytic cohort",
    ]
    flow = flow.iloc[[0, 1, 3, 4, 5]].reset_index(drop=True)
    counts = flow["n"].to_numpy(dtype=int)
    y = np.arange(len(counts))[::-1]
    flow_axis.barh(y, counts, color="#4E6F8E", height=0.62)
    flow_axis.set_yticks(y, labels)
    flow_axis.set_xlabel("Patients or admissions (n)")
    flow_axis.set_xlim(0, counts.max() * 1.16)
    for ypos, value in zip(y, counts):
        flow_axis.text(
            value + counts.max() * 0.018,
            ypos,
            f"{value:,}",
            va="center",
            fontsize=6.5,
        )
    flow_axis.set_title("MIMIC-IV cohort flow", loc="left", fontweight="bold")
    _panel_label(flow_axis, "a")
    flow.assign(label=labels).to_csv(
        SOURCE_DIR / "figure1a_cohort_flow.csv", index=False
    )

    outcome_rows = []
    for dataset, directory in (
        ("MIMIC-IV", "mimic4"),
        ("MIMIC-III", "mimic3"),
        ("eICU", "eicu"),
    ):
        cohort = pd.read_csv(PROCESSED / directory / "cohort.csv")
        counts = cohort["outcome4"].value_counts().sort_index()
        total = counts.sum()
        for level in range(4):
            outcome_rows.append(
                {
                    "dataset": dataset,
                    "class": level,
                    "n": int(counts.get(level, 0)),
                    "fraction": float(counts.get(level, 0) / total),
                }
            )
    outcome_frame = pd.DataFrame(outcome_rows)
    class_colors = ["#D8E2EC", "#94B2CA", "#557C9B", "#9D3F47"]
    names = list(dict.fromkeys(outcome_frame["dataset"]))
    for row_index, name in enumerate(names[::-1]):
        subset = outcome_frame[outcome_frame["dataset"].eq(name)]
        left = 0.0
        for level in range(4):
            value = float(
                subset.loc[subset["class"].eq(level), "fraction"].iloc[0]
            )
            outcome_axis.barh(
                row_index,
                value,
                left=left,
                color=class_colors[level],
                height=0.58,
                edgecolor="white",
                linewidth=0.5,
            )
            if value >= 0.09:
                outcome_axis.text(
                    left + value / 2,
                    row_index,
                    f"{100 * value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
            left += value
    outcome_axis.set_yticks(
        np.arange(len(names)), names[::-1]
    )
    outcome_axis.set_xlim(0, 1)
    outcome_axis.set_xlabel("Proportion")
    outcome_axis.set_title("Harmonized disposition", loc="left", fontweight="bold")
    _panel_label(outcome_axis, "b")
    outcome_axis.legend(
        ["Home", "Rehab/home health", "Facility", "Death/hospice"],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
    )
    outcome_frame.to_csv(
        SOURCE_DIR / "figure1b_outcome_distribution.csv", index=False
    )

    model_axis.set_xlim(0, 1)
    model_axis.set_ylim(0, 1)
    model_axis.axis("off")
    boxes = [
        (0.015, 0.57, 0.15, 0.26, "24 h ICU\n8 channels"),
        (0.205, 0.57, 0.16, 0.26, "Observed\nmask"),
        (0.405, 0.57, 0.18, 0.26, "Dynamic\nchannel graph"),
        (0.625, 0.57, 0.14, 0.26, "GRU\n64 units"),
        (0.805, 0.57, 0.16, 0.26, "Ordinal\nhead"),
        (0.205, 0.12, 0.16, 0.26, "Static\nfeatures"),
        (0.625, 0.12, 0.14, 0.26, "Trajectory\nscore"),
    ]
    box_colors = ["#E6EDF3", "#E6EDF3", "#DCE8DD", "#E8E0D8", "#DDE3EA", "#EEE9DF", "#F0DFDF"]
    for (x, ypos, width, height, text), color in zip(boxes, box_colors):
        patch = FancyBboxPatch(
            (x, ypos),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=0.6,
            edgecolor="#4B4B4B",
            facecolor=color,
        )
        model_axis.add_patch(patch)
        model_axis.text(
            x + width / 2,
            ypos + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=6.5,
        )
    arrows = [
        ((0.165, 0.70), (0.205, 0.70)),
        ((0.365, 0.70), (0.405, 0.70)),
        ((0.585, 0.70), (0.625, 0.70)),
        ((0.765, 0.70), (0.805, 0.70)),
        ((0.285, 0.38), (0.365, 0.58)),
        ((0.695, 0.58), (0.695, 0.38)),
        ((0.695, 0.38), (0.805, 0.62)),
    ]
    for start, end in arrows:
        model_axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.65,
                color="#4B4B4B",
            )
        )
    model_axis.text(
        0.50,
        0.465,
        r"$L=L_{ord}+\lambda_c\,\overline{\|\Delta^2 q_t\|^2},"
        r"\quad q_t=P(Y\geq k\mid x_{1:t})$",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#8C2F39",
    )
    _panel_label(model_axis, "c")
    _save(figure, "figure1_study_design")


def figure2() -> None:
    payload = _read_json(RESULTS / "metrics_mimic4.json")
    benchmark = _read_json(RESULTS / "metrics_mimic4_severity_benchmark.json")
    if payload is None:
        return
    rows = []
    for model, metrics in payload["metrics"].items():
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "macro_auc": metrics["macro_auc"],
                "ci_low": metrics["macro_auc_ci"]["ci_low"],
                "ci_high": metrics["macro_auc_ci"]["ci_high"],
            }
        )
    if benchmark is not None:
        metrics = benchmark["metrics"]
        rows.append(
            {
                "model": "admission_severity_benchmark",
                "label": MODEL_LABELS["admission_severity_benchmark"],
                "macro_auc": metrics["macro_auc"],
                "ci_low": metrics["macro_auc_ci"]["ci_low"],
                "ci_high": metrics["macro_auc_ci"]["ci_high"],
            }
        )
    frame = pd.DataFrame(rows)
    order = [model for model in MODEL_ORDER if model in set(frame["model"])]
    frame["_order"] = frame["model"].map(
        {model: index for index, model in enumerate(order)}
    )
    frame = frame.sort_values("_order")
    figure = plt.figure(figsize=(7.1, 7.9))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 0.9],
        height_ratios=[1.05, 0.9],
        wspace=0.58,
        hspace=0.62,
    )
    axis = figure.add_subplot(grid[0, 0])
    y = np.arange(len(frame))[::-1]
    for ypos, (_, row) in zip(y, frame.iterrows()):
        color = MODEL_COLORS.get(row["model"], "#555555")
        marker = MODEL_MARKERS.get(row["model"], "o")
        axis.plot(
            [row["ci_low"], row["ci_high"]],
            [ypos, ypos],
            color=color,
            linewidth=1.1,
        )
        axis.scatter(
            row["macro_auc"],
            ypos,
            s=26,
            marker=marker,
            color=color,
            zorder=3,
        )
        axis.text(
            row["ci_high"] + 0.004,
            ypos,
            f'{row["macro_auc"]:.3f}',
            va="center",
            fontsize=6.2,
        )
    axis.set_yticks(y, frame["label"])
    axis.set_xlabel("Five-class macro-AUC (95% patient-bootstrap CI)")
    axis.set_xlim(
        max(0.5, frame["ci_low"].min() - 0.03),
        min(1.0, frame["ci_high"].max() + 0.06),
    )
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.45)
    axis.set_axisbelow(True)
    axis.set_title("24 h model comparison", loc="left", fontweight="bold")
    _panel_label(axis, "a")

    class_axis = figure.add_subplot(grid[0, 1])
    model_rows = [
        model
        for model in MODEL_ORDER
        if model != "admission_severity_benchmark"
        and model in payload["metrics"]
    ]
    class_matrix = np.asarray(
        [
            [
                payload["metrics"][model]["per_class_auc"][str(class_index)]
                for class_index in range(5)
            ]
            for model in model_rows
        ],
        dtype=float,
    )
    image = class_axis.imshow(
        class_matrix,
        aspect="auto",
        cmap="YlGnBu",
        vmin=np.nanmin(class_matrix),
        vmax=np.nanmax(class_matrix),
    )
    class_axis.set_xticks(
        np.arange(5),
        ["Home/\nAA", "Home\nhealth", "Rehab/\nAL", "Facility", "Hospice/\ndeath"],
    )
    class_axis.set_yticks(
        np.arange(len(model_rows)),
        [MODEL_LABELS[model] for model in model_rows],
    )
    for row_index in range(class_matrix.shape[0]):
        for column_index in range(class_matrix.shape[1]):
            value = class_matrix[row_index, column_index]
            if np.isfinite(value):
                class_axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color="white" if value > np.nanmedian(class_matrix) else "#222222",
                )
    class_axis.set_title("Per-class AUC", loc="left", fontweight="bold")
    class_axis.tick_params(axis="x", length=0)
    class_axis.tick_params(axis="y", length=0)
    colorbar = figure.colorbar(image, ax=class_axis, fraction=0.045, pad=0.03)
    colorbar.set_label("One-vs-rest AUC")
    _panel_label(class_axis, "b")

    landmark_axis = figure.add_subplot(grid[1, :])
    landmark_models = ("xgb_static", "xgb_static_ts", "dggru", "dggru_tr")
    landmark_sources = {
        6: RESULTS / "metrics_mimic4_6h.json",
        24: RESULTS / "metrics_mimic4.json",
        72: RESULTS / "metrics_mimic4_72h.json",
    }
    landmark_rows = []
    for landmark, path in landmark_sources.items():
        landmark_payload = _read_json(path)
        if landmark_payload is None:
            continue
        for model in landmark_models:
            if model not in landmark_payload.get("metrics", {}):
                continue
            metric = landmark_payload["metrics"][model]
            landmark_rows.append(
                {
                    "landmark_hours": landmark,
                    "model": model,
                    "macro_auc": metric["macro_auc"],
                    "ci_low": metric["macro_auc_ci"]["ci_low"],
                    "ci_high": metric["macro_auc_ci"]["ci_high"],
                }
            )
    landmark_frame = pd.DataFrame(landmark_rows)
    if not landmark_frame.empty:
        x_offsets = {
            "xgb_static": -0.22,
            "xgb_static_ts": -0.07,
            "dggru": 0.07,
            "dggru_tr": 0.22,
        }
        for model in landmark_models:
            subset = landmark_frame.loc[
                landmark_frame["model"].eq(model)
            ].sort_values("landmark_hours")
            if subset.empty:
                continue
            x = subset["landmark_hours"].astype(float) + x_offsets[model]
            landmark_axis.errorbar(
                x,
                subset["macro_auc"],
                yerr=np.vstack(
                    [
                        subset["macro_auc"] - subset["ci_low"],
                        subset["ci_high"] - subset["macro_auc"],
                    ]
                ),
                fmt=MODEL_MARKERS[model],
                markersize=3.8,
                linewidth=0.9,
                capsize=2,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        landmark_axis.set_xticks([6, 24, 72])
        landmark_axis.set_xlabel("ICU landmark (hours)")
        landmark_axis.set_ylabel("Five-class macro-AUC")
        landmark_axis.grid(axis="y", color="#E5E5E5", linewidth=0.45)
        landmark_axis.set_axisbelow(True)
        landmark_axis.legend(
            frameon=False,
            ncol=4,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            handlelength=1.2,
            columnspacing=1.0,
        )
        landmark_axis.set_title(
            "Landmark comparison", loc="left", fontweight="bold", pad=17
        )
        _panel_label(landmark_axis, "c")
    else:
        landmark_axis.axis("off")

    _save(figure, "figure2_model_discrimination")
    frame.drop(columns="_order").to_csv(
        SOURCE_DIR / "figure2_model_discrimination.csv", index=False
    )
    pd.DataFrame(class_matrix, index=model_rows).to_csv(
        SOURCE_DIR / "figure2_per_class_auc.csv"
    )
    landmark_frame.to_csv(
        SOURCE_DIR / "figure2_landmark_comparison.csv", index=False
    )


def figure3() -> None:
    summary_path = (
        RESULTS
        / "sensitivity"
        / "temporal_controls"
        / "summary_seed42.json"
    )
    if not summary_path.is_file():
        return
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    payload = _read_json(RESULTS / "metrics_mimic4.json")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.1, 6.0),
        gridspec_kw={
            "width_ratios": [1.12, 1.0],
            "wspace": 0.76,
            "hspace": 0.78,
        },
    )
    figure.subplots_adjust(left=0.23, right=0.98, top=0.92, bottom=0.13)

    factorial_specs = (
        (
            "same_static_input_mlp_vs_xgb",
            "Static contract: MLP - XGB",
            "Family-specific pipeline",
            "#3E6F9E",
            "o",
        ),
        (
            "same_full_input_xgb_vs_mlp",
            "Static+TS contract: XGB - MLP",
            "Family-specific pipeline",
            "#3E6F9E",
            "o",
        ),
        (
            "same_full_input_gru_vs_dggru",
            "Same input: SeqOrdinal GRU - DG-GRU",
            "Strict architecture",
            "#2F6B5E",
            "^",
        ),
        (
            "time_series_increment",
            "XGB: static -> +TS",
            "Input",
            "#A86F2B",
            "s",
        ),
        (
            "time_series_increment_mlp",
            "MLP: static -> +TS",
            "Input",
            "#A86F2B",
            "s",
        ),
        (
            "static_input_dggru_increment",
            "DG-GRU: sequence+availability off -> on",
            "Input",
            "#A86F2B",
            "s",
        ),
        (
            "regularization_increment",
            r"DG-GRU: +scalar TR ($\lambda_s$=0.2)",
            "Regularizer",
            "#8C2F39",
            "D",
        ),
        (
            "cumulative_curvature_increment",
            r"DG-GRU: +curvature TR ($\lambda_c$=0.01)",
            "Regularizer",
            "#6E4A8A",
            "D",
        ),
    )
    factorial_rows = []
    for key, label, family, color, marker in factorial_specs:
        if payload is None or key not in payload:
            continue
        interval = payload[key]["delta_macro_auc"]
        factorial_rows.append(
            {
                "contrast_key": key,
                "contrast": label,
                "family": family,
                "estimate": interval["delta"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "color": color,
                "marker": marker,
            }
        )
    if factorial_rows:
        factorial_frame = pd.DataFrame(factorial_rows)
        y = np.arange(len(factorial_frame))[::-1]
        x_low = float(factorial_frame["ci_low"].min())
        x_high = float(factorial_frame["ci_high"].max())
        span = max(x_high - x_low, 0.001)
        for ypos, (_, row) in zip(y, factorial_frame.iterrows()):
            axes[0, 0].plot(
                [row["ci_low"], row["ci_high"]],
                [ypos, ypos],
                color=row["color"],
                linewidth=1.0,
            )
            axes[0, 0].scatter(
                row["estimate"],
                ypos,
                color=row["color"],
                marker=row["marker"],
                s=23,
                zorder=3,
            )
            text_x = float(row["ci_high"]) + 0.015 * span
            axes[0, 0].text(
                text_x,
                ypos,
                f"{float(row['estimate']):+.3f}",
                ha="left",
                va="center",
                fontsize=5.2,
            )
        axes[0, 0].axvline(0.0, color="#777777", linewidth=0.7)
        axes[0, 0].set_yticks(y, factorial_frame["contrast"])
        axes[0, 0].set_ylim(-1.8, len(factorial_frame) - 0.35)
        axes[0, 0].set_xlim(x_low - 0.03 * span, x_high + 0.14 * span)
        axes[0, 0].set_xlabel("Delta five-class macro-AUC (95% CI)")
        axes[0, 0].set_title(
            "Factorial contrasts",
            loc="left",
            fontweight="bold",
            pad=20,
        )
        axes[0, 0].tick_params(axis="y", labelsize=6.2)
        axes[0, 0].grid(axis="x", color="#E5E5E5", linewidth=0.45)
        axes[0, 0].set_axisbelow(True)
        family_handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                color="#3E6F9E",
                linestyle="None",
                markersize=3,
                label="Family-specific pipeline",
            ),
            plt.Line2D(
                [],
                [],
                marker="^",
                color="#2F6B5E",
                linestyle="None",
                markersize=3,
                label="Strict architecture",
            ),
            plt.Line2D(
                [],
                [],
                marker="s",
                color="#A86F2B",
                linestyle="None",
                markersize=3,
                label="Input",
            ),
            plt.Line2D(
                [],
                [],
                marker="D",
                color="#8C2F39",
                linestyle="None",
                markersize=3,
                label="Regularizer",
            ),
        ]
        axes[0, 0].legend(
            handles=family_handles,
            frameon=False,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            fontsize=5.2,
            handletextpad=0.3,
            columnspacing=0.7,
        )
        factorial_frame.drop(columns=["color", "marker"]).to_csv(
            SOURCE_DIR / "figure3a_factorial_contrasts.csv",
            index=False,
        )
    else:
        axes[0, 0].axis("off")
    _panel_label(axes[0, 0], "a")

    anchor_path = (
        RESULTS / "tables" / "table4b_multiseed_temporal_anchors.csv"
    )
    if anchor_path.is_file():
        anchors = pd.read_csv(anchor_path)
        reference = anchors.loc[anchors["configuration"].eq("No TR")]
        if len(reference) != 1:
            raise ValueError("Expected exactly one five-seed No TR anchor")
        reference_auc = float(reference["macro_auc"].iloc[0])
        paired_columns = {
            "delta_ci_low",
            "delta_ci_high",
        }
        if not paired_columns.issubset(anchors.columns):
            raise ValueError(
                "Five-seed anchor table lacks paired-bootstrap delta intervals; "
                "rerun work/build_multiseed_temporal_summary.py"
            )
        anchors["delta_macro_auc"] = anchors["macro_auc"] - reference_auc
        y = np.arange(len(anchors))[::-1]
        for ypos, (_, row) in zip(y, anchors.iterrows()):
            axes[0, 1].plot(
                [row["delta_ci_low"], row["delta_ci_high"]],
                [ypos, ypos],
                color="#8C2F39",
                linewidth=1.1,
            )
            axes[0, 1].scatter(
                row["delta_macro_auc"],
                ypos,
                color="#8C2F39",
                s=24,
                zorder=3,
            )
            axes[0, 1].text(
                row["delta_ci_high"] + 0.00008,
                ypos,
                f"{row['delta_macro_auc']:+.5f}",
                va="center",
                fontsize=5.6,
            )
        axes[0, 1].set_yticks(y, anchors["configuration"])
        axes[0, 1].axvline(0.0, color="#6B6B6B", linestyle=":", linewidth=0.75)
        low = float(anchors["delta_ci_low"].min()) - 0.0004
        high = float(anchors["delta_ci_high"].max()) + 0.0008
        axes[0, 1].set_xlim(low, high)
        axes[0, 1].set_xlabel(
            "Paired macro-AUC change vs No TR (95% CI)"
        )
        axes[0, 1].set_title(
            "Scalar TR anchors vs No TR",
            loc="left",
            fontweight="bold",
        )
        axes[0, 1].grid(axis="x", color="#E5E5E5", linewidth=0.45)
        axes[0, 1].set_axisbelow(True)
        anchors.to_csv(
            SOURCE_DIR / "figure3b_multiseed_scalar_anchors.csv",
            index=False,
        )
    else:
        axes[0, 1].axis("off")
    _panel_label(axes[0, 1], "b")

    curvature_path = (
        RESULTS
        / "sensitivity"
        / "cumulative_curvature"
        / "summary_seed42.json"
    )
    curvature_payload = _read_json(curvature_path)
    no_tr_row = frame.loc[frame["name"].eq("no_tr")].iloc[0]
    no_tr_auc = float(no_tr_row["metrics"]["macro_auc"])
    if curvature_payload:
        curvature = pd.DataFrame(
            [
                {
                    "lambda_curvature": float(record["lambda_curvature"]),
                    "macro_auc": float(record["metrics"]["macro_auc"]),
                }
                for record in curvature_payload
            ]
        ).sort_values("lambda_curvature")
        x = np.arange(len(curvature))
        delta_path = curvature["macro_auc"] - no_tr_auc
        axes[1, 0].plot(
            x,
            delta_path,
            marker="o",
            markersize=3.4,
            color=MODEL_COLORS["dggru_ctr"],
            linewidth=1.1,
        )
        axes[1, 0].axhline(
            0.0,
            color="#666666",
            linestyle=":",
            linewidth=0.8,
            label="Matched no TR",
        )
        axes[1, 0].set_xticks(
            x,
            [
                "0"
                if value == 0
                else f"{float(value):g}"
                for value in curvature["lambda_curvature"]
            ],
            rotation=35,
            ha="right",
        )
        axes[1, 0].set_xlabel(r"Cumulative-curvature weight $\lambda_c$")
        axes[1, 0].set_ylabel("Macro-AUC change vs No TR")
        path_margin = max(
            0.0004,
            0.15
            * max(
                abs(float(delta_path.min())),
                abs(float(delta_path.max())),
            ),
        )
        axes[1, 0].set_ylim(
            min(0.0, float(delta_path.min())) - path_margin,
            max(0.0, float(delta_path.max())) + path_margin,
        )
        axes[1, 0].legend(
            frameon=False,
            loc="best",
            fontsize=5.6,
        )
        axes[1, 0].grid(axis="y", color="#E5E5E5", linewidth=0.45)
        axes[1, 0].set_axisbelow(True)
        axes[1, 0].set_title(
            "Curvature TR path (seed 42)",
            loc="left",
            fontweight="bold",
        )
        curvature["delta_macro_auc_vs_no_tr"] = delta_path.to_numpy()
        curvature.to_csv(
            SOURCE_DIR / "figure3c_cumulative_curvature_path.csv",
            index=False,
        )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Cumulative-curvature path not available",
            transform=axes[1, 0].transAxes,
            ha="center",
            va="center",
            fontsize=6,
        )
        axes[1, 0].set_axis_off()
    _panel_label(axes[1, 0], "c")

    control_order = [
        "no_tr",
        "intact",
        "monotone_decrease",
        "monotone_increase",
        "permuted",
        "smooth_only",
    ]
    control_labels = {
        "no_tr": "No TR",
        "intact": "Intact",
        "monotone_decrease": "Monotone down",
        "monotone_increase": "Monotone up",
        "permuted": "Permuted",
        "smooth_only": "Smooth only",
    }
    control_frame = frame.loc[frame["name"].isin(control_order)].copy()
    control_frame["_order"] = control_frame["name"].map(
        {name: index for index, name in enumerate(control_order)}
    )
    control_frame = control_frame.sort_values("_order")
    y = np.arange(len(control_frame))[::-1]
    for ypos, (_, row) in zip(y, control_frame.iterrows()):
        value = float(row["metrics"]["macro_auc"])
        delta = value - no_tr_auc
        axes[1, 1].plot(
            [0.0, delta],
            [ypos, ypos],
            color="#A9B8C8",
            linewidth=0.8,
        )
        axes[1, 1].scatter(delta, ypos, color="#3E6F9E", s=22, zorder=3)
        axes[1, 1].text(
            delta + 0.00007,
            ypos,
            f"{delta:+.5f}",
            va="center",
            fontsize=5.5,
        )
    axes[1, 1].set_yticks(
        y,
        [control_labels.get(name, name) for name in control_frame["name"]],
    )
    axes[1, 1].axvline(
        0.0,
        color="#6B6B6B",
        linestyle=":",
        linewidth=0.75,
    )
    control_deltas = control_frame["metrics"].map(
        lambda value: float(value["macro_auc"]) - no_tr_auc
    )
    control_low = min(0.0, float(control_deltas.min())) - 0.00025
    control_high = max(0.0, float(control_deltas.max())) + 0.00035
    axes[1, 1].set_xlim(control_low, control_high)
    axes[1, 1].set_xlabel("Macro-AUC change vs No TR")
    axes[1, 1].grid(axis="x", color="#E5E5E5", linewidth=0.45)
    axes[1, 1].set_axisbelow(True)
    axes[1, 1].set_title(
        "Matched controls (seed 42)",
        loc="left",
        fontweight="bold",
    )
    control_frame["delta_macro_auc_vs_no_tr"] = control_deltas.to_numpy()
    control_frame.drop(columns="_order").to_csv(
        SOURCE_DIR / "figure3d_regularization_controls.csv",
        index=False,
    )
    _panel_label(axes[1, 1], "d")
    figure.text(
        0.01,
        0.005,
        "Panels b-d show signed changes from the matched No TR reference; "
        "panel b intervals are paired patient-bootstrap intervals. "
        "Panels a-b use five-seed averaged out-of-fold probabilities; "
        "panels c-d are seed-42 single-run experiments and are exploratory.",
        fontsize=5.4,
        color="#555555",
        ha="left",
        va="bottom",
    )
    _save(figure, "figure3_temporal_regularization")
    frame.to_csv(SOURCE_DIR / "figure3_temporal_controls.csv", index=False)


def figure4() -> None:
    predictions = {}
    for model in ("xgb_static_ts", "dggru_tr"):
        paths = sorted(
            (RESULTS / "predictions").glob(f"mimic4_{model}_seed*.npz")
        )
        if len(paths) != 5:
            continue
        arrays = []
        for path in paths:
            with np.load(path, allow_pickle=True) as data:
                arrays.append(data["probabilities"].astype(float))
                labels = data["labels"].astype(int)
        predictions[model] = np.mean(np.stack(arrays), axis=0)
    if len(predictions) < 2:
        return
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    calibration_payload = _read_json(RESULTS / "metrics_mimic4.json")[
        "calibration"
    ]
    for model in predictions:
        values = pd.DataFrame(
            calibration_payload[model]["reliability_y_ge_3"]
        )
        axes[0].plot(
            values["mean_predicted"],
            values["observed"],
            marker="o" if model == "dggru_tr" else "s",
            markersize=3,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axes[0].plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=0.7)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel(r"Predicted $P(y\geq3)$")
    axes[0].set_ylabel("Observed proportion")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title("Calibration", loc="left", fontweight="bold")
    _panel_label(axes[0], "a")
    for model in predictions:
        path = (
            RESULTS
            / "tables"
            / f"mimic4_{model}_dca_y_ge_3.csv"
        )
        if not path.is_file():
            continue
        dca = pd.read_csv(path)
        axes[1].plot(
            dca["threshold"],
            dca["net_benefit"],
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        axes[1].plot(
            dca["threshold"],
            dca["treat_all"],
            color="#999999",
            linestyle="--",
            linewidth=0.7,
            label="Treat all" if model == "xgb_static_ts" else None,
        )
    axes[1].axhline(0, color="#444444", linewidth=0.7)
    axes[1].set_xlabel("Threshold probability")
    axes[1].set_ylabel("Net benefit")
    axes[1].set_title(r"Decision curves for $y\geq3$", loc="left", fontweight="bold")
    _panel_label(axes[1], "b")
    _save(figure, "figure4_calibration_utility")
    frame = pd.DataFrame(
        {
            "model": list(predictions),
            "labels": [labels for _ in predictions],
        }
    )
    frame.to_csv(SOURCE_DIR / "figure4_model_manifest.csv", index=False)


def figure5() -> None:
    payload = _read_json(RESULTS / "metrics_transport.json")
    if payload is None:
        return
    rows = []
    for dataset in ("mimic3", "eicu"):
        for model, result in payload[dataset]["models"].items():
            highest = result["highest_class_auc_ci"]
            strict = result.get("strict_mortality_auc_ci", {})
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "macro_auc": result["macro_auc_ci"]["estimate"],
                    "macro_low": result["macro_auc_ci"]["ci_low"],
                    "macro_high": result["macro_auc_ci"]["ci_high"],
                    "highest_class_auc": highest["estimate"],
                    "highest_class_low": highest["ci_low"],
                    "highest_class_high": highest["ci_high"],
                    "strict_death_auc": strict.get("estimate", np.nan),
                    "strict_death_low": strict.get("ci_low", np.nan),
                    "strict_death_high": strict.get("ci_high", np.nan),
                }
            )
    frame = pd.DataFrame(rows)
    figure = plt.figure(figsize=(7.1, 4.8))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[0.95, 1.1, 0.95],
        height_ratios=[1.0, 0.95],
        wspace=0.72,
        hspace=0.72,
    )

    def transport_axis(axis: plt.Axes, dataset: str, title: str, label: str) -> None:
        positions = []
        labels = []
        for row_index, model in enumerate(("xgb_static", "xgb_static_ts", "dggru_tr")):
            match = frame[
                frame["dataset"].eq(dataset) & frame["model"].eq(model)
            ]
            if match.empty:
                continue
            row = match.iloc[0]
            ypos = 2 - row_index
            axis.plot(
                [row["macro_low"], row["macro_high"]],
                [ypos, ypos],
                color=MODEL_COLORS[model],
                linewidth=1.1,
            )
            axis.scatter(
                row["macro_auc"],
                ypos,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                s=26,
                zorder=3,
            )
            axis.text(
                row["macro_high"] + 0.008,
                ypos,
                f'{row["macro_auc"]:.3f}',
                va="center",
                fontsize=5.8,
            )
            positions.append(ypos)
            labels.append(MODEL_LABELS[model])
        axis.set_yticks(positions, labels)
        axis.set_xlim(0.55, 0.95)
        axis.set_xlabel("Four-class macro-AUC")
        axis.grid(axis="x", color="#E5E5E5", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.set_title(title, loc="left", fontweight="bold")
        _panel_label(axis, label)

    eicu_axis = figure.add_subplot(grid[:, 0])
    transport_axis(eicu_axis, "eicu", "eICU transport", "a")
    mimic3_axis = figure.add_subplot(grid[:, 2])
    transport_axis(
        mimic3_axis,
        "mimic3",
        "MIMIC-III same-institution",
        "c",
    )

    site_axis = figure.add_subplot(grid[0, 1])
    site_metrics = pd.read_csv(
        RESULTS / "tables" / "eicu_site_metrics.csv"
    )
    site_values = (
        site_metrics.loc[
            site_metrics["model"].eq("dggru_tr"), "macro_auc"
        ]
        .dropna()
        .to_numpy(dtype=float)
    )
    if site_values.size:
        counts, _, _ = site_axis.hist(
            site_values,
            bins=np.linspace(0.55, 0.93, 13),
            color="#8C2F39",
            alpha=0.72,
            edgecolor="white",
            linewidth=0.5,
        )
        site_axis.set_ylim(0, max(float(counts.max()) * 1.35, 1.0))
        median = float(np.median(site_values))
        q1, q3 = np.quantile(site_values, [0.25, 0.75])
        site_axis.axvline(median, color="#222222", linewidth=0.9)
        site_axis.axvspan(q1, q3, color="#B05B62", alpha=0.12)
        site_axis.set_xlabel("Hospital macro-AUC")
        site_axis.set_ylabel("Hospitals")
        site_axis.set_title("eICU site heterogeneity", loc="left", fontweight="bold")
        site_axis.text(
            0.02,
            0.96,
            f"computable: {len(site_values)}/68\nmedian {median:.3f}; IQR {q1:.3f}-{q3:.3f}",
            transform=site_axis.transAxes,
            va="top",
            fontsize=5.6,
        )
    _panel_label(site_axis, "b")

    loho_axis = figure.add_subplot(grid[1, 1])
    site_report = _read_json(
        RESULTS / "audits" / "eicu_site_heterogeneity_report.json"
    )
    if site_report:
        summary = site_report["models"]["dggru_tr"]
        full = float(summary["cluster_holdout_full_macro_auc"])
        delta_low = float(summary["cluster_holdout_min_macro_auc"]) - full
        delta_high = float(summary["cluster_holdout_max_macro_auc"]) - full
        loho_axis.plot(
            [delta_low, delta_high],
            [0, 0],
            color="#4E6F8E",
            linewidth=1.4,
        )
        loho_axis.scatter(
            [delta_low, delta_high],
            [0, 0],
            s=16,
            color="#4E6F8E",
            zorder=3,
        )
        loho_axis.axvline(0, color="#666666", linewidth=0.8)
        loho_axis.set_ylim(-0.6, 0.6)
        loho_axis.set_yticks([0], ["147 patient-safe clusters"])
        loho_axis.set_xlabel("Delta macro-AUC after cluster holdout")
        loho_axis.set_title(
            "Leave-one-cluster-out",
            loc="left",
            fontweight="bold",
        )
        loho_axis.text(
            0.02,
            0.92,
            (
                f"range {delta_low:+.4f} to {delta_high:+.4f}\n"
                "negative "
                f"{int(summary['cluster_holdout_negative_delta_count'])}; "
                "positive "
                f"{int(summary['cluster_holdout_positive_delta_count'])}"
            ),
            transform=loho_axis.transAxes,
            va="top",
            fontsize=5.6,
        )
    _save(figure, "figure5_external_validation")
    frame.to_csv(SOURCE_DIR / "figure5_external_validation.csv", index=False)
    pd.DataFrame({"site_macro_auc": site_values}).to_csv(
        SOURCE_DIR / "figure5_eicu_site_macro_auc.csv", index=False
    )


def supplement_subgroups() -> None:
    payload = _read_json(RESULTS / "audits" / "mimic4_subgroup_analysis.json")
    if payload is None:
        return
    rows = []
    for subgroup in payload["subgroups"]:
        contrast = subgroup["dggru_tr_minus_xgb"]
        if (
            subgroup["n"] < 30
            or not np.isfinite(contrast["delta"])
            or not np.isfinite(contrast["ci_low"])
            or not np.isfinite(contrast["ci_high"])
        ):
            continue
        rows.append(
            {
                "subgroup": subgroup["subgroup"],
                "n": subgroup["n"],
                "delta": contrast["delta"],
                "ci_low": contrast["ci_low"],
                "ci_high": contrast["ci_high"],
            }
        )
    frame = pd.DataFrame(rows).iloc[::-1]
    if frame.empty:
        return
    figure, axis = plt.subplots(figsize=(4.2, 4.1))
    y = np.arange(len(frame))
    axis.axvline(0, color="#666666", linewidth=0.7)
    axis.errorbar(
        frame["delta"],
        y,
        xerr=np.vstack(
            [
                frame["delta"] - frame["ci_low"],
                frame["ci_high"] - frame["delta"],
            ]
        ),
        fmt="o",
        markersize=3.3,
        linewidth=0.8,
        color="#8C2F39",
        ecolor="#8C2F39",
        capsize=2,
    )
    axis.set_yticks(y, frame["subgroup"])
    axis.set_xlabel("DG-GRU+TR minus XGB-static+TS (macro-AUC)")
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.45)
    axis.set_axisbelow(True)
    _save(figure, "figureS1_subgroup_contrasts")
    frame.to_csv(SOURCE_DIR / "figureS1_subgroup_contrasts.csv", index=False)


def figure6() -> None:
    subgroup_payload = _read_json(
        RESULTS / "audits" / "mimic4_subgroup_analysis.json"
    )
    sensitivity_path = (
        RESULTS / "tables" / "tableS15_full_16_scenario_sensitivity.csv"
    )
    if subgroup_payload is None or not sensitivity_path.is_file():
        return

    subgroup_rows = []
    for subgroup in subgroup_payload["subgroups"]:
        contrast = subgroup["dggru_tr_minus_xgb"]
        if (
            subgroup["n"] < 30
            or not np.isfinite(contrast["delta"])
            or not np.isfinite(contrast["ci_low"])
            or not np.isfinite(contrast["ci_high"])
        ):
            continue
        subgroup_rows.append(
            {
                "subgroup": subgroup["subgroup"],
                "n": subgroup["n"],
                "delta": contrast["delta"],
                "ci_low": contrast["ci_low"],
                "ci_high": contrast["ci_high"],
            }
        )
    subgroup_frame = pd.DataFrame(subgroup_rows)
    sensitivity = pd.read_csv(sensitivity_path)
    sensitivity_labels = {
        1: "6 h landmark",
        2: "72 h landmark",
        3: "Six-level outcome",
        4: "Primary-position AMI",
        5: "Exclude type 2 MI",
        6: "Exclude against advice",
        7: "Exclude troponin",
        8: "Multiple imputation",
        9: "Pre-24 h revascularization",
        10: "Pre-24 h vasopressor/MCS",
        11: "Reversed monotone",
        12: "Time-permuted trajectory",
        13: "Generic smoothness",
        14: "eICU hospital bootstrap",
        15: "Merged disposition labels",
        16: "eICU Other excluded",
    }
    sensitivity["short_label"] = sensitivity["scenario_id"].map(
        sensitivity_labels
    )

    figure, (subgroup_axis, sensitivity_axis) = plt.subplots(
        1,
        2,
        figsize=(7.1, 6.4),
        gridspec_kw={"width_ratios": [0.9, 1.15], "wspace": 0.72},
    )
    if not subgroup_frame.empty:
        subgroup_frame = subgroup_frame.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(subgroup_frame))
        subgroup_axis.axvline(0, color="#666666", linewidth=0.7)
        subgroup_axis.errorbar(
            subgroup_frame["delta"],
            y,
            xerr=np.vstack(
                [
                    subgroup_frame["delta"] - subgroup_frame["ci_low"],
                    subgroup_frame["ci_high"] - subgroup_frame["delta"],
                ]
            ),
            fmt="o",
            markersize=3.2,
            linewidth=0.8,
            color="#8C2F39",
            ecolor="#8C2F39",
            capsize=2,
        )
        subgroup_axis.set_yticks(y, subgroup_frame["subgroup"])
        subgroup_axis.set_xlabel(
            "DG-GRU+TR minus XGB-static+TS\n(five-class macro-AUC)"
        )
        subgroup_axis.grid(axis="x", color="#E5E5E5", linewidth=0.45)
        subgroup_axis.set_axisbelow(True)
    subgroup_axis.set_title("Prespecified subgroups", loc="left", fontweight="bold")
    _panel_label(subgroup_axis, "a")

    sensitivity_axis.axvline(0, color="#666666", linewidth=0.7)
    sensitivity = sensitivity.sort_values(
        "scenario_id", ascending=False
    ).reset_index(drop=True)
    y = np.arange(len(sensitivity))
    for ypos, (_, row) in zip(y, sensitivity.iterrows()):
        if row["database"] == "eICU-CRD":
            color = "#3E6F9E"
            marker = "s"
        elif int(row["scenario_id"]) in {11, 12, 13}:
            color = "#8C2F39"
            marker = "D"
        else:
            color = "#4D7C4A"
            marker = "o"
        low = (
            float(row["macro_auc_ci_low"] - row["reference_macro_auc"])
            if np.isfinite(row["macro_auc_ci_low"])
            and np.isfinite(row["reference_macro_auc"])
            else np.nan
        )
        high = (
            float(row["macro_auc_ci_high"] - row["reference_macro_auc"])
            if np.isfinite(row["macro_auc_ci_high"])
            and np.isfinite(row["reference_macro_auc"])
            else np.nan
        )
        if np.isfinite(low) and np.isfinite(high):
            sensitivity_axis.plot(
                [low, high], [ypos, ypos], color=color, linewidth=0.9
            )
        sensitivity_axis.scatter(
            row["delta_from_reference"],
            ypos,
            s=22,
            marker=marker,
            color=color,
            zorder=3,
        )
    ylabels = []
    for _, row in sensitivity.iterrows():
        label = f"{int(row.scenario_id)}. {row.short_label}"
        if pd.isna(row["delta_from_reference"]):
            label += " (separate scale)"
        ylabels.append(label)
    sensitivity_axis.set_yticks(y, ylabels)
    sensitivity_axis.set_xlabel("Delta macro-AUC (same scale)")
    sensitivity_axis.grid(axis="x", color="#E5E5E5", linewidth=0.45)
    sensitivity_axis.set_axisbelow(True)
    sensitivity_axis.set_title(
        "Prespecified sensitivity analyses", loc="left", fontweight="bold"
    )
    _panel_label(sensitivity_axis, "b")
    legend_handles = [sensitivity_axis.scatter(
        [],
        [],
        marker="o",
        color="#4D7C4A",
        label="MIMIC-IV, five-level",
    )]
    legend_handles.append(sensitivity_axis.scatter(
        [],
        [],
        marker="D",
        color="#8C2F39",
        label="Temporal control, five-level",
    ))
    legend_handles.append(sensitivity_axis.scatter(
        [],
        [],
        marker="s",
        color="#3E6F9E",
        label="eICU-CRD, four-class",
    ))
    figure.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        frameon=False,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
        handletextpad=0.3,
        borderaxespad=0.1,
    )
    figure.subplots_adjust(bottom=0.14)
    _save(figure, "figure6_subgroups_and_sensitivity")
    subgroup_frame.to_csv(
        SOURCE_DIR / "figure6a_subgroups.csv", index=False
    )
    sensitivity.to_csv(
        SOURCE_DIR / "figure6b_sensitivity.csv", index=False
    )


def figure7() -> None:
    calibration_path = (
        RESULTS / "calibration" / "external_calibration_curves.csv"
    )
    dca_path = (
        RESULTS / "ordinal_evaluation" / "external_dca_curves.csv"
    )
    if not calibration_path.is_file() or not dca_path.is_file():
        return
    calibration = pd.read_csv(calibration_path)
    dca = pd.read_csv(dca_path)
    dataset_order = (
        ("MIMIC-III", "mimic3", "MIMIC-III patient-grouped CV"),
        (
            "eICU direct transport",
            "eicu",
            "eICU patient-safe hospital-cluster CV",
        ),
    )
    model_order = ("xgb_static", "xgb_static_ts", "dggru_tr")
    model_lookup = {
        "xgb_static": "XGB-static",
        "xgb_static_ts": "XGB-static+TS",
        "dggru_tr": "DG-GRU+TR",
    }
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.1, 5.3),
        gridspec_kw={"wspace": 0.46, "hspace": 0.62},
    )
    for column, (dataset, dataset_key, title) in enumerate(dataset_order):
        axis = axes[0, column]
        calibration_subset = calibration.loc[
            calibration["dataset"].eq(dataset)
            & calibration["target"].eq("high_need")
        ]
        for model in model_order:
            model_curves = calibration_subset.loc[
                calibration_subset["model"].eq(model_lookup[model])
            ]
            if model_curves.empty:
                continue
            for score_type, linestyle, suffix in (
                ("raw", "--", "raw"),
                ("grouped_cv_recalibrated", "-", "recalibrated"),
            ):
                curve = model_curves.loc[
                    model_curves["score_type"].eq(score_type)
                ].sort_values("bin_id")
                if curve.empty:
                    continue
                axis.plot(
                    curve["predicted_mean"],
                    curve["observed_rate"],
                    marker="o",
                    markersize=2.4,
                    linewidth=0.9 if score_type == "raw" else 1.15,
                    linestyle=linestyle,
                    color=MODEL_COLORS.get(model, "#555555"),
                    alpha=0.75 if score_type == "raw" else 1.0,
                    label=(
                        f"{model_lookup[model]} {suffix}"
                        if column == 0
                        else None
                    ),
                )
        axis.plot(
            [0, 1],
            [0, 1],
            color="#777777",
            linestyle=":",
            linewidth=0.7,
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel(r"Predicted $P(y\geq2)$")
        axis.set_ylabel("Observed proportion")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(color="#EEEEEE", linewidth=0.4)
        axis.set_axisbelow(True)
        _panel_label(axis, chr(ord("a") + column))

    axes[0, 0].legend(
        frameon=False,
        loc="upper left",
        fontsize=5.2,
        handlelength=1.5,
    )
    axes[0, 0].text(
        0.04,
        0.04,
        "Dashed: raw\nSolid: OOF-recalibrated",
        transform=axes[0, 0].transAxes,
        fontsize=5.2,
        color="#555555",
        va="bottom",
    )

    for column, (_, dataset_key, title) in enumerate(dataset_order):
        axis = axes[1, column]
        subset = dca.loc[
            dca["dataset"].eq(dataset_key)
            & dca["target"].eq("high_need_y_ge_2")
        ]
        for model in model_order:
            curve = subset.loc[subset["model"].eq(model)].sort_values(
                "threshold"
            )
            if curve.empty:
                continue
            color = MODEL_COLORS.get(model, "#555555")
            axis.plot(
                curve["threshold"],
                curve["net_benefit"],
                color=color,
                linewidth=1.15,
                label=model_lookup[model] if column == 0 else None,
            )
        if not subset.empty:
            treat_all = (
                subset.groupby("threshold", as_index=False)[
                    "treat_all_net_benefit"
                ]
                .first()
                .sort_values("threshold")
            )
            axis.plot(
                treat_all["threshold"],
                treat_all["treat_all_net_benefit"],
                color="#777777",
                linestyle="--",
                linewidth=0.8,
                label="Treat all" if column == 0 else None,
            )
            xgb_curve = subset.loc[
                subset["model"].eq("xgb_static_ts")
            ].sort_values("threshold")
            if not xgb_curve.empty:
                axis.fill_between(
                    xgb_curve["threshold"],
                    xgb_curve["net_benefit_ci_low"],
                    xgb_curve["net_benefit_ci_high"],
                    color=MODEL_COLORS["xgb_static_ts"],
                    alpha=0.13,
                    linewidth=0,
                )
        axis.axhline(0, color="#555555", linewidth=0.7)
        axis.set_xlim(0.05, 0.60)
        axis.set_xlabel("Threshold probability")
        axis.set_ylabel("Net benefit")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(color="#EEEEEE", linewidth=0.4)
        axis.set_axisbelow(True)
        _panel_label(axis, chr(ord("c") + column))
    axes[1, 0].legend(
        frameon=False,
        loc="best",
        fontsize=5.4,
        handlelength=1.6,
    )
    figure.text(
        0.01,
        0.004,
        "Calibration curves show raw and grouped out-of-fold recalibrated "
        "transport predictions. Decision curves use raw transported "
        "probabilities. Bands show patient- or patient-safe hospital-cluster "
        "bootstrap intervals.\n"
        "Within-model raw vs OOF-recalibrated comparisons are primary; "
        "all models used the same five-class target before four-class collapse.",
        fontsize=5.0,
        color="#555555",
        ha="left",
        va="bottom",
    )
    _save(figure, "figure7_external_calibration_dca")
    calibration.loc[
        calibration["target"].eq("high_need")
        & calibration["dataset"].isin(
            [dataset for dataset, _, _ in dataset_order]
        )
    ].to_csv(
        SOURCE_DIR / "figure7_external_calibration.csv",
        index=False,
    )
    dca.loc[
        dca["target"].eq("high_need_y_ge_2")
        & dca["dataset"].isin([dataset for _, dataset, _ in dataset_order])
    ].to_csv(
        SOURCE_DIR / "figure7_external_dca.csv",
        index=False,
    )


def main() -> None:
    _style()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    figure1()
    for function in (
        figure2,
        figure3,
        figure4,
        figure5,
        figure6,
        figure7,
        supplement_subgroups,
    ):
        try:
            function()
        except (FileNotFoundError, KeyError, ValueError) as error:
            print(f"Skipped {function.__name__}: {error}")
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
