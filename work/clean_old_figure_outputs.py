"""Remove superseded generated R figure files from the figure output folder."""
from __future__ import annotations

from pathlib import Path


root = Path(r"D:\TT paper\1107Invariant Continuous-Time\results\figures_r")
root = root.resolve()
old = (
    "figure1_technical_route.pdf",
    "figure1_technical_route.png",
    "figure2_study_design.pdf",
    "figure2_study_design.png",
    "figure2_performance.pdf",
    "figure2_performance.png",
    "figure3_model_architecture.pdf",
    "figure3_model_architecture.png",
    "figure3_model_selection.pdf",
    "figure3_model_selection.png",
    "figure3_transport_utility.pdf",
    "figure3_transport_utility.png",
    "figure4_endpoint_ascertainment.pdf",
    "figure4_endpoint_ascertainment.png",
    "figure4_transport_utility.pdf",
    "figure4_transport_utility.png",
    "figure5_calibration_utility.pdf",
    "figure5_calibration_utility.png",
    "figure5_endpoint_ascertainment.pdf",
    "figure5_endpoint_ascertainment.png",
    "figure6_interval_sensitivity.pdf",
    "figure6_interval_sensitivity.png",
    "figure6_attention.pdf",
    "figure6_attention.png",
    "figure7_attention.pdf",
    "figure7_attention.png",
)
for name in old:
    path = (root / name).resolve()
    if root not in path.parents:
        raise RuntimeError(f"Refusing to delete outside figure root: {path}")
    if path.is_file():
        path.unlink()
print("Removed superseded figure outputs.")
