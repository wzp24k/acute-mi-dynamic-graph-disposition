"""Run live Crossref validation and finalize the journal-only registry."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")

subprocess.run(
    [sys.executable, str(ROOT / "work" / "live_validate_reference_dois.py")],
    check=True,
)
subprocess.run(
    [sys.executable, str(ROOT / "work" / "finalize_live_journal_registry.py")],
    check=True,
)
