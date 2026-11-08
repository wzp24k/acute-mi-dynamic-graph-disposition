"""Publish code and non-sensitive aggregate results to the configured public repo."""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
OWNER = "wzp24k"
REPO = "acute-mi-dynamic-graph-disposition"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"


def token() -> str:
    return subprocess.check_output(["gh", "auth", "token"], text=True).strip()


def selected_files() -> list[Path]:
    files = []
    for name in (
        "README.md",
        ".gitignore",
        "CITATION.cff",
        "requirements.txt",
    ):
        path = ROOT / name
        if path.is_file():
            files.append(path)
    for directory in ("configs", "docs", "src", "tests", "work"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                continue
            if path.suffix.lower() in {".py", ".r", ".ps1", ".md", ".json"}:
                files.append(path)
    for directory in ("results/tables", "results/figures_r"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file():
                files.append(path)

    exclude_names = {
        "manual_endpoint_review.csv",
        "manual_endpoint_review_pilot.csv",
        "manual_endpoint_review_structured.csv",
        "eicu_timing_audit.json",
        "mimic_event_timing_audit.json",
    }
    for path in (ROOT / "results").glob("*"):
        if not path.is_file():
            continue
        if path.name in exclude_names:
            continue
        if path.name.startswith("endpoint_audit_"):
            continue
        if path.suffix.lower() in {".json", ".csv", ".md"}:
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def request_json(
    url: str,
    auth_token: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    last_error = None
    for attempt in range(6):
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "CTOGT-publisher/1.0",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"GitHub API {error.code}: {body}")
            if error.code < 500 and error.code != 429:
                raise last_error from error
        except Exception as error:
            last_error = error
        time.sleep(2 + attempt * 2)
    raise RuntimeError(f"GitHub API failed after retries: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    files = selected_files()
    manifest = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(ROOT).as_posix()
        if relative == "1108manuscript.docx" or not relative:
            continue
        manifest.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
            }
        )
    forbidden = [
        item for item in manifest
        if item["path"].startswith(("outputs/", "data/", "results/predictions/"))
        or item["path"].endswith(".docx")
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden repository paths: {forbidden}")
    print(
        json.dumps(
            {
                "file_count": len(manifest),
                "total_bytes": sum(item["bytes"] for item in manifest),
                "files": manifest,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    auth_token = token()
    ref = request_json(f"{API}/git/ref/heads/main", auth_token)
    parent_sha = ref["object"]["sha"]
    tree_entries = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(ROOT).as_posix()
        content = base64.b64encode(path.read_bytes()).decode("ascii")
        blob = request_json(
            f"{API}/git/blobs",
            auth_token,
            method="POST",
            payload={"content": content, "encoding": "base64"},
        )
        tree_entries.append(
            {
                "path": relative,
                "mode": "100644",
                "type": "blob",
                "sha": blob["sha"],
            }
        )
        print(f"uploaded {index}/{len(files)} {relative}", flush=True)
    tree = request_json(
        f"{API}/git/trees",
        auth_token,
        method="POST",
        payload={"tree": tree_entries},
    )
    commit = request_json(
        f"{API}/git/commits",
        auth_token,
        method="POST",
        payload={
            "message": (
                "Update repository for diagnosis-coded MACCE temporal ensemble study"
            ),
            "tree": tree["sha"],
            "parents": [parent_sha],
        },
    )
    request_json(
        f"{API}/git/refs/heads/main",
        auth_token,
        method="PATCH",
        payload={"sha": commit["sha"], "force": False},
    )
    request_json(
        API,
        auth_token,
        method="PATCH",
        payload={
            "description": (
                "Code and non-sensitive aggregate results for continuous-time "
                "organ graph learning and temporal ensembling in "
                "diagnosis-coded MACCE"
            ),
            "homepage": "https://github.com/wzp24k/acute-mi-dynamic-graph-disposition",
        },
    )
    output = {
        "repository": f"https://github.com/{OWNER}/{REPO}",
        "commit": commit["sha"],
        "parent": parent_sha,
        "file_count": len(tree_entries),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "manuscript_uploaded": False,
    }
    (ROOT / "results" / "github_publish_manifest.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
