"""Shared pytest helpers for the sx test suite.

Everything here operates on synthetic data under pytest's ``tmp_path`` fixture.
No test ever touches the user's real ``~/.claude``, ``~/.codex`` or
``~/.gemini`` data.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path: Path, records: list[dict]) -> Path:
    """Write a list of dicts as a JSONL file and return the path.

    Args:
        path: Destination file (parent dirs are created if needed).
        records: Objects to serialize, one JSON object per line.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records),
        encoding="utf-8",
    )
    return path
