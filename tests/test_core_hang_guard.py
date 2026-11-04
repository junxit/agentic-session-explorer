"""The suite's hang guard: a test that never finishes ends the run instead of stalling it.

A TUI test whose screen never settles awaits it forever, so a broken screen
used to stall the whole suite. ``tests/conftest.py`` arms a per-test timer
that names the test, dumps every thread's stack, and ends the run with status
1. It is shown here on a child run of one hanging test, with a two-second
limit; nothing in it touches a harness store.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HANGING = '''
import asyncio

import pytest


@pytest.mark.asyncio
async def test_awaits_forever():
    await asyncio.Event().wait()
'''


def test_a_test_that_never_finishes_ends_the_run_and_is_named(tmp_path):
    """The child run exits with status 1 within seconds, naming the test that hung."""
    hanging = tmp_path / "test_hangs.py"
    hanging.write_text(HANGING, encoding="utf-8")

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.conftest",
            f"--basetemp={tmp_path / 'child'}",
            str(hanging),
        ],
        check=False,  # the exit status is what the test is about
        cwd=ROOT,
        env={**os.environ, "SX_TEST_TIMEOUT": "2"},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert run.returncode == 1, run.stdout + run.stderr
    assert "test_hangs.py::test_awaits_forever has run for 2 s: taken for a hang" in run.stderr
    assert "Thread" in run.stderr, "every thread's stack is dumped"
