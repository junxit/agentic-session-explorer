"""Tests for the GitHub update check.

Every test is offline: the network fetch is monkeypatched and the on-disk cache
is redirected into ``tmp_path`` via ``XDG_CACHE_HOME``. No real HTTP request is
made and no real cache file is touched.
"""

from __future__ import annotations

import time

import pytest

from sx import update as up


# --- version parsing / comparison -----------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("0.2.0", (0, 2, 0)),
        ("v1.3.4", (1, 3, 4)),
        ("1.2", (1, 2, 0)),
        ("2", (2, 0, 0)),
        ("v0.2.0-rc1", (0, 2, 0)),
        ("nonsense", None),
        ("", None),
    ],
)
def test_parse_version(text, expected):
    assert up._parse_version(text) == expected


@pytest.mark.parametrize(
    "latest,current,result",
    [
        ("0.2.0", "0.1.0", True),
        ("v0.2.0", "0.1.9", True),
        ("0.1.0", "0.1.0", False),
        ("0.1.0", "0.2.0", False),
        ("garbage", "0.1.0", False),
        ("0.2.0", "garbage", False),
    ],
)
def test_is_newer(latest, current, result):
    assert up.is_newer(latest, current) is result


# --- opt-out ---------------------------------------------------------------

def test_opted_out(monkeypatch):
    monkeypatch.setenv("SX_NO_UPDATE_CHECK", "1")
    assert up.opted_out() is True
    # Even if a newer version exists, the check returns None when opted out.
    monkeypatch.setattr(up, "fetch_latest_version", lambda *a, **k: "99.0.0")
    assert up.check_for_update() is None


def test_not_opted_out_by_default(monkeypatch):
    monkeypatch.delenv("SX_NO_UPDATE_CHECK", raising=False)
    assert up.opted_out() is False


# --- caching behavior ------------------------------------------------------

@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Redirect the update-check cache into a temp dir and clear opt-out."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("SX_NO_UPDATE_CHECK", raising=False)
    return tmp_path


def test_fresh_cache_skips_network(cache_dir, monkeypatch):
    """A fresh cache is used; the network is not touched."""
    up._write_cache("99.0.0")  # checked_at = now

    def _boom(*a, **k):
        raise AssertionError("network must not be called when cache is fresh")

    monkeypatch.setattr(up, "fetch_latest_version", _boom)
    monkeypatch.setattr(up, "current_version", lambda: "0.2.0")

    info = up.check_for_update()
    assert info is not None
    assert info.latest == "99.0.0"
    assert info.current == "0.2.0"
    assert info.command == up.UPGRADE_COMMAND


def test_stale_cache_triggers_fetch(cache_dir, monkeypatch):
    """A stale cache causes a fetch, and the result is cached."""
    up._write_cache("0.0.1")
    # Backdate the cache well beyond the TTL.
    path = up._cache_file()
    import json

    data = json.loads(path.read_text())
    data["checked_at"] = time.time() - 10_000_000
    path.write_text(json.dumps(data))

    monkeypatch.setattr(up, "fetch_latest_version", lambda *a, **k: "99.0.0")
    monkeypatch.setattr(up, "current_version", lambda: "0.2.0")

    info = up.check_for_update()
    assert info is not None and info.latest == "99.0.0"
    # Cache was refreshed with the new value.
    assert json.loads(path.read_text())["latest"] == "99.0.0"


def test_no_update_when_current_is_latest(cache_dir, monkeypatch):
    monkeypatch.setattr(up, "fetch_latest_version", lambda *a, **k: "0.2.0")
    monkeypatch.setattr(up, "current_version", lambda: "0.2.0")
    assert up.check_for_update(force=True) is None


def test_network_failure_returns_none(cache_dir, monkeypatch):
    """No cache + failed fetch must yield None, not raise."""
    monkeypatch.setattr(up, "fetch_latest_version", lambda *a, **k: None)
    monkeypatch.setattr(up, "current_version", lambda: "0.2.0")
    assert up.check_for_update(force=True) is None


def test_offline_falls_back_to_stale_cache(cache_dir, monkeypatch):
    """When the fetch fails, a stale cached value is still used to prompt."""
    up._write_cache("99.0.0")
    path = up._cache_file()
    import json

    data = json.loads(path.read_text())
    data["checked_at"] = time.time() - 10_000_000
    path.write_text(json.dumps(data))

    monkeypatch.setattr(up, "fetch_latest_version", lambda *a, **k: None)
    monkeypatch.setattr(up, "current_version", lambda: "0.2.0")

    info = up.check_for_update()
    assert info is not None and info.latest == "99.0.0"


def test_current_version_returns_something():
    """current_version() resolves to a parseable version, not an error."""
    v = up.current_version()
    assert up._parse_version(v) is not None
