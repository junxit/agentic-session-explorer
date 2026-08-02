"""Lightweight "is there a newer release?" check against GitHub.

Design goals (a CLI that phones home on every launch is an anti-pattern unless
it is careful):

* **Stdlib only** — uses ``urllib``; adds no runtime dependency.
* **Throttled** — the network is hit at most once per
  :data:`DEFAULT_TTL_HOURS` (default 24h); results are cached on disk.
* **Fail-silent** — any network/parse error returns ``None``; the check must
  never break or noticeably slow down ``sx``.
* **Opt-out** — honours ``SX_NO_UPDATE_CHECK`` (and ``--no-update-check``).
* **Version signal** — the latest GitHub *release* tag, falling back to the
  highest ``vX.Y.Z`` git tag. Publishing an update therefore means cutting a
  release/tag, not just pushing to ``main``.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

#: Canonical GitHub repository ``owner/name``.
REPO = "junxit/agentic-session-explorer"
#: HTTPS clone URL (used in the upgrade hint and uvx instructions).
GIT_URL = f"https://github.com/{REPO}.git"
#: Command shown to the user to upgrade a tool-installed copy.
UPGRADE_COMMAND = "uv tool upgrade sx"

#: How long a cached "latest version" result stays fresh.
DEFAULT_TTL_HOURS = 24.0
#: Network timeout for the GitHub API call, in seconds.
DEFAULT_TIMEOUT = 1.5

_VERSION_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")


@dataclass(frozen=True)
class UpdateInfo:
    """Describes an available upgrade.

    Attributes:
        current: The installed version string.
        latest: The newest published version string.
        command: The recommended upgrade command.
    """

    current: str
    latest: str
    command: str = UPGRADE_COMMAND


def _truthy(value: str | None) -> bool:
    """Return True if an environment-style string is a truthy value."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def opted_out() -> bool:
    """Return True if the update check is disabled via ``SX_NO_UPDATE_CHECK``."""
    return _truthy(os.environ.get("SX_NO_UPDATE_CHECK"))


def current_version() -> str:
    """Return the installed ``sx`` version.

    Prefers installed distribution metadata; falls back to the in-tree
    ``__version__`` when metadata is unavailable (e.g. running from source).
    """
    try:
        from importlib.metadata import version

        return version("sx")
    except Exception:  # noqa: BLE001 - metadata may be missing when run from source
        from sx import __version__

        return __version__


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """Parse a ``MAJOR.MINOR.PATCH`` prefix from a version/tag string.

    Tolerates a leading ``v`` and any pre-release/build suffix. Missing minor or
    patch components default to 0.

    Args:
        text: A version or tag string (e.g. ``"v0.2.0"``, ``"1.3"``).

    Returns:
        A ``(major, minor, patch)`` tuple, or ``None`` if no number is found.
    """
    if not text:
        return None
    match = _VERSION_RE.search(text)
    if not match:
        return None
    major, minor, patch = match.groups()
    return (int(major), int(minor or 0), int(patch or 0))


def is_newer(latest: str, current: str) -> bool:
    """Return True if ``latest`` is a strictly higher version than ``current``."""
    lp, cp = _parse_version(latest), _parse_version(current)
    if lp is None or cp is None:
        return False
    return lp > cp


#: Upper bound on an update-check response body (GitHub's is a few KB).
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _cache_file() -> Path:
    """Return the on-disk cache path (honours ``XDG_CACHE_HOME``)."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "sx" / "update-check.json"


def _read_cache() -> dict | None:
    """Return the cached check result, or ``None`` if absent/unusable.

    A cache file holding valid JSON that is not an object (``[]``, ``null``, a
    bare string) must be rejected here, not merely parsed: callers go straight to
    ``cache.get(...)``, so returning a list would raise ``AttributeError`` and
    crash ``sx update``.
    """
    try:
        data = json.loads(_cache_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_cache(latest: str | None) -> None:
    """Persist the latest-known version and the current time (best effort)."""
    path = _cache_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"checked_at": time.time(), "latest": latest}),
            encoding="utf-8",
        )
    except OSError:
        pass


def _ttl_seconds() -> float:
    """Return the cache TTL in seconds (from env or the default)."""
    try:
        hours = float(os.environ.get("SX_UPDATE_TTL_HOURS", DEFAULT_TTL_HOURS))
    except ValueError:
        hours = DEFAULT_TTL_HOURS
    return hours * 3600.0


def _timeout() -> float:
    """Return the network timeout in seconds (from env or the default)."""
    try:
        return float(os.environ.get("SX_UPDATE_TIMEOUT", DEFAULT_TIMEOUT))
    except ValueError:
        return DEFAULT_TIMEOUT


def _get_json(url: str, timeout: float):
    """GET a URL and parse JSON, or return ``None`` on any failure.

    Args:
        url: The GitHub API URL.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON (dict or list), or ``None``.
    """
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "sx-update-check",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Bounded read: the socket timeout limits each read, not the total
            # transfer, so an oversized body could otherwise exhaust memory.
            body = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(body) > _MAX_RESPONSE_BYTES:
                return None
            return json.loads(body.decode("utf-8"))
    except Exception:  # noqa: BLE001 - network/parse errors are non-fatal
        return None


def fetch_latest_version(timeout: float | None = None) -> str | None:
    """Fetch the latest published version from GitHub.

    Tries the ``releases/latest`` endpoint first, then falls back to the highest
    semver-looking git tag. Returns ``None`` if neither is available (e.g. the
    project has not published a release yet) or on any network error.

    Args:
        timeout: Optional socket timeout override.

    Returns:
        A version string such as ``"0.2.0"``, or ``None``.
    """
    t = timeout if timeout is not None else _timeout()

    release = _get_json(f"https://api.github.com/repos/{REPO}/releases/latest", t)
    if isinstance(release, dict):
        tag = release.get("tag_name")
        if isinstance(tag, str) and tag:
            return tag.lstrip("v")

    tags = _get_json(f"https://api.github.com/repos/{REPO}/tags?per_page=100", t)
    if isinstance(tags, list):
        parsed = []
        for entry in tags:
            name = entry.get("name") if isinstance(entry, dict) else None
            version = _parse_version(name) if name else None
            if version is not None:
                parsed.append((version, name.lstrip("v")))
        if parsed:
            parsed.sort()
            return parsed[-1][1]

    return None


def check_for_update(*, force: bool = False, timeout: float | None = None) -> UpdateInfo | None:
    """Return an :class:`UpdateInfo` if a newer release exists, else ``None``.

    Uses the on-disk cache to avoid hitting the network more than once per TTL.
    Never raises: any failure results in ``None``.

    Args:
        force: If True, bypass the cache and always query GitHub.
        timeout: Optional socket timeout override.

    Returns:
        An :class:`UpdateInfo` when an upgrade is available, otherwise ``None``.
    """
    if opted_out():
        return None

    cache = _read_cache()
    latest: str | None = None

    fresh = (
        cache is not None
        and isinstance(cache.get("checked_at"), (int, float))
        and (time.time() - cache["checked_at"]) < _ttl_seconds()
    )
    if not force and fresh:
        latest = cache.get("latest")
    else:
        latest = fetch_latest_version(timeout)
        if latest is not None:
            _write_cache(latest)
        else:
            # Record the miss too. Without this the throttle never engages when
            # a project has no published release, so every single run would hit
            # the API twice and stall on a slow network.
            _write_cache(None)
            if cache is not None:
                latest = cache.get("latest")  # stale value beats nothing offline

    if not latest:
        return None

    current = current_version()
    if is_newer(latest, current):
        return UpdateInfo(current=current, latest=latest)
    return None
