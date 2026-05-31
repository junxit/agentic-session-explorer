"""Command-line entry point for ``sx``.

Commands:

* bare ``sx``      — launch the interactive TUI.
* ``sx list``      — list discovered sessions, grouped by harness then project.
* ``sx harnesses`` — show every known harness and its status.
* ``sx version``   — show the installed version and check GitHub for a newer one.
* ``sx update``    — show (or run) the upgrade command when a newer release exists.

When run interactively, ``sx`` checks GitHub at most once a day for a newer
release and prints a one-line notice. Disable with ``--no-update-check`` or
``SX_NO_UPDATE_CHECK=1``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from sx import __version__
from sx import update as update_mod
from sx.model import Capability, Session
from sx.registry import build_registry
from sx.util import human_size


def _maybe_notify_update() -> None:
    """Print a one-line upgrade notice to stderr when interactive.

    Cheap and safe: skips entirely when stderr is not a TTY (so piped/scripted
    output stays clean) or the check is opted out, and never raises.
    """
    if not sys.stderr.isatty() or update_mod.opted_out():
        return
    try:
        info = update_mod.check_for_update()
    except Exception:  # noqa: BLE001 - update check must never break the CLI
        return
    if info is not None:
        print(
            f"↑ sx {info.latest} is available (you have {info.current}). "
            f"Upgrade: {info.command}",
            file=sys.stderr,
        )


def _sort_key(session: Session) -> float:
    """Sort key placing most-recently-modified sessions first (missing last)."""
    return session.modified.timestamp() if session.modified else 0.0


def _fmt_date(session: Session) -> str:
    """Format a session's modified date as ``YYYY-MM-DD`` (or padded blank)."""
    if session.modified:
        return session.modified.strftime("%Y-%m-%d")
    return "          "


def _print_load_errors(errors: list[tuple[str, str]]) -> None:
    """Print any adapter load errors to stderr."""
    for name, err in errors:
        print(f"⚠ adapter {name} failed to load: {err}", file=sys.stderr)


def cmd_list(args: argparse.Namespace) -> int:
    """Handle ``sx list``: print discovered sessions grouped by harness/project.

    Args:
        args: Parsed CLI arguments (uses ``args.harness`` optional filter).

    Returns:
        Process exit code.
    """
    adapters, errors = build_registry()
    _print_load_errors(errors)

    total = 0
    for adapter in adapters:
        if args.harness and adapter.name != args.harness:
            continue
        if Capability.BROWSE not in adapter.capabilities:
            continue
        if not adapter.available():
            continue

        try:
            sessions = list(adapter.discover())
        except Exception as exc:  # noqa: BLE001 - never let one harness break list
            print(f"⚠ {adapter.display}: discovery failed: {exc!r}", file=sys.stderr)
            continue

        if not sessions:
            continue

        total += len(sessions)
        print(f"\n=== {adapter.display} ({len(sessions)}) ===")

        groups: dict[str, list[Session]] = {}
        for session in sessions:
            key = session.project_path or "(unknown project)"
            groups.setdefault(key, []).append(session)

        for project in sorted(groups):
            sessions_in_project = sorted(groups[project], key=_sort_key, reverse=True)
            flag = "  ⚠ orphan" if any(s.is_orphan for s in sessions_in_project) else ""
            print(f"  {project}{flag}")
            for session in sessions_in_project:
                count = "?" if session.message_count is None else str(session.message_count)
                short_id = session.session_id[:8]
                size = human_size(session.size_bytes)
                title = (session.title or "").replace("\n", " ").strip()
                if len(title) > 60:
                    title = title[:57] + "..."
                print(
                    f"    {short_id}  {_fmt_date(session)}  "
                    f"{count:>4} msgs  {size:>9}  {title}"
                )

    if total == 0:
        print("No sessions found.")
    else:
        print(f"\n{total} session(s) across all harnesses.")
    return 0


def cmd_harnesses(args: argparse.Namespace) -> int:
    """Handle ``sx harnesses``: show every known harness and its status.

    Args:
        args: Parsed CLI arguments (unused).

    Returns:
        Process exit code.
    """
    _maybe_notify_update()
    adapters, errors = build_registry()
    _print_load_errors(errors)

    print("Known harnesses:\n")
    for adapter in adapters:
        installed = adapter.available()
        browsable = Capability.BROWSE in adapter.capabilities
        if not browsable:
            status = "dormant (not yet supported)"
        elif installed:
            status = "ready"
        else:
            status = "not installed"
        mark = "✓" if (installed and browsable) else ("○" if installed else "·")
        print(f"  {mark} {adapter.display:<14} {status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``sx`` command."""
    parser = argparse.ArgumentParser(
        prog="sx",
        description="Browse and delete AI coding-harness sessions from the terminal.",
    )
    parser.add_argument("--version", action="version", version=f"sx {__version__}")
    parser.add_argument(
        "--no-update-check",
        action="store_true",
        help="Skip the once-a-day check for a newer release on GitHub.",
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List discovered sessions.")
    p_list.add_argument(
        "--harness",
        help="Only list sessions from this harness (e.g. claude, codex, gemini).",
    )
    p_list.set_defaults(func=cmd_list)

    p_harnesses = sub.add_parser("harnesses", help="Show all known harnesses and status.")
    p_harnesses.set_defaults(func=cmd_harnesses)

    p_tui = sub.add_parser("tui", help="Launch the interactive TUI (default).")
    p_tui.set_defaults(func=cmd_tui)

    p_version = sub.add_parser(
        "version", help="Show the installed version and check for a newer one."
    )
    p_version.set_defaults(func=cmd_version)

    p_update = sub.add_parser(
        "update", help="Show (or run) the upgrade command if a newer release exists."
    )
    p_update.add_argument(
        "--run", action="store_true", help="Run the upgrade command instead of printing it."
    )
    p_update.set_defaults(func=cmd_update)

    return parser


def cmd_tui(args: argparse.Namespace) -> int:
    """Handle ``sx tui``: launch the interactive browser.

    Args:
        args: Parsed CLI arguments (unused).

    Returns:
        Process exit code.
    """
    from sx.tui.app import run_app

    return run_app(check_updates=not update_mod.opted_out())


def cmd_version(args: argparse.Namespace) -> int:
    """Handle ``sx version``: show installed version and the latest release.

    Args:
        args: Parsed CLI arguments (unused).

    Returns:
        Process exit code.
    """
    current = update_mod.current_version()
    print(f"sx {current}")
    if update_mod.opted_out():
        print("(update check disabled via SX_NO_UPDATE_CHECK)")
        return 0
    latest = update_mod.fetch_latest_version()
    if latest is None:
        print("Latest release: unknown (no release published yet, or offline).")
    elif update_mod.is_newer(latest, current):
        print(f"Latest release: {latest} — upgrade with: {update_mod.UPGRADE_COMMAND}")
    else:
        print(f"Latest release: {latest} — you are up to date.")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """Handle ``sx update``: report or run the upgrade when one is available.

    Args:
        args: Parsed CLI arguments (uses ``args.run``).

    Returns:
        Process exit code.
    """
    info = update_mod.check_for_update(force=True)
    if info is None:
        print("sx is up to date (or the latest release could not be determined).")
        return 0
    print(f"sx {info.latest} is available (you have {info.current}).")
    if not args.run:
        print(f"Upgrade with: {info.command}")
        print("Or re-run with --run to execute it now.")
        return 0
    print(f"Running: {info.command}")
    try:
        completed = subprocess.run(info.command.split(), check=False)
    except OSError as exc:
        print(f"Could not run upgrade: {exc}", file=sys.stderr)
        return 1
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``sx`` console script.

    Args:
        argv: Optional explicit argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # A global --no-update-check disables the check for this run (and any
    # subprocess) by setting the same env var users can set permanently.
    if getattr(args, "no_update_check", False):
        import os

        os.environ["SX_NO_UPDATE_CHECK"] = "1"

    # Bare `sx` launches the interactive TUI.
    if not getattr(args, "command", None):
        return cmd_tui(args)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
