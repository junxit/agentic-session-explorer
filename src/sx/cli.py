"""Command-line entry point for ``sx``.

Milestone M1 ships read-only commands:

* ``sx list``      — list discovered sessions, grouped by harness then project.
* ``sx harnesses`` — show every known harness and its status (installed,
  browsable, greyed/dormant).

The full TUI (``sx`` with no arguments) arrives in M2; until then a bare ``sx``
runs ``list`` so the tool is useful immediately.
"""

from __future__ import annotations

import argparse
import sys

from sx import __version__
from sx.model import Capability, Session
from sx.registry import build_registry
from sx.util import human_size


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

    return parser


def cmd_tui(args: argparse.Namespace) -> int:
    """Handle ``sx tui``: launch the interactive browser.

    Args:
        args: Parsed CLI arguments (unused).

    Returns:
        Process exit code.
    """
    from sx.tui.app import run_app

    return run_app()


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``sx`` console script.

    Args:
        argv: Optional explicit argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Bare `sx` launches the interactive TUI.
    if not getattr(args, "command", None):
        return cmd_tui(args)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
