"""Command-line entry point for ``sx``.

Commands:

* bare ``sx``      — launch the interactive TUI.
* ``sx list``      — list discovered sessions, grouped by harness then project.
* ``sx move``      — re-point a project's sessions at a new directory (and,
  with ``--relocate``, move the project directory itself first).
* ``sx memory``    — list, export or delete the memory a harness keeps between
  sessions for a project.
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
from pathlib import Path

from sx import __version__
from sx import update as update_mod
from sx.memory import discover_memories, export_memory
from sx.model import Capability, Session
from sx.registry import build_registry
from sx.service import DeleteService, MoveService, move_summary
from sx.util import (
    CLAUDE_CONFIG_ENV,
    claude_config_override,
    claude_config_warning,
    human_size,
    sanitize_text,
)

#: Follows the live-session count in a move plan; the TUI says the same.
LIVE_MOVE_WARNING = (
    "moving a session its harness still has open breaks it: the next write "
    "fails, or lands in a header-less fragment at the old path"
)


def _safe(value: object) -> str:
    """Return ``value`` as one line of text that cannot drive the terminal.

    For anything read off disk — titles, ids, paths, and the notes and reasons
    that embed them — on its way to stdout. Newlines are folded into spaces so
    one value cannot pose as several lines of output.

    Args:
        value: The value, or anything ``str()`` turns into it.

    Returns:
        The text, control characters replaced by visible glyphs.
    """
    return sanitize_text(str(value).replace("\n", " "))


def _open_session_caveat(adapter) -> str | None:
    """Return what an adapter says about moving a session it has open.

    ``open_session_caveat`` is optional on adapters, so it is read with
    ``getattr``.

    Args:
        adapter: A harness adapter, or ``None``.

    Returns:
        The sentence, sanitized and with an initial capital, or ``None``.
    """
    caveat = getattr(adapter, "open_session_caveat", None)
    if not isinstance(caveat, str) or not caveat.strip():
        return None
    caveat = _safe(caveat.strip())
    return caveat[:1].upper() + caveat[1:]


def _store_owner(adapter) -> str | None:
    """Return the harness an adapter has handed its store over to, if any.

    ``store_owner`` is optional on adapters, so it is read with ``getattr``;
    one that raises counts as no answer.

    Args:
        adapter: A harness adapter.

    Returns:
        The other harness's display name, sanitized, or ``None``.
    """
    method = getattr(adapter, "store_owner", None)
    if not callable(method):
        return None
    try:
        owner = method()
    except Exception:  # noqa: BLE001 - one adapter must not break the CLI
        return None
    if not isinstance(owner, str) or not owner.strip():
        return None
    return _safe(owner.strip())


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


def _registry(args: argparse.Namespace) -> tuple[list, list[tuple[str, str]]]:
    """Return this run's adapter registry, building it on first use.

    ``main`` needs the adapters for their store warnings before a command runs,
    and most commands need them again; caching the result on ``args`` keeps
    that to one instantiation per run.

    Args:
        args: The parsed CLI arguments the registry is cached on.

    Returns:
        ``(adapters, errors)``, as from :func:`sx.registry.build_registry`.
    """
    registry = getattr(args, "registry", None)
    if registry is None:
        registry = build_registry()
        args.registry = registry
    return registry


def _store_lines(method) -> list[str]:
    """Call an adapter's ``store_notes`` or ``store_warnings``, never raising.

    Args:
        method: The bound method to call.

    Returns:
        Its lines, sanitized for the terminal. A check that raises is reported
        as a line of its own rather than aborting the command.
    """
    try:
        return [sanitize_text(str(line)) for line in method()]
    except Exception as exc:  # noqa: BLE001 - one adapter must not break the CLI
        return [f"store check failed: {exc!r}"]


def cmd_list(args: argparse.Namespace) -> int:
    """Handle ``sx list``: print discovered sessions grouped by harness/project.

    Args:
        args: Parsed CLI arguments (uses ``args.harness`` optional filter).

    Returns:
        Process exit code.
    """
    adapters, errors = _registry(args)
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
            # Projects, ids and titles are all read from the session files.
            print(f"  {_safe(project)}{flag}")
            for session in sessions_in_project:
                count = "?" if session.message_count is None else str(session.message_count)
                short_id = _safe(session.session_id[:8])
                size = human_size(session.size_bytes)
                title = _safe(session.title or "").strip()
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
    _maybe_notify_update()
    return 0


def cmd_harnesses(args: argparse.Namespace) -> int:
    """Handle ``sx harnesses``: show every known harness and its status.

    After the list come each adapter's notes about where its store was found,
    and its warnings about the store, marked ``⚠``. A harness that handed its
    store over to another one names the owner instead of reading as not
    installed.

    Args:
        args: Parsed CLI arguments (used only to share the run's registry).

    Returns:
        Process exit code.
    """
    _maybe_notify_update()
    adapters, errors = _registry(args)
    _print_load_errors(errors)

    print("Known harnesses:\n")
    for adapter in adapters:
        installed = adapter.available()
        browsable = Capability.BROWSE in adapter.capabilities
        if not browsable:
            status = "dormant (not yet supported)"
        elif installed:
            status = "ready"
        elif owner := _store_owner(adapter):
            status = f"store belongs to {owner}"
        else:
            status = "not installed"
        mark = "✓" if (installed and browsable) else ("○" if installed else "·")
        print(f"  {mark} {adapter.display:<14} {status}")

    override = claude_config_override()
    if override is not None:
        print(f"\n  Claude Code config home: {override}  ({CLAUDE_CONFIG_ENV})")

    lines: list[str] = []
    for adapter in adapters:
        lines += [f"  {adapter.display}: {note}" for note in _store_lines(adapter.store_notes)]
        lines += [
            f"  ⚠ {adapter.display}: {warning}"
            for warning in _store_lines(adapter.store_warnings)
        ]
    if lines:
        print()
        print("\n".join(lines))
    return 0


def _print_move_plan(plans: dict, results: dict, adapters: dict) -> tuple[int, int, int]:
    """Print the per-harness plan and return ``(files, fields, relocations)``.

    Printed whatever the flags, ``--yes`` included: the live-session warning
    and each harness's open-session caveat are part of the plan, not of the
    confirmation that ``--yes`` skips.
    """
    files = fields = moves = 0
    for name, plan in plans.items():
        result = results[name]
        adapter = adapters.get(name)
        label = adapter.display if adapter else _safe(name)
        print(f"\n  {label} ({len(plan.sessions)} session(s))")
        caveat = _open_session_caveat(adapter)
        if caveat and plan.sessions:
            print(f"    ⚠ {caveat}")
        for path in result.rewritten:
            print(f"    • re-point {_safe(path)}")
        for src, dst in result.moved:
            print(f"    • move  {_safe(src)}")
            print(f"            → {_safe(dst)}")
        if result.note:
            print(f"    • {_safe(result.note)}")
        if plan.live:
            print(f"    ● {len(plan.live)} live session(s) — {LIVE_MOVE_WARNING}")
        for path, reason in result.refused.items():
            print(f"    ✗ {_safe(path)} — {_safe(reason)}")
        files += len(result.rewritten)
        fields += result.fields_updated
        moves += len(result.moved)
    return files, fields, moves


def _confirm_move(*, strict: bool) -> bool:
    """Ask for confirmation on a TTY, refusing outright when there is none.

    Args:
        strict: Require the phrase ``MOVE`` rather than a simple yes.

    Returns:
        True if the user confirmed.
    """
    if not sys.stdin.isatty():
        print(
            "Refusing to proceed without confirmation: not an interactive "
            "terminal. Re-run with --yes.",
            file=sys.stderr,
        )
        return False
    try:
        if strict:
            return input('\nType "MOVE" to proceed: ').strip() == "MOVE"
        return input("\nProceed? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def cmd_move(args: argparse.Namespace) -> int:
    """Handle ``sx move``: re-point a project's sessions at a new directory.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code: non-zero when anything was refused, or a dry run
        predicts a refusal — a script must be able to tell a partial move
        before it runs one, not only after.
    """
    old = Path(args.source).expanduser()
    new = Path(args.destination).expanduser()
    for label, path in (("--from", old), ("--to", new)):
        if not path.is_absolute():
            print(f"{label} must be an absolute path: {path}", file=sys.stderr)
            return 2
    if old == new:
        print("--from and --to are the same directory.", file=sys.stderr)
        return 2

    adapters, errors = _registry(args)
    _print_load_errors(errors)
    by_name = {a.name: a for a in adapters}
    service = MoveService(by_name)

    if args.relocate:
        reason = service.check_relocation(old, new)
        if reason is not None:
            print(f"The project directory cannot be moved: {_safe(reason)}", file=sys.stderr)
            return 1

    plans = service.plan(old, new, include_config=args.claude_config)
    if not plans and not args.relocate:
        print(f"No harness has sessions at {old}.")
        return 1

    print(f"Re-point sessions\n  from {old}\n    to {new}")
    if args.relocate:
        print("\n  the project directory itself moves first")
        if service.crosses_devices(old, new):
            print("  crosses filesystems — copies, then deletes the original")

    preview = service.move(plans, dry_run=True)
    files, fields, moves = _print_move_plan(plans, preview, by_name)
    print(
        f"\nTotal: {files} file(s) re-pointed, {fields} recorded path(s) updated, "
        f"{moves} relocation(s)"
    )

    if args.dry_run:
        refused = sum(len(result.refused) for result in preview.values())
        if refused:
            print(f"\nDry run — nothing was changed; {refused} target(s) would be refused.")
            return 1
        print("\nDry run — nothing was changed.")
        return 0

    live = sum(len(plan.live) for plan in plans.values())
    strict = bool(live or args.relocate)
    if not args.yes and not _confirm_move(strict=strict):
        print("Canceled — nothing was changed.")
        return 1

    if args.relocate:
        reason = service.relocate_project(old, new)
        if reason is not None:
            print(f"Nothing was moved — {_safe(reason)}", file=sys.stderr)
            return 1
        print(f"Moved {old} → {new}")

    plans = service.plan(old, new, include_config=args.claude_config)
    if not args.yes and not strict and any(plan.live for plan in plans.values()):
        # Confirmed with a plain yes because nothing was live when the plan was
        # shown; a session that went live at the prompt needs the phrase.
        print(
            "Nothing was moved — a session became active while waiting for "
            "confirmation. Run the move again.",
            file=sys.stderr,
        )
        return 1
    results = service.move(plans)
    print(_safe(move_summary(results)))
    if service.last_log_error:
        print(
            f"⚠ the move happened but was NOT logged — {_safe(service.last_log_error)}",
            file=sys.stderr,
        )
    # Anything refused means the move did not do everything it was asked to,
    # even when the rest succeeded: a script must be able to tell.
    if any(result.refused for result in results.values()):
        return 1
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    """Handle ``sx memory``: list, export or delete project memory.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    memories = discover_memories()
    if args.project:
        wanted = str(Path(args.project).expanduser())
        memories = [m for m in memories if m.project_path == wanted]
    if not memories:
        print("No project memory found." if not args.project else f"No memory for {args.project}.")
        return 1

    current = object()
    for memory in memories:
        if memory.project_path != current:
            current = memory.project_path
            print(f"\n  {_safe(current or '(unknown project)')}")
        # A memory's name and type are model-written frontmatter.
        print(
            f"    {_safe(memory.name[:38]):<38} {_safe(memory.kind):<10} "
            f"from {_safe(memory.origin_label):<18} {human_size(memory.size_bytes):>9}"
        )
    total = sum(m.size_bytes for m in memories)
    print(f"\n{len(memories)} memory file(s) · {human_size(total)}")

    if args.export:
        dest = Path(args.export).expanduser()
        for memory in memories:
            print(f"  exported → {_safe(export_memory(memory, dest))}")

    if not args.delete:
        return 0

    if args.dry_run:
        print("\nWould permanently delete:")
        for memory in memories:
            print(f"  {_safe(memory.path)}")
        print("\nDry run — nothing was changed.")
        return 0

    print(
        "\nMemory is kept between sessions on purpose. Deleting it is permanent, "
        "and the sessions that wrote it will not restore it."
    )
    if not args.yes and not _confirm_move(strict=True):
        print("Canceled — nothing was changed.")
        return 1

    adapters, _ = _registry(args)
    service = DeleteService({a.name: a for a in adapters})
    removed = freed = 0
    refused: list[str] = []
    for memory in memories:
        result = service.delete_memory(memory)
        if result.failed:
            refused.append(next(iter(result.refused.values()), "refused"))
            continue
        removed += 1
        freed += result.freed_bytes
    print(f"Deleted {removed} memory file(s) · freed {human_size(freed)}")
    if refused:
        print(f"⚠ {len(refused)} refused ({_safe(refused[0])})", file=sys.stderr)
    return 0 if removed else 1


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

    p_move = sub.add_parser(
        "move",
        help="Re-point a project's sessions at a new directory.",
        description=(
            "Tell every harness that a project now lives somewhere else. Use "
            "--relocate when the directory has not been moved yet and sx should "
            "move it too. Reversible: run the inverse move to undo."
        ),
    )
    p_move.add_argument(
        "--from", dest="source", required=True, metavar="PATH",
        help="The project directory the sessions currently point at.",
    )
    p_move.add_argument(
        "--to", dest="destination", required=True, metavar="PATH",
        help="The project directory they should point at instead.",
    )
    p_move.add_argument(
        "--relocate", action="store_true",
        help="Move the project directory itself first, then re-point the sessions.",
    )
    p_move.add_argument(
        "--dry-run", action="store_true",
        help="Show exactly what would change and stop; exit 1 if any of it would be refused.",
    )
    p_move.add_argument(
        "--claude-config", action="store_true",
        help=(
            "Also re-point Claude's own project state (the .claude.json trust "
            "and allowedTools entry, and history.jsonl)."
        ),
    )
    p_move.add_argument(
        "--yes", action="store_true", help="Skip the interactive confirmation."
    )
    p_move.set_defaults(func=cmd_move)

    p_memory = sub.add_parser(
        "memory",
        help="List, export or delete project memory.",
        description=(
            "Project memory is what a harness keeps between sessions for a "
            "directory — facts, corrections, ongoing state. It belongs to the "
            "project rather than to any conversation, so deleting a session "
            "never removes it."
        ),
    )
    p_memory.add_argument(
        "--project", metavar="PATH", help="Only memories for this project directory."
    )
    p_memory.add_argument(
        "--export", metavar="DIR", help="Copy every listed memory into DIR."
    )
    p_memory.add_argument(
        "--delete", action="store_true", help="Permanently delete every listed memory."
    )
    p_memory.add_argument(
        "--dry-run", action="store_true", help="Show what --delete would remove and stop."
    )
    p_memory.add_argument(
        "--yes", action="store_true", help="Skip the interactive confirmation."
    )
    p_memory.set_defaults(func=cmd_memory)

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

    warning = claude_config_warning()
    if warning:
        print(f"⚠ {warning}", file=sys.stderr)
    # `sx harnesses` lists these itself, after its table; printing them here
    # too would show every warning twice.
    if getattr(args, "command", None) != "harnesses":
        for adapter in _registry(args)[0]:
            for line in _store_lines(adapter.store_warnings):
                print(f"⚠ {adapter.display}: {line}", file=sys.stderr)

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
