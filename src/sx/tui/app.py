"""The ``sx`` Textual application: a two-pane session browser with cleanup.

Left pane: a tree of sessions, grouped by project / date / recency (cycle with
``m``) and filterable (``/``). Installed-but-unsupported harnesses appear greyed.
Right pane: the selected session's transcript, scrollable, parsed lazily on a
background thread.

Destructive actions are deliberate and gated:

* ``e`` exports the highlighted session to Markdown;
* ``d`` permanently deletes it, after a preview modal (and, for live sessions,
  a typed confirmation); deletion can export first;
* ``o`` opens the orphan-cleanup screen.

There is no undo — the guard rails are the preview, the typed confirmation, the
store-root allowlist (in the adapter), and the append-only op-log.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from rich.text import Text

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Tree
from textual.widgets.tree import TreeNode

from sx.export import export_session
from sx.grouping import GroupMode, filter_sessions, group_sessions
from sx.model import Capability, Orphan, Session
from sx.registry import build_registry
from sx.render import messages_to_text
from sx.service import DeleteService
from sx.tui.screens import ConfirmDeleteScreen, OrphanScreen
from sx.util import human_size

_READY_STYLE = "bold green"
_DORMANT_STYLE = "dim"


class SxApp(App):
    """Two-pane terminal browser for AI coding-harness sessions."""

    CSS = """
    #left { width: 52; border-right: solid $panel-darken-2; }
    #tree { padding: 0 1; height: 1fr; }
    #filter { display: none; margin: 0 1; }
    #filter.visible { display: block; }
    #transcript { padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_tree", "Refresh"),
        Binding("m", "cycle_group", "Group"),
        Binding("slash", "start_filter", "Filter"),
        Binding("o", "show_orphans", "Orphans"),
        Binding("e", "export_session", "Export"),
        Binding("d", "delete_session", "Delete"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("g", "scroll_top", "Top", show=False),
        Binding("G", "scroll_bottom", "Bottom", show=False),
        Binding("tab", "focus_next", "Switch pane", show=False),
    ]

    def __init__(self) -> None:
        """Initialize empty state; data is loaded on mount."""
        super().__init__()
        self._adapters_by_name: dict = {}
        self._sessions_by_harness: dict[str, list[Session]] = {}
        self._delete_service: DeleteService | None = None
        self._group_mode = GroupMode.PROJECT
        self._filter = ""

    def compose(self) -> ComposeResult:
        """Build the static widget layout."""
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left"):
                yield Input(placeholder="filter title/project…", id="filter")
                yield Tree("Harnesses", id="tree")
            yield RichLog(id="transcript", wrap=True, highlight=False, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        """Load adapters, build the tree, and show the welcome text."""
        self.title = "sx"
        self._reload_data()
        self._populate_tree()
        log = self.query_one("#transcript", RichLog)
        log.write(
            Text.from_markup(
                "[bold]sx[/bold] — select a session to view its transcript.\n\n"
                "[dim]↑/↓ move · enter open · m group · / filter · o orphans · "
                "e export · d delete · q quit[/dim]"
            )
        )

    # --- data ------------------------------------------------------------

    def _reload_data(self) -> None:
        """Re-scan every harness and rebuild the in-memory session lists."""
        adapters, errors = build_registry()
        self._adapters_by_name = {a.name: a for a in adapters}
        self._delete_service = DeleteService(self._adapters_by_name)
        self._load_errors = errors
        self._sessions_by_harness = {}
        for adapter in adapters:
            if Capability.BROWSE not in adapter.capabilities or not adapter.available():
                continue
            try:
                self._sessions_by_harness[adapter.name] = list(adapter.discover())
            except Exception as exc:  # noqa: BLE001
                self._sessions_by_harness[adapter.name] = []
                self._load_errors.append((adapter.display, repr(exc)))

    # --- tree construction ----------------------------------------------

    def _subtitle(self) -> str:
        """Return the header subtitle reflecting group mode and filter."""
        bits = [f"group: {self._group_mode.label()}"]
        if self._filter:
            bits.append(f"filter: {self._filter!r}")
        return " · ".join(bits)

    def _populate_tree(self) -> None:
        """Rebuild the tree from current data, group mode, and filter."""
        tree = self.query_one("#tree", Tree)
        tree.clear()
        tree.root.expand()
        self.sub_title = self._subtitle()

        for adapter_name, err in getattr(self, "_load_errors", []):
            tree.root.add_leaf(Text(f"⚠ {adapter_name}: {err}", style="red"))

        ready_names = list(self._sessions_by_harness)
        for name in ready_names:
            adapter = self._adapters_by_name[name]
            sessions = filter_sessions(self._sessions_by_harness[name], self._filter)
            self._add_ready_harness(tree.root, adapter, sessions)

        # Dormant / unavailable harnesses, greyed.
        for adapter in self._adapters_by_name.values():
            if adapter.name in self._sessions_by_harness:
                continue
            label = Text()
            label.append("· ", style=_DORMANT_STYLE)
            label.append(adapter.display, style=_DORMANT_STYLE)
            state = "dormant" if adapter.available() else "not installed"
            label.append(f"  ({state})", style="dim italic")
            tree.root.add(label, data=None, allow_expand=False)

    def _add_ready_harness(self, root: TreeNode, adapter, sessions: list[Session]) -> None:
        """Add one browsable harness and its grouped sessions to the tree."""
        header = Text()
        header.append("● ", style=_READY_STYLE)
        header.append(adapter.display, style=_READY_STYLE)
        header.append(f"  ({len(sessions)})", style="dim")
        harness_node = root.add(header, data=None, expand=True)

        if not sessions:
            harness_node.add_leaf(Text("(no matching sessions)", style="dim italic"))
            return

        for group_label, group_sessions_list in group_sessions(sessions, self._group_mode):
            orphan = any(s.is_orphan for s in group_sessions_list)
            glabel = Text()
            glabel.append(group_label, style="yellow" if orphan else "white")
            glabel.append(f"  ({len(group_sessions_list)})", style="dim")
            if orphan:
                glabel.append("  ⚠ orphan", style="bold yellow")
            group_node = harness_node.add(glabel, data=None, expand=True)
            for session in group_sessions_list:
                group_node.add_leaf(self._session_label(session), data=session)

    def _session_label(self, session: Session) -> Text:
        """Render a session leaf label (live badge, date, size, title)."""
        live = self._delete_service is not None and self._delete_service.is_active(session)
        date = session.modified.strftime("%Y-%m-%d") if session.modified else "          "
        title = (session.title or session.session_id).replace("\n", " ").strip()
        label = Text()
        if live:
            label.append("● ", style="bold red")
        label.append(date + "  ", style="dim")
        label.append(f"{human_size(session.size_bytes):>9}  ", style="cyan")
        label.append(title)
        return label

    # --- selection + lazy load ------------------------------------------

    def _current_session(self) -> Session | None:
        """Return the session under the tree cursor, if any."""
        tree = self.query_one("#tree", Tree)
        node = tree.cursor_node
        if node is not None and isinstance(node.data, Session):
            return node.data
        return None

    @on(Tree.NodeSelected)
    def _on_node_selected(self, event: Tree.NodeSelected) -> None:
        """Load a session's transcript when its leaf is selected."""
        if isinstance(event.node.data, Session):
            log = self.query_one("#transcript", RichLog)
            log.clear()
            log.write(Text("Loading…", style="dim italic"))
            self.open_session(event.node.data)

    @work(exclusive=True, thread=True)
    def open_session(self, session: Session) -> None:
        """Parse and display a session transcript off the UI thread."""
        adapter = self._adapters_by_name.get(session.harness)
        if adapter is None:
            self.call_from_thread(
                self._set_transcript, session,
                Text(f"No adapter for {session.harness!r}", style="red"), 0)
            return
        try:
            messages = adapter.load(session)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(
                self._set_transcript, session,
                Text(f"Failed to load: {exc!r}", style="red"), 0)
            return
        text = messages_to_text(messages)
        self.call_from_thread(self._set_transcript, session, text, len(messages))

    def _set_transcript(self, session: Session, text: Text, count: int) -> None:
        """Replace the transcript pane contents (UI thread)."""
        log = self.query_one("#transcript", RichLog)
        log.clear()
        log.write(text)
        log.scroll_home(animate=False)

    # --- filter ----------------------------------------------------------

    def action_start_filter(self) -> None:
        """Reveal and focus the filter input."""
        flt = self.query_one("#filter", Input)
        flt.add_class("visible")
        flt.value = self._filter
        flt.focus()

    @on(Input.Submitted, "#filter")
    def _on_filter_submitted(self, event: Input.Submitted) -> None:
        """Apply the filter and return focus to the tree."""
        self._filter = event.value.strip()
        self._populate_tree()
        self.query_one("#tree", Tree).focus()

    @on(Input.Changed, "#filter")
    def _on_filter_changed(self, event: Input.Changed) -> None:
        """Live-filter the tree as the query changes."""
        self._filter = event.value.strip()
        self._populate_tree()

    # --- actions ---------------------------------------------------------

    def action_refresh_tree(self) -> None:
        """Re-scan all harness stores and rebuild the tree."""
        self._reload_data()
        self._populate_tree()
        self.notify("Refreshed.")

    def action_cycle_group(self) -> None:
        """Cycle the grouping mode (project → date → recency)."""
        self._group_mode = self._group_mode.next()
        self._populate_tree()

    def action_cursor_down(self) -> None:
        """Move the tree cursor down (vim ``j``)."""
        self.query_one("#tree", Tree).action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the tree cursor up (vim ``k``)."""
        self.query_one("#tree", Tree).action_cursor_up()

    def action_scroll_top(self) -> None:
        """Scroll the transcript to the top (``g``)."""
        self.query_one("#transcript", RichLog).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        """Scroll the transcript to the bottom (``G``)."""
        self.query_one("#transcript", RichLog).scroll_end(animate=False)

    # --- export ----------------------------------------------------------

    @work
    async def action_export_session(self) -> None:
        """Export the highlighted session to Markdown."""
        session = self._current_session()
        if session is None:
            self.notify("No session selected.", severity="warning")
            return
        adapter = self._adapters_by_name[session.harness]
        try:
            path = await asyncio.to_thread(export_session, adapter, session)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Export failed: {exc!r}", severity="error")
            return
        self.notify(f"Exported → {path}")

    # --- delete ----------------------------------------------------------

    def _session_preview_text(self, session: Session) -> Text:
        """Render the delete preview for a session (paths + total size)."""
        assert self._delete_service is not None
        result = self._delete_service.preview(session)
        text = Text()
        text.append(f"{session.title}\n", style="bold")
        text.append(f"{session.harness} · {session.project_path or '(unknown)'}\n\n", style="dim")
        if not result.removed:
            text.append("Nothing to remove.\n", style="yellow")
        for path in result.removed:
            text.append(f"  • {path}\n")
        text.append(f"\nTotal: {len(result.removed)} path(s), "
                    f"{human_size(result.freed_bytes)}", style="bold")
        cascade = len(result.removed) - 1
        if cascade > 0:
            text.append(f"  (includes {cascade} correlated file(s))", style="dim")
        return text

    @work
    async def action_delete_session(self) -> None:
        """Permanently delete the highlighted session, after confirmation."""
        session = self._current_session()
        if session is None:
            self.notify("No session selected.", severity="warning")
            return
        assert self._delete_service is not None
        is_live = self._delete_service.is_active(session)
        preview = self._session_preview_text(session)
        screen = ConfirmDeleteScreen(
            heading="Permanently delete this session?",
            preview=preview,
            can_export=True,
            is_live=is_live,
            typed_phrase="DELETE" if is_live else None,
        )
        result = await self.push_screen_wait(screen)
        if result is None:
            return
        adapter = self._adapters_by_name[session.harness]
        if result.get("export"):
            try:
                path = await asyncio.to_thread(export_session, adapter, session)
                self.notify(f"Exported → {path}")
            except Exception as exc:  # noqa: BLE001
                self.notify(f"Export failed, aborting delete: {exc!r}", severity="error")
                return
        outcome = self._delete_service.delete(session)
        self.notify(
            f"Deleted {len(outcome.removed)} path(s) · freed {human_size(outcome.freed_bytes)}"
        )
        self._reload_data()
        self._populate_tree()

    # --- orphans ---------------------------------------------------------

    def _collect_orphans(self) -> list[Orphan]:
        """Gather orphans from every capable, available adapter."""
        orphans: list[Orphan] = []
        for adapter in self._adapters_by_name.values():
            if Capability.ORPHANS not in adapter.capabilities or not adapter.available():
                continue
            try:
                orphans.extend(adapter.find_orphans())
            except Exception:  # noqa: BLE001
                continue
        return orphans

    async def _confirm_orphan(self, orphans: list[Orphan]) -> bool:
        """Show a confirm modal for one or many orphans.

        The count shown and the typed-confirmation phrase are both derived from
        ``orphans`` (the exact list the caller will delete), so what the user
        sees always matches what they must type — no re-scan drift.

        Args:
            orphans: The orphan(s) about to be deleted.

        Returns:
            True if the user confirmed, False if they cancelled.
        """
        assert self._delete_service is not None
        if not orphans:
            return False

        if len(orphans) == 1:
            orphan = orphans[0]
            result = self._delete_service.preview_orphan(orphan)
            preview = Text()
            preview.append(f"{orphan.reason}\n", style="bold")
            preview.append(f"{orphan.harness} · {orphan.kind.value}\n\n", style="dim")
            for path in result.removed:
                preview.append(f"  • {path}\n")
            preview.append(f"\nTotal: {human_size(result.freed_bytes)}", style="bold")
            screen = ConfirmDeleteScreen(
                heading="Permanently delete this orphan?",
                preview=preview,
            )
        else:
            total = sum(o.size_bytes for o in orphans)
            preview = Text()
            preview.append(f"Delete ALL {len(orphans)} orphan(s)\n", style="bold")
            preview.append(f"Total reclaimable: {human_size(total)}\n", style="dim")
            screen = ConfirmDeleteScreen(
                heading="Permanently delete every orphan?",
                preview=preview,
                typed_phrase=f"DELETE {len(orphans)}",
            )
        result = await self.push_screen_wait(screen)
        return result is not None

    def action_show_orphans(self) -> None:
        """Open the orphan-cleanup screen."""
        assert self._delete_service is not None
        orphans = self._collect_orphans()
        screen = OrphanScreen(
            orphans=orphans,
            on_delete=self._delete_service.delete_orphan,
            on_delete_all=self._delete_service.delete_orphan,
            confirm=self._confirm_orphan,
        )
        self.push_screen(screen, lambda _result: self._after_orphans())

    def _after_orphans(self) -> None:
        """Refresh data after returning from the orphan screen."""
        self._reload_data()
        self._populate_tree()


def run_app() -> int:
    """Launch the interactive TUI.

    Returns:
        Process exit code (always ``0`` after a clean exit).
    """
    SxApp().run()
    return 0
