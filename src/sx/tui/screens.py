"""Modal and full-screen dialogs for the sx TUI.

* :class:`ConfirmDeleteScreen` — a modal that previews exactly what will be
  permanently removed, optionally exports first, and (for live sessions or bulk
  operations) requires a typed confirmation phrase before enabling Delete.
* :class:`MoveScreen` — a modal that takes a new project directory, previews
  exactly what each harness would re-point, and gates the move behind a typed
  confirmation when it would touch a live session or move a real directory.
* :class:`MemoryScreen` — a full screen listing the memory documents a harness
  keeps between sessions, with export and delete.
* :class:`OrphanScreen` — a full screen listing orphaned artifacts across
  harnesses, from which they can be cleaned up.
"""

from __future__ import annotations

from rich.text import Text

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
)

from sx.model import Orphan
from sx.util import human_size, sanitize_text


class ConfirmDeleteScreen(ModalScreen):
    """Confirm a permanent deletion.

    The screen dismisses with ``None`` if canceled, or a result dict
    ``{"export": bool}`` if confirmed. When ``typed_phrase`` is set the Delete
    button stays disabled until the user types that exact phrase — used for live
    sessions and bulk operations.

    Args:
        heading: Short description of what is being deleted.
        preview: Pre-rendered preview text (files and sizes).
        can_export: Whether to offer "export to Markdown first".
        is_live: Whether the target appears to be in active use.
        typed_phrase: Phrase the user must type to enable Delete, or ``None``.
        bundle_label: When set, offers an extra, unticked checkbox for removing
            the project's own state (its memory and settings) as well. Shown only
            when this is the project's last session, and never pre-selected —
            memory is written to outlive conversations, so taking it has to be a
            decision rather than a default.
    """

    CSS = """
    ConfirmDeleteScreen {
        align: center middle;
    }
    #dialog {
        width: 90;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        border: thick $error;
        background: $surface;
        padding: 1 2;
    }
    #heading { text-style: bold; color: $error; }
    #preview {
        height: auto;
        max-height: 18;
        margin: 1 0;
        border: round $panel-darken-2;
        padding: 0 1;
    }
    #live-warning { color: $warning; text-style: bold; margin-bottom: 1; }
    #buttons { height: auto; align-horizontal: right; margin-top: 1; }
    Button { margin-left: 2; }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        *,
        heading: str,
        preview: Text,
        can_export: bool = False,
        is_live: bool = False,
        typed_phrase: str | None = None,
        bundle_label: str | None = None,
    ) -> None:
        """Store dialog configuration."""
        super().__init__()
        self._heading = heading
        self._preview = preview
        self._can_export = can_export
        self._is_live = is_live
        self._typed_phrase = typed_phrase
        self._bundle_label = bundle_label

    def compose(self) -> ComposeResult:
        """Build the dialog widgets."""
        with Container(id="dialog"):
            yield Label(self._heading, id="heading")
            if self._is_live:
                yield Label(
                    "● LIVE — this session changed within the last 90s. "
                    "Delete only if you are sure.",
                    id="live-warning",
                )
            with VerticalScroll(id="preview"):
                yield Static(self._preview)
            if self._can_export:
                yield Checkbox("Export to Markdown before deleting", id="export")
            if self._bundle_label:
                yield Checkbox(self._bundle_label, id="bundle")
            if self._typed_phrase:
                yield Label(f'Type "{self._typed_phrase}" to confirm:')
                yield Input(id="confirm-input", placeholder=self._typed_phrase)
            with Horizontal(id="buttons"):
                yield Button("Cancel", variant="primary", id="cancel")
                yield Button("Delete permanently", variant="error", id="confirm",
                             disabled=bool(self._typed_phrase))

    def on_mount(self) -> None:
        """Focus the safest control by default."""
        if self._typed_phrase:
            self.query_one("#confirm-input", Input).focus()
        else:
            self.query_one("#cancel", Button).focus()

    @on(Input.Changed, "#confirm-input")
    def _on_typed(self, event: Input.Changed) -> None:
        """Enable Delete only when the typed phrase matches exactly."""
        match = event.value.strip() == self._typed_phrase
        self.query_one("#confirm", Button).disabled = not match

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self, event: Button.Pressed) -> None:
        """Dismiss with no result."""
        self.dismiss(None)

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self, event: Button.Pressed) -> None:
        """Dismiss with the chosen options."""
        export = False
        if self._can_export:
            export = self.query_one("#export", Checkbox).value
        bundle = False
        if self._bundle_label:
            bundle = self.query_one("#bundle", Checkbox).value
        self.dismiss({"export": export, "bundle": bundle})

    def action_cancel(self) -> None:
        """Escape cancels the dialog."""
        self.dismiss(None)


class OrphanScreen(Screen):
    """Full-screen list of orphaned artifacts, with cleanup actions.

    Args:
        orphans: The orphans to display.
        on_delete: Callback invoked with a single :class:`Orphan` to delete it;
            should return a result object with ``removed`` and ``freed_bytes``.
        on_delete_all: Callback invoked to delete every listed orphan.
        confirm: Async callable ``(orphan|None) -> bool`` the screen awaits to
            confirm a deletion (the app supplies the modal).
    """

    CSS = """
    #orphan-table { height: 1fr; }
    #orphan-summary { height: auto; padding: 0 1; color: $text-muted; }
    """

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("d", "delete_selected", "Delete selected"),
        Binding("D", "delete_all", "Delete ALL"),
        Binding("q", "go_back", "Back", show=False),
    ]

    def __init__(self, orphans, on_delete, on_delete_all, confirm) -> None:
        """Store orphans and callbacks."""
        super().__init__()
        self._orphans: list[Orphan] = list(orphans)
        self._on_delete = on_delete
        self._on_delete_all = on_delete_all
        self._confirm = confirm

    def compose(self) -> ComposeResult:
        """Build the orphan table layout."""
        yield Header(show_clock=False)
        yield Static(self._summary(), id="orphan-summary")
        table = DataTable(id="orphan-table", cursor_type="row", zebra_stripes=True)
        table.add_columns("Harness", "Kind", "Size", "Reason")
        yield table
        yield Footer()

    def on_mount(self) -> None:
        """Populate the table once mounted."""
        self.title = "sx — orphan cleanup"
        self._reload_table()

    def _summary(self) -> str:
        """Return a one-line summary of total orphans and reclaimable size."""
        total = sum(o.size_bytes for o in self._orphans)
        return (
            f"{len(self._orphans)} orphan(s) · {human_size(total)} reclaimable   "
            "[d] delete selected   [D] delete ALL   [esc] back"
        )

    def _reload_table(self) -> None:
        """Rebuild the table rows from the current orphan list."""
        table = self.query_one("#orphan-table", DataTable)
        table.clear()
        for orphan in self._orphans:
            # Cells are Text objects, never str: Textual runs str cells through
            # Text.from_markup, and orphan reasons embed untrusted data (a
            # project cwd read from a session file, or a .project_root's
            # contents). As markup that could restyle the table, emit real
            # terminal hyperlinks, or raise MarkupError and wedge this screen.
            table.add_row(
                Text(orphan.harness),
                Text(orphan.kind.value),
                Text(human_size(orphan.size_bytes)),
                Text(sanitize_text(orphan.reason)),
            )
        self.query_one("#orphan-summary", Static).update(self._summary())
        if not self._orphans:
            self.query_one("#orphan-summary", Static).update(
                "No orphans found.   [esc] back"
            )

    def action_go_back(self) -> None:
        """Return to the main browser."""
        self.dismiss(None)

    def action_delete_selected(self) -> None:
        """Delete the highlighted orphan (with confirmation)."""
        if not self._orphans:
            self.app.notify("No orphans to delete.", severity="warning")
            return
        table = self.query_one("#orphan-table", DataTable)
        row = table.cursor_row
        if row is None or row < 0 or row >= len(self._orphans):
            return
        self._run_delete_one(self._orphans[row])

    def action_delete_all(self) -> None:
        """Delete every listed orphan (with a typed confirmation)."""
        if not self._orphans:
            self.app.notify("No orphans to delete.", severity="warning")
            return
        self._run_delete_all()

    @work
    async def _run_delete_one(self, orphan: Orphan) -> None:
        """Confirm and delete a single orphan, then refresh the table.

        An orphan whose deletion was refused stays in the list: removing the row
        would tell the user it is gone when it is still on disk.
        """
        ok = await self._confirm([orphan])
        if not ok:
            return
        result = self._on_delete(orphan)
        if result.failed:
            reason = next(iter(result.refused.values()), "unknown reason")
            self.app.notify(
                f"Not deleted — {reason}", severity="error", timeout=10
            )
            return
        self._orphans.remove(orphan)
        self._reload_table()
        self.app.notify(f"Deleted orphan · {result.summary()}")

    @work
    async def _run_delete_all(self) -> None:
        """Confirm and delete all orphans, then refresh the table.

        Only the orphans that were actually removed leave the list; refused ones
        remain visible with an explanation.
        """
        ok = await self._confirm(list(self._orphans))
        if not ok:
            return
        freed = 0
        deleted = 0
        refused: list[str] = []
        for orphan in list(self._orphans):
            result = self._on_delete_all(orphan)
            if result.failed:
                refused.append(next(iter(result.refused.values()), "refused"))
                continue
            freed += result.freed_bytes
            deleted += 1
            self._orphans.remove(orphan)
        self._reload_table()

        if not deleted:
            self.app.notify(
                f"Nothing was deleted — {len(refused)} refused "
                f"({refused[0] if refused else 'unknown reason'})",
                severity="error",
                timeout=10,
            )
        elif refused:
            self.app.notify(
                f"Deleted {deleted} · freed {human_size(freed)} · "
                f"{len(refused)} refused ({refused[0]})",
                severity="warning",
                timeout=10,
            )
        else:
            self.app.notify(f"Deleted {deleted} orphan(s) · freed {human_size(freed)}")


class MoveScreen(ModalScreen):
    """Re-point a project's sessions at a new directory.

    The screen owns no logic of its own: it collects a destination and hands it
    to ``preview``, which returns the rendered plan along with whether the move
    can proceed and whether it needs a typed confirmation. Nothing is previewed
    until a destination is submitted, because planning walks the session stores.

    Dismisses with ``None`` if canceled, or ``{"new": str, "include_config":
    bool}`` if confirmed.

    Args:
        heading: Short description of the operation.
        old: The project directory the sessions currently point at.
        preview: Callable ``(new, include_config) -> (text, can_move,
            needs_phrase)`` producing the plan for a candidate destination.
        can_config: Whether to offer re-pointing Claude's own project settings.
        phrase: The phrase required when ``needs_phrase`` comes back True.
    """

    CSS = """
    MoveScreen {
        align: center middle;
    }
    #dialog {
        width: 96;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #heading { text-style: bold; color: $accent; }
    #from { color: $text-muted; margin-bottom: 1; }
    #preview {
        height: auto;
        max-height: 16;
        margin: 1 0;
        border: round $panel-darken-2;
        padding: 0 1;
    }
    #buttons { height: auto; align-horizontal: right; margin-top: 1; }
    Button { margin-left: 2; }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        *,
        heading: str,
        old: str,
        preview,
        can_config: bool = False,
        phrase: str = "MOVE",
    ) -> None:
        """Store dialog configuration."""
        super().__init__()
        self._heading = heading
        self._old = old
        self._preview = preview
        self._can_config = can_config
        self._phrase = phrase
        #: Destination the current preview was computed for; the Move button is
        #: only ever enabled for a plan the user has actually seen.
        self._previewed: str | None = None
        self._needs_phrase = False

    def compose(self) -> ComposeResult:
        """Build the dialog widgets."""
        with Container(id="dialog"):
            yield Label(self._heading, id="heading")
            yield Label(f"from  {self._old}", id="from")
            yield Input(value=self._old, placeholder="new absolute path…", id="destination")
            if self._can_config:
                yield Checkbox(
                    "Also re-point Claude's project settings and prompt history",
                    id="config",
                )
            with VerticalScroll(id="preview"):
                yield Static(
                    Text(
                        "Type the new directory and press enter to preview.",
                        style="dim italic",
                    ),
                    id="preview-body",
                )
            yield Label("", id="phrase-label")
            yield Input(id="confirm-input", classes="hidden")
            with Horizontal(id="buttons"):
                yield Button("Cancel", variant="primary", id="cancel")
                yield Button("Preview", id="refresh")
                yield Button("Move", variant="warning", id="confirm", disabled=True)

    def on_mount(self) -> None:
        """Focus the destination field, with the current path pre-filled."""
        self.query_one("#confirm-input", Input).display = False
        self.query_one("#phrase-label", Label).display = False
        self.query_one("#destination", Input).focus()

    # --- preview ---------------------------------------------------------

    def _include_config(self) -> bool:
        """Return whether the Claude-settings checkbox is ticked."""
        if not self._can_config:
            return False
        return self.query_one("#config", Checkbox).value

    def _run_preview(self) -> None:
        """Plan the move for the current destination and render the result."""
        destination = self.query_one("#destination", Input).value.strip()
        body = self.query_one("#preview-body", Static)
        confirm = self.query_one("#confirm", Button)
        if not destination:
            body.update(Text("Enter a destination.", style="yellow"))
            self._previewed = None
            confirm.disabled = True
            return

        text, can_move, needs_phrase = self._preview(destination, self._include_config())
        body.update(text)
        self._previewed = destination if can_move else None
        self._needs_phrase = needs_phrase and can_move

        label = self.query_one("#phrase-label", Label)
        phrase_input = self.query_one("#confirm-input", Input)
        label.display = self._needs_phrase
        phrase_input.display = self._needs_phrase
        if self._needs_phrase:
            label.update(f'Type "{self._phrase}" to confirm:')
            phrase_input.placeholder = self._phrase
            confirm.disabled = phrase_input.value.strip() != self._phrase
        else:
            confirm.disabled = not can_move

    @on(Input.Submitted, "#destination")
    def _on_destination_submitted(self, event: Input.Submitted) -> None:
        """Preview when the destination is submitted."""
        self._run_preview()

    @on(Input.Changed, "#destination")
    def _on_destination_changed(self, event: Input.Changed) -> None:
        """Invalidate a stale preview as soon as the destination is edited."""
        if event.value.strip() != self._previewed:
            self._previewed = None
            self.query_one("#confirm", Button).disabled = True

    @on(Checkbox.Changed, "#config")
    def _on_config_changed(self, event: Checkbox.Changed) -> None:
        """Re-plan when the optional Claude-settings step is toggled."""
        if self._previewed is not None:
            self._run_preview()

    @on(Button.Pressed, "#refresh")
    def _on_refresh(self, event: Button.Pressed) -> None:
        """Preview the current destination."""
        self._run_preview()

    @on(Input.Changed, "#confirm-input")
    def _on_typed(self, event: Input.Changed) -> None:
        """Enable Move only when the typed phrase matches exactly."""
        if self._needs_phrase:
            self.query_one("#confirm", Button).disabled = (
                event.value.strip() != self._phrase or self._previewed is None
            )

    # --- dismissal -------------------------------------------------------

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self, event: Button.Pressed) -> None:
        """Dismiss with no result."""
        self.dismiss(None)

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self, event: Button.Pressed) -> None:
        """Dismiss with the previewed destination, never an unplanned one."""
        if self._previewed is None:
            return
        self.dismiss({"new": self._previewed, "include_config": self._include_config()})

    def action_cancel(self) -> None:
        """Escape cancels the dialog."""
        self.dismiss(None)


class MemoryScreen(Screen):
    """Browse the memory documents a harness keeps between sessions.

    Memory belongs to a project directory rather than to any one conversation,
    so it never appears in the session tree and a session delete never removes
    it. This screen is where it can actually be read, archived and removed.

    Args:
        memories: The :class:`~sx.memory.MemoryFile` records to display.
        on_export: Called with a memory to archive it; returns the written path.
        on_delete: Called with a memory to remove it; returns a delete result.
        confirm: Async ``(list[MemoryFile]) -> bool`` used to confirm removals.
    """

    CSS = """
    #memory-table { height: 1fr; }
    #memory-summary { height: auto; padding: 0 1; color: $text-muted; }
    #memory-preview {
        height: 40%;
        border-top: solid $panel-darken-2;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("enter", "preview", "Preview"),
        Binding("e", "export_selected", "Export"),
        Binding("d", "delete_selected", "Delete"),
        Binding("D", "delete_project", "Delete project's"),
        Binding("q", "go_back", "Back", show=False),
    ]

    def __init__(self, memories, on_export, on_delete, confirm) -> None:
        """Store memories and callbacks."""
        super().__init__()
        self._memories = list(memories)
        self._on_export = on_export
        self._on_delete = on_delete
        self._confirm = confirm

    def compose(self) -> ComposeResult:
        """Build the memory table layout."""
        yield Header(show_clock=False)
        yield Static(self._summary(), id="memory-summary")
        table = DataTable(id="memory-table", cursor_type="row", zebra_stripes=True)
        table.add_columns("Project", "Memory", "Type", "From session", "Size")
        yield table
        with VerticalScroll(id="memory-preview"):
            yield Static(
                Text("Select a memory to read it.", style="dim italic"),
                id="memory-body",
            )
        yield Footer()

    def on_mount(self) -> None:
        """Populate the table once mounted."""
        self.title = "sx — project memory"
        self._reload_table()

    def _summary(self) -> str:
        """Return a one-line summary of the listed memories."""
        if not self._memories:
            return "No memories found.   [esc] back"
        total = sum(m.size_bytes for m in self._memories)
        projects = len({m.project_path for m in self._memories})
        orphaned = sum(
            1 for m in self._memories if m.origin_session_id and not m.origin_exists
        )
        bits = [
            f"{len(self._memories)} memory file(s) across {projects} project(s)",
            human_size(total),
        ]
        if orphaned:
            bits.append(f"{orphaned} from deleted session(s)")
        return "   ".join(bits) + "   [enter] preview  [e] export  [d] delete  [D] project's  [esc] back"

    def _reload_table(self) -> None:
        """Rebuild the table rows from the current memory list."""
        table = self.query_one("#memory-table", DataTable)
        table.clear()
        for memory in self._memories:
            # Cells are Text, never str: a memory's name and description are
            # model-written prose and Textual would parse a str cell as markup.
            table.add_row(
                Text(memory.project_name),
                Text(memory.name),
                Text(memory.kind),
                Text(memory.origin_label),
                Text(human_size(memory.size_bytes)),
            )
        self.query_one("#memory-summary", Static).update(self._summary())

    def _selected(self):
        """Return the memory under the cursor, or ``None``."""
        if not self._memories:
            return None
        row = self.query_one("#memory-table", DataTable).cursor_row
        if row is None or row < 0 or row >= len(self._memories):
            return None
        return self._memories[row]

    # --- actions ---------------------------------------------------------

    def action_go_back(self) -> None:
        """Return to the main browser."""
        self.dismiss(None)

    @on(DataTable.RowHighlighted, "#memory-table")
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Render the highlighted memory as the cursor moves."""
        self.action_preview()

    def action_preview(self) -> None:
        """Render the highlighted memory's contents in the preview pane.

        The body is read here rather than handed back to the app: these are small
        Markdown documents, and keeping the read local means the list keeps its
        place instead of the screen being dismissed to show one file.
        """
        body = self.query_one("#memory-body", Static)
        memory = self._selected()
        if memory is None:
            body.update(Text("Select a memory to read it.", style="dim italic"))
            return
        try:
            text = memory.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            body.update(Text(f"Could not read {memory.path}: {exc}", style="red"))
            return
        # Untrusted content: a memory is model-written and may quote anything.
        rendered = Text()
        rendered.append(f"{memory.path}\n", style="dim")
        if memory.description:
            rendered.append(f"{memory.description}\n", style="italic")
        rendered.append("\n")
        rendered.append(sanitize_text(text))
        body.update(rendered)

    @work
    async def action_export_selected(self) -> None:
        """Archive the highlighted memory to a Markdown file."""
        memory = self._selected()
        if memory is None:
            self.app.notify("No memory selected.", severity="warning")
            return
        try:
            path = self._on_export(memory)
        except Exception as exc:  # noqa: BLE001
            self.app.notify(f"Export failed: {exc!r}", severity="error")
            return
        self.app.notify(f"Exported → {path}")

    def action_delete_selected(self) -> None:
        """Delete the highlighted memory (with confirmation)."""
        memory = self._selected()
        if memory is None:
            self.app.notify("No memory selected.", severity="warning")
            return
        self._run_delete([memory])

    def action_delete_project(self) -> None:
        """Delete every memory belonging to the highlighted memory's project."""
        memory = self._selected()
        if memory is None:
            self.app.notify("No memory selected.", severity="warning")
            return
        group = [m for m in self._memories if m.project_path == memory.project_path]
        self._run_delete(group)

    @work
    async def _run_delete(self, memories: list) -> None:
        """Confirm and delete memories, then refresh the table.

        A memory whose deletion was refused stays in the list: removing the row
        would say it is gone while it is still on disk.
        """
        if not await self._confirm(memories):
            return
        removed = 0
        freed = 0
        refused: list[str] = []
        for memory in list(memories):
            result = self._on_delete(memory)
            if result.failed:
                refused.append(next(iter(result.refused.values()), "refused"))
                continue
            removed += 1
            freed += result.freed_bytes
            self._memories.remove(memory)
        self._reload_table()

        if not removed:
            self.app.notify(
                f"Nothing was deleted — {refused[0] if refused else 'unknown reason'}",
                severity="error",
                timeout=10,
            )
        elif refused:
            self.app.notify(
                f"Deleted {removed} · freed {human_size(freed)} · "
                f"{len(refused)} refused ({refused[0]})",
                severity="warning",
                timeout=10,
            )
        else:
            self.app.notify(f"Deleted {removed} memory file(s) · freed {human_size(freed)}")
