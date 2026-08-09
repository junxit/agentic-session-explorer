"""Modal and full-screen dialogs for the sx TUI.

* :class:`ConfirmDeleteScreen` — a modal that previews exactly what will be
  permanently removed, optionally exports first, and (for live sessions or bulk
  operations) requires a typed confirmation phrase before enabling Delete.
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
    ) -> None:
        """Store dialog configuration."""
        super().__init__()
        self._heading = heading
        self._preview = preview
        self._can_export = can_export
        self._is_live = is_live
        self._typed_phrase = typed_phrase

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
        self.dismiss({"export": export})

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
