"""Modal and full-screen dialogs for the sx TUI.

* :class:`ConfirmDeleteScreen` — a modal that previews exactly what will be
  permanently removed, optionally exports first, and (for live sessions or bulk
  operations) requires a typed confirmation phrase before enabling Delete.
* :class:`BranchModeScreen` — a small modal asking how to export a session that
  branched: its current branch, the current branch plus the other branches, or
  the whole tree.
* :class:`BranchModeList` — the list both of those dialogs offer the branch
  views in, where the highlighted row is the choice.
* :class:`MoveScreen` — a modal that takes a new project directory, previews
  exactly what each harness would re-point, and gates the move behind a typed
  confirmation when it would touch a live session or move a real directory.
* :class:`MemoryScreen` — a full screen listing the memory documents a harness
  keeps between sessions, with export and delete.
* :class:`OrphanScreen` — a full screen listing orphaned artifacts across
  harnesses, from which they can be cleaned up.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.await_complete import AwaitComplete
from textual.binding import Binding, BindingType
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
    OptionList,
    Static,
)
from textual.widgets.option_list import Option

from sx.model import BranchMode, DeleteResult, Orphan
from sx.util import human_size, sanitize_text

#: How many paths a toast names before it only counts the rest.
_PATHS_NAMED = 3


def toast_safe(text: object) -> str:
    """Make untrusted text safe to pass to :meth:`~textual.app.App.notify`.

    Notifications are rendered as markup and these strings embed paths and
    reasons read off disk, so a directory containing square brackets would
    otherwise be parsed as a style tag. Control characters are neutralized too.

    Args:
        text: The text to show, or anything ``str()`` turns into it.

    Returns:
        The text with control characters replaced and ``[`` escaped.
    """
    return sanitize_text(str(text)).replace("[", r"\[")


def _mode_title(mode: BranchMode) -> str:
    """Return a mode's label with an initial capital, for headings."""
    label = mode.label()
    return label[:1].upper() + label[1:]


def _stayed_notice(stayed: dict[Path, str]) -> str:
    """Say which paths a delete left on disk, and why, for a toast.

    Args:
        stayed: Each path that stayed, with why; not empty.

    Returns:
        How many stayed, the first few by name, and the first one's reason —
        with how many other reasons there are, when they differ — escaped for
        :meth:`~textual.app.App.notify`.
    """
    paths = list(stayed)
    names = ", ".join(toast_safe(path.name or path) for path in paths[:_PATHS_NAMED])
    if len(paths) > _PATHS_NAMED:
        names += f" and {len(paths) - _PATHS_NAMED} more"
    reasons = list(dict.fromkeys(stayed.values()))
    why = toast_safe(reasons[0])
    if len(reasons) > 1:
        why += f"; {len(reasons) - 1} other reason(s)"
    return f"{len(paths)} path(s) stayed on disk: {names} ({why})"


def _what_stayed(orphan: Orphan, result: DeleteResult) -> Orphan:
    """Return ``orphan`` narrowed to the paths a partial delete of it left on disk.

    A path stays when it was refused, or holds something that was: an adapter
    may empty a directory around what it keeps.

    Args:
        orphan: The orphan as it was listed.
        result: What deleting it did; it removed something and refused something.

    Returns:
        The orphan with only the paths that stayed and the size not freed, or
        ``orphan`` itself when none of its paths holds a refusal.
    """
    refused = list(result.refused)
    stayed = [p for p in orphan.paths if any(r == p or p in r.parents for r in refused)]
    if not stayed:
        return orphan
    return replace(
        orphan, paths=stayed, size_bytes=max(0, orphan.size_bytes - result.freed_bytes)
    )


class _Dialog(ModalScreen):
    """A modal dialog that is dismissed at most once; later calls do nothing.

    :meth:`~textual.screen.Screen.dismiss` pops whatever screen is on top. Two
    quick Enters each choose before the first dismissal lands, so the second
    one popped the screen below the dialog too — and with nothing below that,
    raised ``ScreenStackError`` and took the app down.
    """

    _dismissed = False

    def dismiss(self, result=None) -> AwaitComplete:
        """Dismiss the dialog with ``result``, unless it was dismissed already.

        Args:
            result: What the dialog was dismissed with.

        Returns:
            What :meth:`~textual.screen.Screen.dismiss` returns, or an
            already-completed awaitable when this call is ignored.
        """
        if self._dismissed:
            return AwaitComplete.nothing()
        self._dismissed = True
        return super().dismiss(result)


class BranchModeList(OptionList):
    """The :class:`~sx.model.BranchMode` choices, where the highlighted row is the choice.

    Each row starts with a mark, ``●`` on the chosen row and ``○`` on the
    others, and the mark follows the highlight. There is no separate "pressed"
    state for the highlight to drift away from, and the choice still reads when
    the list loses focus or the terminal shows no color. Both dialogs that ask
    for a branch view use this list, so the arrows (or a click) choose in both.

    Args:
        descriptions: A one-line description of each mode offered, in order.
        chosen: The mode chosen when the list opens; the first one offered if
            it is not among them.
        **kwargs: Passed on to :class:`~textual.widgets.OptionList`.
    """

    def __init__(self, descriptions: dict[BranchMode, str], chosen: BranchMode, **kwargs) -> None:
        """Build one row per mode, with ``chosen`` highlighted."""
        # Set before the base class initializes: it highlights the first row
        # while adding the options, and that already runs the watcher below.
        self._descriptions = dict(descriptions)
        self._modes = list(self._descriptions)
        self._initial = self._modes.index(chosen) if chosen in self._descriptions else 0
        super().__init__(
            *(
                Option(self._prompt(index, index == self._initial), id=mode.value)
                for index, mode in enumerate(self._modes)
            ),
            **kwargs,
        )
        self.highlighted = self._initial

    def _prompt(self, index: int, chosen: bool) -> Text:
        """Render one row: its mark, the mode's name, and what it would include.

        Prompts are ``Text``, never ``str``: an ``OptionList`` parses a ``str``
        as markup, and the descriptions quote branch labels from the session.

        Args:
            index: The row's position.
            chosen: Whether the row is the choice.

        Returns:
            The row's prompt.
        """
        mode = self._modes[index]
        return Text.assemble(
            ("● " if chosen else "○ ", "bold"),
            (_mode_title(mode), "bold"),
            "\n  ",
            (sanitize_text(self._descriptions[mode]), "dim"),
        )

    def watch_highlighted(self, highlighted: int | None) -> None:
        """Move the mark along with the highlight.

        Args:
            highlighted: The row now highlighted, if any.
        """
        super().watch_highlighted(highlighted)
        for index in range(len(self._modes)):
            self.replace_option_prompt_at_index(index, self._prompt(index, index == highlighted))

    @property
    def chosen(self) -> BranchMode:
        """The mode on the highlighted row."""
        index = self.highlighted
        return self._modes[self._initial if index is None else index]


class ConfirmDeleteScreen(_Dialog):
    """Confirm a permanent deletion.

    The screen dismisses with ``None`` if canceled, or a result dict
    ``{"export": bool, "bundle": bool}`` if confirmed — plus ``"mode"``, the
    chosen :class:`~sx.model.BranchMode`, when ``export_modes`` was offered.
    When ``typed_phrase`` is set the Delete button stays disabled until the user
    types that exact phrase — used for live sessions and bulk operations.

    Args:
        heading: Short description of what is being deleted.
        preview: Pre-rendered preview text (files and sizes).
        can_export: Whether to offer "export to Markdown first".
        is_live: Whether the target appears to be in active use.
        typed_phrase: Phrase the user must type to enable Delete, or ``None``.
        bundle_label: When set, offers an extra, unticked checkbox for removing
            the project state the preview lists as well. Shown only when this
            is the project's last session, and never pre-selected — memory is
            written to outlive conversations, so taking it has to be a decision
            rather than a default.
        export_modes: For a session that branched, a one-line description of
            each :class:`~sx.model.BranchMode`, offered under the export
            checkbox in a :class:`BranchModeList` that is enabled only while
            the box is ticked. ``None`` (every session that never branched)
            leaves the dialog as it always was.
        export_mode: The mode chosen initially.
        caveat: What the adapters involved say about deleting something their
            harness has open, shown above the preview — one sentence per
            line, each marked ``⚠``; ``None`` for none.
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
        overflow-y: auto;
    }
    #heading { text-style: bold; color: $error; }
    #preview {
        height: auto;
        max-height: 18;
        margin: 1 0;
        border: round $panel-darken-2;
        padding: 0 1;
    }
    /* The full width, so the warnings wrap: a label sized to its text ran
       past the dialog's edge and was cut off there. */
    #live-warning { width: 100%; color: $warning; text-style: bold; margin-bottom: 1; }
    #caveat { width: 100%; color: $warning; text-style: bold; margin-bottom: 1; }
    #export-mode { margin-left: 4; }
    #buttons { height: auto; align-horizontal: right; margin-top: 1; }
    Button { margin-left: 2; }
    """

    #: The fewest preview rows, borders included, the dialog shrinks it to.
    _MIN_PREVIEW_ROWS = 5

    #: No automatic focus. Textual's own would focus the first focusable widget,
    #: the preview, and scroll to center it — animated, landing after the
    #: dialog was fitted and scrolled to its top. :meth:`on_mount` focuses the
    #: safest control instead, without scrolling.
    AUTO_FOCUS = ""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        *,
        heading: str,
        preview: Text,
        can_export: bool = False,
        is_live: bool = False,
        typed_phrase: str | None = None,
        bundle_label: str | None = None,
        export_modes: dict[BranchMode, str] | None = None,
        export_mode: BranchMode = BranchMode.TREE,
        caveat: str | None = None,
    ) -> None:
        """Store dialog configuration."""
        super().__init__()
        self._heading = heading
        self._preview = preview
        self._can_export = can_export
        self._is_live = is_live
        self._typed_phrase = typed_phrase
        self._bundle_label = bundle_label
        # The choice belongs to the export step, so it exists only with it.
        self._export_modes = dict(export_modes) if can_export and export_modes else {}
        self._export_mode = export_mode
        self._caveat = caveat

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
            if self._caveat:
                # Text, never str: a str label is parsed as markup.
                lines = sanitize_text(self._caveat).splitlines()
                yield Label(Text("\n".join(f"⚠ {line}" for line in lines)), id="caveat")
            with VerticalScroll(id="preview"):
                yield Static(self._preview)
            if self._can_export:
                yield Checkbox("Export to Markdown before deleting", id="export")
            if self._export_modes:
                yield BranchModeList(
                    self._export_modes,
                    self._export_mode,
                    id="export-mode",
                    compact=True,
                    disabled=True,
                )
            if self._bundle_label:
                yield Checkbox(Text(self._bundle_label), id="bundle")
            if self._typed_phrase:
                yield Label(f'Type "{self._typed_phrase}" to confirm:')
                yield Input(id="confirm-input", placeholder=self._typed_phrase)
            with Horizontal(id="buttons"):
                yield Button("Cancel", variant="primary", id="cancel")
                yield Button("Delete permanently", variant="error", id="confirm",
                             disabled=bool(self._typed_phrase))

    def on_mount(self) -> None:
        """Focus the safest control by default, and fit the dialog on screen.

        The control is focused where it is, without scrolling to it: when the
        dialog is taller than the screen, scrolling down to it hid the
        heading, the live warning and the preview above it.
        """
        if self._typed_phrase:
            self.query_one("#confirm-input", Input).focus(scroll_visible=False)
        else:
            self.query_one("#cancel", Button).focus(scroll_visible=False)
        self.call_after_refresh(self._fit_preview)

    def on_resize(self) -> None:
        """Fit the dialog to the new size, starting again from the full preview."""
        self.query_one("#preview").styles.max_height = None
        self.call_after_refresh(self._fit_preview)

    def _fit_preview(self) -> None:
        """Shrink the preview until the rest of the dialog fits on screen.

        The preview is the one part whose height depends on what is being
        deleted; everything below it is a control. When the dialog is taller
        than the screen allows, the preview gives up the rows it overflows by
        (it scrolls on its own), so the export choice, the typed phrase and the
        buttons stay in view. Below its minimum the dialog itself scrolls, and
        is then scrolled to its top: what is being deleted, and whether it is
        live, is read before the controls under it.
        """
        dialog = self.query_one("#dialog")
        overflow = dialog.max_scroll_y
        if overflow <= 0:
            return
        preview = self.query_one("#preview")
        rows = preview.outer_size.height - overflow
        preview.styles.max_height = max(self._MIN_PREVIEW_ROWS, rows)
        dialog.scroll_home(animate=False, immediate=True)

    @on(Input.Changed, "#confirm-input")
    def _on_typed(self, event: Input.Changed) -> None:
        """Enable Delete only when the typed phrase matches exactly."""
        match = event.value.strip() == self._typed_phrase
        self.query_one("#confirm", Button).disabled = not match

    @on(Checkbox.Changed, "#export")
    def _on_export_toggled(self, event: Checkbox.Changed) -> None:
        """Enable the branch choice only while the export step is ticked."""
        if self._export_modes:
            self.query_one("#export-mode", BranchModeList).disabled = not event.value

    def _chosen_mode(self) -> BranchMode:
        """Return the branch mode on the highlighted row of the export choice."""
        return self.query_one("#export-mode", BranchModeList).chosen

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
        result: dict = {"export": export, "bundle": bundle}
        if self._export_modes:
            result["mode"] = self._chosen_mode()
        self.dismiss(result)

    def action_cancel(self) -> None:
        """Escape cancels the dialog."""
        self.dismiss(None)


class BranchModeScreen(_Dialog):
    """Choose how a session that branched is exported.

    Lists every :class:`~sx.model.BranchMode` with a one-line description of
    what it would include, in a :class:`BranchModeList`. Enter (or a click)
    exports in the highlighted mode, as does the Export button; Escape cancels
    the export.

    The screen dismisses with the chosen mode, or ``None`` if canceled.

    Args:
        descriptions: A one-line description of each mode offered, in order.
        default: The mode highlighted when the dialog opens — the one the
            transcript viewer is showing.
    """

    CSS = """
    BranchModeScreen {
        align: center middle;
    }
    #dialog {
        width: 84;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #heading { text-style: bold; color: $accent; }
    #modes { margin: 1 0; }
    #buttons { height: auto; align-horizontal: right; }
    Button { margin-left: 2; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, *, descriptions: dict[BranchMode, str], default: BranchMode) -> None:
        """Store the modes on offer and the one to start on."""
        super().__init__()
        self._descriptions = dict(descriptions)
        self._modes = list(self._descriptions)
        self._default = default if default in self._descriptions else self._modes[0]

    def compose(self) -> ComposeResult:
        """Build the dialog widgets."""
        with Container(id="dialog"):
            yield Label("This session branched — what should the export include?", id="heading")
            yield BranchModeList(self._descriptions, self._default, id="modes")
            with Horizontal(id="buttons"):
                yield Button("Cancel", variant="primary", id="cancel")
                yield Button("Export", variant="success", id="confirm")

    def on_mount(self) -> None:
        """Focus the list, which starts on the viewer's mode, for Enter."""
        self.query_one("#modes", BranchModeList).focus()

    def _highlighted_mode(self) -> BranchMode:
        """Return the mode on the highlighted row."""
        return self.query_one("#modes", BranchModeList).chosen

    @on(OptionList.OptionSelected, "#modes")
    def _on_selected(self, event: OptionList.OptionSelected) -> None:
        """Export in the mode just chosen."""
        self.dismiss(self._modes[event.option_index])

    @on(Button.Pressed, "#confirm")
    def _on_confirm(self, event: Button.Pressed) -> None:
        """Export in the highlighted mode."""
        self.dismiss(self._highlighted_mode())

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self, event: Button.Pressed) -> None:
        """Dismiss with no result, canceling the export."""
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Escape cancels the export."""
        self.dismiss(None)


class OrphanScreen(Screen):
    """Full-screen list of orphaned artifacts, with cleanup actions.

    Args:
        orphans: The orphans to display.
        on_delete: Callback invoked with a single :class:`Orphan` to delete it,
            and ``refused=`` the paths its confirmation showed as refused, with
            why; returns a :class:`~sx.model.DeleteResult` that counts those
            among its refusals.
        on_delete_all: The same, invoked for each orphan when every listed one
            is deleted.
        confirm: Async callable the screen awaits to confirm deleting a list
            of orphans (the app supplies the modal). It returns each one
            narrowed to the paths the user was shown as removable, paired with
            the paths shown as refused and why, or ``None`` if the user
            canceled.
    """

    CSS = """
    #orphan-table { height: 1fr; }
    #orphan-summary { height: auto; padding: 0 1; color: $text-muted; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
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

        An orphan stays in the list while anything of it stays on disk:
        removing the row would tell the user it is gone. One refused in part is
        narrowed to what stayed, and the toast says what that is and why. What
        the confirmation showed as refused goes to the delete with its reason,
        so the op-log and the toast name it too — even when nothing else is
        left to delete.
        """
        confirmed = await self._confirm([orphan])
        if not confirmed:
            return
        target, refused = confirmed[0]
        result = self._on_delete(target, refused=refused)
        if result.failed:
            reason = next(iter(result.refused.values()), "unknown reason")
            self.app.notify(
                f"Not deleted — {toast_safe(reason)}", severity="error", timeout=10
            )
            return
        if result.refused:
            self._orphans[self._orphans.index(orphan)] = _what_stayed(orphan, result)
            self._reload_table()
            done: list[str] = []
            if result.removed:
                done.append(f"{len(result.removed)} path(s) · {human_size(result.freed_bytes)}")
            if result.note:
                done.append(toast_safe(result.note))
            self.app.notify(
                f"Partly deleted — {_stayed_notice(result.refused)}. "
                f"Removed {' · '.join(done)}.",
                severity="warning",
                timeout=10,
            )
            return
        self._orphans.remove(orphan)
        self._reload_table()
        self.app.notify(f"Deleted orphan · {toast_safe(result.summary())}")

    @work
    async def _run_delete_all(self) -> None:
        """Confirm and delete all orphans, then refresh the table.

        Only the orphans removed whole leave the list. One refused whole stays
        as it was, and one refused in part stays narrowed to what is left; the
        toast counts every path that stayed, those of orphans deleted in part
        included, and says why.
        """
        listed = list(self._orphans)
        confirmed = await self._confirm(listed)
        if not confirmed:
            return
        freed = 0
        deleted = 0
        partly = 0
        refused: list[str] = []
        stayed: dict[Path, str] = {}
        for orphan, (target, shown) in zip(listed, confirmed, strict=True):
            result = self._on_delete_all(target, refused=shown)
            stayed.update(result.refused)
            if result.failed:
                refused.append(next(iter(result.refused.values()), "refused"))
                continue
            freed += result.freed_bytes
            if result.refused:
                partly += 1
                self._orphans[self._orphans.index(orphan)] = _what_stayed(orphan, result)
                continue
            deleted += 1
            self._orphans.remove(orphan)
        self._reload_table()

        if not deleted and not partly:
            self.app.notify(
                f"Nothing was deleted — {len(refused)} refused "
                f"({toast_safe(refused[0]) if refused else 'unknown reason'})",
                severity="error",
                timeout=10,
            )
        elif stayed:
            if not partly:
                done = f"Deleted {deleted} orphan(s)"
            elif deleted:
                done = f"Deleted {deleted} orphan(s) whole and {partly} in part"
            else:
                done = f"Deleted {partly} orphan(s) in part"
            if refused:
                done += f" · {len(refused)} refused"
            self.app.notify(
                f"Partly deleted — {_stayed_notice(stayed)}. "
                f"{done} · freed {human_size(freed)}.",
                severity="warning",
                timeout=10,
            )
        else:
            self.app.notify(f"Deleted {deleted} orphan(s) · freed {human_size(freed)}")


class MoveScreen(_Dialog):
    """Re-point a project's sessions at a new directory.

    The screen owns no logic of its own: it collects a destination and hands it
    to ``preview``, which returns the rendered plan along with whether the move
    can proceed and whether it needs a typed confirmation. Nothing is previewed
    until a destination is submitted, because planning walks the session stores.

    Dismisses with ``None`` if canceled, or ``{"new": str, "include_config":
    bool, "typed_phrase": bool}`` if confirmed — ``typed_phrase`` saying
    whether the plan confirmed needed, and got, the typed phrase.

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

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

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
            # The old path was read from a session file. As Text it is not
            # markup, and sanitized it cannot drive the terminal; it is only
            # pre-filled when that changes nothing, because a sanitized copy
            # left in place would be a different, unintended destination.
            safe_old = sanitize_text(self._old)
            yield Label(Text(f"from  {safe_old}"), id="from")
            yield Input(
                value=self._old if safe_old == self._old else "",
                placeholder="new absolute path…",
                id="destination",
            )
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
        self.dismiss(
            {
                "new": self._previewed,
                "include_config": self._include_config(),
                "typed_phrase": self._needs_phrase,
            }
        )

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

    BINDINGS: ClassVar[list[BindingType]] = [
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
            # Sanitized too, since a Text reaches the terminal as it is.
            table.add_row(
                Text(sanitize_text(memory.project_name)),
                Text(sanitize_text(memory.name)),
                Text(sanitize_text(memory.kind)),
                Text(sanitize_text(memory.origin_label)),
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
            body.update(
                Text(sanitize_text(f"Could not read {memory.path}: {exc}"), style="red")
            )
            return
        # Untrusted content: a memory is model-written and may quote anything.
        rendered = Text()
        rendered.append(f"{sanitize_text(str(memory.path))}\n", style="dim")
        if memory.description:
            rendered.append(f"{sanitize_text(memory.description)}\n", style="italic")
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
            self.app.notify(f"Export failed: {toast_safe(repr(exc))}", severity="error")
            return
        self.app.notify(f"Exported → {toast_safe(path)}")

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
                f"Nothing was deleted — "
                f"{toast_safe(refused[0]) if refused else 'unknown reason'}",
                severity="error",
                timeout=10,
            )
        elif refused:
            self.app.notify(
                f"Deleted {removed} · freed {human_size(freed)} · "
                f"{len(refused)} refused ({toast_safe(refused[0])})",
                severity="warning",
                timeout=10,
            )
        else:
            self.app.notify(f"Deleted {removed} memory file(s) · freed {human_size(freed)}")
