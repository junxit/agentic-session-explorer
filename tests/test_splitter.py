"""The divider between the tree and the transcript resizes the tree by mouse drag."""

from __future__ import annotations

import pytest

from sx.tui.app import SxApp, Splitter
from sx.util import load_left_pane_width


@pytest.mark.asyncio
async def test_dragging_the_splitter_resizes_the_left_pane():
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        left = app.query_one("#left")
        x = left.region.width
        await pilot.mouse_down(Splitter, offset=(0, 5))
        await pilot.hover(None, offset=(x + 20, 5))
        await pilot.pause()
        assert left.region.width == x + 20
        await pilot.hover(None, offset=(2, 5))
        await pilot.pause()
        assert left.region.width == Splitter.MIN_PANE
        await pilot.mouse_up(None, offset=(2, 5))
        assert not app.query_one(Splitter).has_class("dragging")


@pytest.mark.asyncio
async def test_width_is_saved_on_release_and_restored_on_next_start():
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        x = app.query_one("#left").region.width
        await pilot.mouse_down(Splitter, offset=(0, 5))
        await pilot.hover(None, offset=(x + 10, 5))
        await pilot.mouse_up(None, offset=(x + 10, 5))
        assert load_left_pane_width() == x + 10

    async with SxApp().run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#left").region.width == x + 10
