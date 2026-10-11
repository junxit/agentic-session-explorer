"""Harness rows can be hidden from the tree through the palette submenu."""

from __future__ import annotations

import pytest
from textual.widgets import Tree

from sx.model import Capability
from sx.tui import app as app_module
from sx.tui.app import HarnessMenu, SxApp
from sx.util import load_hidden_harnesses


class FakeAdapter:
    """Stand-in harness: no disk access, no sessions."""

    capabilities = Capability.BROWSE

    def __init__(self, name: str, display: str, installed: bool = True) -> None:
        self.name = name
        self.display = display
        self._installed = installed

    def available(self) -> bool:
        return self._installed

    def discover(self) -> list:
        return []


@pytest.fixture(autouse=True)
def _fake_harnesses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the registry so the tests never depend on what is installed."""
    adapters = [
        FakeAdapter("alpha", "Alpha"),
        FakeAdapter("beta", "Beta"),
        FakeAdapter("gamma", "Gamma", installed=False),
    ]
    monkeypatch.setattr(app_module, "build_registry", lambda: (adapters, []))


def _labels(app: SxApp) -> list[str]:
    return [str(n.label) for n in app.query_one("#tree", Tree).root.children]


def _toggle(app: SxApp, display: str):
    return next(t for label, _, t in HarnessMenu(app.screen)._entries() if display in label)


@pytest.mark.asyncio
async def test_palette_offers_the_harness_submenu():
    app = SxApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert "Harnesses" in [c.title for c in app.get_system_commands(app.screen)]


@pytest.mark.asyncio
async def test_submenu_lists_every_harness_with_its_state():
    app = SxApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        labels = [label for label, _, _ in HarnessMenu(app.screen)._entries()]
        assert labels == ["✓ Alpha", "✓ Beta", "✓ Gamma (not installed)"]


@pytest.mark.asyncio
async def test_toggle_hides_then_shows_a_harness_and_persists():
    app = SxApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert any("Beta" in x for x in _labels(app))

        _toggle(app, "Beta")()
        assert load_hidden_harnesses() == {"beta"}
        assert not any("Beta" in x for x in _labels(app))
        assert any("Alpha" in x for x in _labels(app))
        assert next(l for l, _, _ in HarnessMenu(app.screen)._entries() if "Beta" in l).startswith("✗")

        _toggle(app, "Beta")()
        assert load_hidden_harnesses() == set()
        assert any("Beta" in x for x in _labels(app))


@pytest.mark.asyncio
async def test_hidden_choice_is_restored_on_next_start():
    first = SxApp()
    async with first.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        _toggle(first, "Gamma")()
    second = SxApp()
    async with second.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert not any("Gamma" in x for x in _labels(second))
