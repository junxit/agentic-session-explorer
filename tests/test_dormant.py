"""Tests for sx.adapters.dormant: every dormant adapter is inert."""

from __future__ import annotations

import pytest

from sx.adapters.dormant import DORMANT_ADAPTERS
from sx.model import Capability


@pytest.mark.parametrize("adapter_cls", DORMANT_ADAPTERS)
def test_dormant_adapter_is_inert(adapter_cls):
    """Each dormant adapter has NONE capabilities and yields no data."""
    adapter = adapter_cls()
    assert adapter.capabilities == Capability.NONE
    assert Capability.BROWSE not in adapter.capabilities
    assert list(adapter.discover()) == []
    assert adapter.load(None) == []
