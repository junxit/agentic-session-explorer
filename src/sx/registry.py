"""Adapter registry: the single place that knows every harness adapter.

Adapters are imported defensively — if one module is missing or raises on
import/instantiation, it is recorded as a load error and skipped rather than
taking down the whole tool. This keeps ``sx`` usable even while individual
adapters are still being built or a harness changes its on-disk format.
"""

from __future__ import annotations

import importlib

from sx.adapters.base import HarnessAdapter

# (module path, class name) for the fully-supported adapters.
_REAL_ADAPTERS: list[tuple[str, str]] = [
    ("sx.adapters.claude", "ClaudeAdapter"),
    ("sx.adapters.codex", "CodexAdapter"),
    ("sx.adapters.gemini", "GeminiAdapter"),
]


def build_registry() -> tuple[list[HarnessAdapter], list[tuple[str, str]]]:
    """Instantiate every known adapter, isolating failures.

    Returns:
        A tuple ``(adapters, errors)`` where ``adapters`` is the list of
        successfully instantiated adapters (real first, then dormant), and
        ``errors`` is a list of ``(adapter_name, error_repr)`` for any that
        could not be loaded.
    """
    adapters: list[HarnessAdapter] = []
    errors: list[tuple[str, str]] = []

    for module_path, class_name in _REAL_ADAPTERS:
        try:
            module = importlib.import_module(module_path)
            adapter_cls = getattr(module, class_name)
            adapters.append(adapter_cls())
        except Exception as exc:  # noqa: BLE001 - isolate any adapter failure
            errors.append((class_name, repr(exc)))

    try:
        from sx.adapters.dormant import DORMANT_ADAPTERS

        for adapter_cls in DORMANT_ADAPTERS:
            try:
                adapters.append(adapter_cls())
            except Exception as exc:  # noqa: BLE001
                errors.append((adapter_cls.__name__, repr(exc)))
    except Exception as exc:  # noqa: BLE001
        errors.append(("dormant", repr(exc)))

    return adapters, errors


def all_adapters() -> list[HarnessAdapter]:
    """Return every adapter that loaded successfully."""
    return build_registry()[0]


def available_adapters() -> list[HarnessAdapter]:
    """Return adapters whose harness appears installed on this machine."""
    return [adapter for adapter in all_adapters() if adapter.available()]
