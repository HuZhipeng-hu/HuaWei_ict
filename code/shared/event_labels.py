"""Shared helpers for the event-onset CONTINUE/RELAX label contract."""

from __future__ import annotations

from typing import Iterable, Mapping

PUBLIC_CONTINUE_LABEL = "CONTINUE"
LEGACY_CONTINUE_LABEL = "RELAX"
CONTINUE_LABEL_ALIASES = {PUBLIC_CONTINUE_LABEL, LEGACY_CONTINUE_LABEL}


def is_continue_label(value: str | None) -> bool:
    return str(value or "").strip().upper() in CONTINUE_LABEL_ALIASES


def normalize_event_label_input(value: str | None) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in CONTINUE_LABEL_ALIASES:
        return LEGACY_CONTINUE_LABEL
    return normalized


def public_event_label(value: str | None) -> str:
    raw = str(value or "").strip()
    normalized = normalize_event_label_input(value)
    if normalized == LEGACY_CONTINUE_LABEL:
        return PUBLIC_CONTINUE_LABEL
    return raw


def public_event_labels(values: Iterable[str | None]) -> list[str]:
    return [public_event_label(value) for value in values]


def public_event_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    return {
        public_event_label(key): public_event_label(value)
        for key, value in mapping.items()
    }
