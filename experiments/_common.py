"""Piccole utility condivise dagli script di esperimento."""

from __future__ import annotations


def section(title: str) -> None:
    """Stampa un titolo ben visibile, utile quando si proietta il terminale in aula."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
