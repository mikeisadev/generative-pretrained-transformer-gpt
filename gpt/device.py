"""Scelta automatica del dispositivo di calcolo.

Ordine di preferenza: GPU NVIDIA (CUDA) → Apple Silicon (MPS) → CPU.
Lo stesso codice gira quindi su Linux con o senza GPU e su qualsiasi Mac.
"""

from __future__ import annotations

import torch

_VALID_CHOICES = ("auto", "cpu", "cuda", "mps")


def resolve_device(preferred: str = "auto") -> torch.device:
    """Restituisce il device richiesto, oppure il migliore disponibile con ``"auto"``.

    Se l'utente chiede esplicitamente un device che non c'è, alziamo un errore
    chiaro invece di ricadere in silenzio sulla CPU: meglio saperlo subito.
    """
    choice = preferred.lower()
    if choice not in _VALID_CHOICES:
        raise ValueError(f"device sconosciuto '{preferred}', scegli tra {_VALID_CHOICES}")

    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available()

    if choice == "auto":
        if cuda_ok:
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA richiesto ma non disponibile su questa macchina")
    if choice == "mps" and not mps_ok:
        raise RuntimeError("MPS (Apple Silicon) richiesto ma non disponibile su questa macchina")
    return torch.device(choice)
