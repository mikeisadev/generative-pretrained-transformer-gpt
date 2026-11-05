"""LayerNorm scritta a mano, per vedere cosa fa ``nn.LayerNorm``.

Per OGNI token (riga) calcola media e varianza sulle sue C feature,
normalizza a media 0 / varianza 1 e poi applica γ (scala) e β (spostamento) appresi.

Differenze rispetto al ``LayerNorm1d`` del notebook (corrette qui):
- il notebook usava ``x.var()`` con la correzione di Bessel (divide per C-1);
  ``nn.LayerNorm`` divide per C. Ora i due risultati coincidono;
- i commenti parlavano di "batch mean": era un residuo di BatchNorm, la media
  è sulle feature del singolo esempio;
- il parametro ``momentum`` non era usato ed è stato rimosso.
"""

from __future__ import annotations

import torch


class LayerNorm1d:
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.gamma = torch.ones(dim)  # appreso nel modello vero: parte da 1 (nessuna scala)
        self.beta = torch.zeros(dim)  # appreso nel modello vero: parte da 0 (nessuno spostamento)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)  # media delle feature di ogni esempio
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # varianza "di popolazione", come nn.LayerNorm
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * xhat + self.beta

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]
