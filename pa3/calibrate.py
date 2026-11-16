from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0)
LAMBDA_DEFAULT = 0.1  # usado quando não há 2 rótulos por classe para validar (orçamento 1)


class BiasTemperature:
    """logits' = logits / T + b, ajustado por máxima verossimilhança com L2 em b."""

    def __init__(self, num_classes: int):
        self.K = num_classes
        self.b = torch.zeros(num_classes)
        self.log_t = torch.zeros(())

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp() + self.b

    def fit(self, logits: torch.Tensor, y: torch.Tensor, lam: float, max_iter: int = 200) -> "BiasTemperature":
        b = torch.zeros(self.K, requires_grad=True)
        lt = torch.zeros((), requires_grad=True)
        opt = torch.optim.LBFGS([b, lt], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / lt.clamp(-2, 3).exp() + b, y) + lam * (b ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.b, self.log_t = b.detach(), lt.detach().clamp(-2, 3)
        return self


def choose_lambda(logits: torch.Tensor, y: torch.Tensor, K: int, grid=LAMBDA_GRID, max_folds: int = 5) -> float:
    """λ pela perda de validação cruzada estratificada dentro dos rótulos do orçamento."""
    y_np = y.numpy()
    counts = np.bincount(y_np, minlength=K)
    per_class = counts[counts > 0].min()
    if per_class < 2:
        return LAMBDA_DEFAULT
    nf = int(min(per_class, max_folds))
    fold = np.zeros(len(y_np), dtype=int)
    for k in range(K):
        idx = np.where(y_np == k)[0]
        fold[idx] = np.arange(len(idx)) % nf  # cada dobra recebe ~1 imagem de cada classe
    best = None
    for lam in grid:
        loss = 0.0
        for f in range(nf):
            te = torch.from_numpy(fold == f)
            cal = BiasTemperature(K).fit(logits[~te], y[~te], lam)
            loss += F.cross_entropy(cal.apply(logits[te]), y[te], reduction="sum").item()
        if best is None or loss < best[0]:
            best = (loss, lam)
    return best[1]


def calibrate_teacher(logits_labeled: torch.Tensor, y_labeled: torch.Tensor, K: int) -> tuple[BiasTemperature, float]:
    """Ajusta o calibrador nos rótulos do orçamento e devolve (calibrador, λ escolhido)."""
    lam = choose_lambda(logits_labeled, y_labeled, K)
    return BiasTemperature(K).fit(logits_labeled, y_labeled, lam), lam
