"""Aluno da Trilha C: ResNet-18 treinada nos pseudo-rótulos do professor + os rótulos verdadeiros.

Cada imagem do train tem um alvo (distribuição sobre as K classes) e um peso:
  - sem rótulo verdadeiro: alvo do professor, duro (one-hot do argmax) ou suave (softmax com
    temperatura), peso 1;
  - com rótulo verdadeiro (os do orçamento): alvo one-hot verdadeiro, peso `true_weight`.
A época é escolhida no val (F1 macro), como no piso. O val nunca passa pelo professor.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .data import PA3Dataset, make_loader
from .train import FitConfig, _param_groups, evaluate
from .utils import set_seed


class SoftDataset(PA3Dataset):
    """PA3Dataset que devolve (imagem, alvo K-dim, peso) em vez de (imagem, rótulo)."""

    def __init__(self, items, size, transform, targets, weights, **kw):
        super().__init__(items, size, transform, **kw)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __getitem__(self, i):
        img, _ = super().__getitem__(i)
        return img, self.targets[i], self.weights[i]


def build_targets(teacher_logits: torch.Tensor, mode: str, temperature: float, labeled_idx: np.ndarray,
                  labeled_y: np.ndarray, true_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Alvos (N, K) e pesos (N,) do aluno. `mode`: 'hard' ou 'soft'."""
    N, K = teacher_logits.shape
    if mode == "hard":
        t = F.one_hot(teacher_logits.argmax(1), K).float()
    elif mode == "soft":
        t = (teacher_logits / temperature).softmax(1)
    else:
        raise ValueError(f"mode desconhecido: {mode} (use 'hard' ou 'soft')")
    w = torch.ones(N)
    idx = torch.as_tensor(labeled_idx, dtype=torch.long)
    t[idx] = F.one_hot(torch.as_tensor(labeled_y, dtype=torch.long), K).float()
    w[idx] = true_weight
    return t, w


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_sample = -(targets * F.log_softmax(logits, dim=1)).sum(1)
    return (weights * per_sample).sum() / weights.sum()


def fit_soft(model, train_ds: SoftDataset, val_ds, cfg: FitConfig, device, num_classes: int, seed: int = 0,
             verbose: bool = True) -> tuple[list[dict], int]:
    """Como pa3.train.fit, mas com alvos suaves e pesos por imagem. Escolhe a época pelo F1 macro no val.

    `train_acc` aqui é a concordância com o argmax do alvo (não é acurácia verdadeira).
    """
    set_seed(seed)
    model.to(device)
    loader = make_loader(train_ds, cfg.batch_size, shuffle=True, seed=seed, num_workers=cfg.num_workers)
    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(loader)
    warm = max(1, int(cfg.warmup_frac * total))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    history, best_key, best_state, best_epoch = [], None, None, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = agree = seen = 0
        for x, t, w in loader:
            x, t, w = x.to(device), t.to(device), w.to(device)
            logits = model(x)
            loss = soft_cross_entropy(logits, t, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(x)
            agree += (logits.argmax(1) == t.argmax(1)).sum().item()
            seen += len(x)
        val = evaluate(model, val_ds, device, num_classes, num_workers=cfg.num_workers)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": agree / seen,
               "val_loss": val["loss"], "val_acc": val["acc"], "val_f1": val["f1_macro"]}
        history.append(row)
        key = (val["f1_macro"], -val["loss"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  época {epoch:3d} | treino loss {row['train_loss']:.3f} concord. {row['train_acc']:.3f}"
                  f" | val loss {row['val_loss']:.3f} acc {row['val_acc']:.3f} F1 {row['val_f1']:.3f}")

    model.load_state_dict(best_state)
    return history, best_epoch
