"""Treino supervisionado com seleção de época no val (nunca no test_public)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import make_loader
from .metrics import summarize
from .utils import set_seed


@dataclass
class FitConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3               # taxa da cabeça (fc)
    backbone_lr_mult: float = 0.1  # corpo aprende mais devagar (1.0 = igual à cabeça)
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    warmup_frac: float = 0.1
    num_workers: int = 0


@torch.no_grad()
def predict_logits(model, ds: Dataset, device, batch_size: int = 128, num_workers: int = 0) -> torch.Tensor:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return torch.cat([model(x.to(device)).float().cpu() for x, _ in loader])


def evaluate(model, ds, device, num_classes: int, batch_size: int = 128, num_workers: int = 0) -> dict:
    """Perda, acurácia, F1 macro, F1 por classe e matriz de confusão num conjunto rotulado."""
    logits = predict_logits(model, ds, device, batch_size, num_workers)
    y = ds.labels
    out = summarize(y.numpy(), logits.argmax(1).numpy(), num_classes)
    out["loss"] = float(F.cross_entropy(logits, y))
    out["logits"] = logits
    return out


def _param_groups(model, cfg: FitConfig):
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    return [{"params": body, "lr": cfg.lr * cfg.backbone_lr_mult},
            {"params": head, "lr": cfg.lr}]


def fit(model, train_ds: Dataset, val_ds: Dataset, cfg: FitConfig, device, num_classes: int,
        seed: int = 0, verbose: bool = True) -> tuple[list[dict], int]:
    """Treina `cfg.epochs` épocas e deixa em `model` os pesos da melhor época no val.

    Critério: maior F1 macro no val (desempate: menor perda no val). Devolve
    (histórico por época, melhor época). O test_public NÃO é tocado aqui.
    """
    set_seed(seed)
    model.to(device)
    loader = make_loader(train_ds, cfg.batch_size, shuffle=True, seed=seed, num_workers=cfg.num_workers)
    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(loader)
    warm = max(1, int(cfg.warmup_frac * total))

    def factor(step: int) -> float:  # aquecimento linear + cosseno até 0
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    history, best_key, best_state, best_epoch = [], None, None, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = correct = seen = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        val = evaluate(model, val_ds, device, num_classes, num_workers=cfg.num_workers)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
               "val_loss": val["loss"], "val_acc": val["acc"], "val_f1": val["f1_macro"]}
        history.append(row)
        key = (val["f1_macro"], -val["loss"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  época {epoch:3d} | treino loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}"
                  f" | val loss {row['val_loss']:.3f} acc {row['val_acc']:.3f} F1 {row['val_f1']:.3f}")

    model.load_state_dict(best_state)
    return history, best_epoch
