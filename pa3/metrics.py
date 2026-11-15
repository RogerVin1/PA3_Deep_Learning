"""Métricas (F1 macro é a principal do PA) e gráficos da Parte 0."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def summarize(y_true, y_pred, num_classes: int) -> dict:
    """F1 macro sobre as K classes (classe sem acerto conta 0, não some da média)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(num_classes))
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_per_class": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "support": np.bincount(y_true, minlength=num_classes).tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def memorization_summary(history: list[dict], train_acc_thr: float = 0.95) -> dict:
    """Responde 'em que época começa a decorar?' a partir da curva.

    - epoch_min_val_loss: depois dela a perda de validação sobe enquanto a de treino cai.
    - epoch_train_acc_thr: primeira época em que a acurácia (com aumentação) passa do limiar.
    """
    val_loss = [h["val_loss"] for h in history]
    hit = next((h["epoch"] for h in history if h["train_acc"] >= train_acc_thr), None)
    return {
        "epoch_min_val_loss": history[int(np.argmin(val_loss))]["epoch"],
        "epoch_best_val_f1": history[int(np.argmax([h["val_f1"] for h in history]))]["epoch"],
        "epoch_train_acc_thr": hit,
        "train_acc_thr": train_acc_thr,
    }


def plot_history(history: list[dict], path: str | Path, title: str = "", best_epoch: int | None = None) -> None:
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="treino")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validação")
    ax[0].set(xlabel="época", ylabel="perda (cross-entropy)", title="Perda")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="acurácia treino")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="acurácia val")
    ax[1].plot(ep, [h["val_f1"] for h in history], label="F1 macro val", linestyle="--")
    ax[1].set(xlabel="época", ylabel="valor", title="Acurácia e F1", ylim=(0, 1.02))
    for a in ax:
        if best_epoch is not None:
            a.axvline(best_epoch, color="gray", linestyle=":", label=f"época escolhida ({best_epoch})")
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, class_names: list[str], path: str | Path, title: str = "") -> None:
    """Matriz de confusão normalizada por linha (recall), com a contagem em cada célula."""
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    k = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * k + 3), max(5, 0.55 * k + 2)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(k), yticks=range(k), xlabel="previsto", ylabel="verdadeiro", title=title)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    for i in range(k):
        for j in range(k):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fração da classe verdadeira")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
