"""Utilitários: semente, dispositivo, JSON e tabela da curva de orçamento."""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CURVA_COLS = ["metodo", "orcamento", "seed", "f1_macro", "acuracia"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit() -> str:
    """Hash curto do commit atual, para amarrar cada resultado ao código que o gerou."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=5, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or "sem-git"
    except (OSError, subprocess.SubprocessError):
        return "sem-git"


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def append_curve(csv_path: str | Path, metodo: str, orcamento: int, seed: int,
                 f1_macro: float, acuracia: float) -> None:
    """Acrescenta/atualiza uma linha de curva_orcamento.csv (chave: metodo, orcamento, seed)."""
    csv_path = Path(csv_path)
    row = pd.DataFrame([[metodo, orcamento, seed, round(f1_macro, 4), round(acuracia, 4)]], columns=CURVA_COLS)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        keep = ~((old.metodo == metodo) & (old.orcamento == orcamento) & (old.seed == seed))
        row = pd.concat([old[keep], row], ignore_index=True)
    row = row.sort_values(["metodo", "orcamento", "seed"]).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(csv_path, index=False)
