"""Parte 1 — onde o professor (zero-shot) erra de forma sistemática? (noite, animais pequenos, classes parecidas)

Usa só o test_public e os embeddings já salvos pelo part1_teacher.py. Não treina nada nem escolhe nada.
  - noite: acurácia em imagens em tons de cinza (infravermelho) x coloridas, e por faixa de brilho;
  - pequenos: grupo de classes de corpo pequeno, definido a priori (lista abaixo), x o resto;
  - parecidas: os pares (verdadeira -> prevista) mais confundidos.

    python scripts/part1_erros_professor.py --data /caminho/pa3_dados --teacher bioclip2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from pa3.data import find_root, load_image, load_split, read_classes  # noqa: E402
from pa3.teacher import zero_shot_logits  # noqa: E402

# Definido a priori (antes de olhar os resultados): classes de corpo pequeno, difíceis de ver na câmera.
PEQUENOS = ("bird", "chicken", "eastern_chipmunk", "eastern_cottontail", "eastern_fox_squirrel",
            "eastern_gray_squirrel", "woodchuck", "domestic_cat")


def foto_stats(path: Path) -> tuple[float, bool]:
    """(brilho médio 0-255, imagem em tons de cinza?) numa versão 64x64."""
    a = np.asarray(load_image(path, 64), dtype=np.float32)
    cinza = np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() < 6.0
    return float(a.mean()), bool(cinza)


def grupo(df: pd.DataFrame, nome: str) -> dict:
    return {"grupo": nome, "n": len(df), "acuracia": round(float(df.acerto.mean()), 3) if len(df) else float("nan"),
            "previsto_woodchuck(%)": round(100 * float((df.pred == "woodchuck").mean()), 1) if len(df) else float("nan")}


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.environ.get("PA3_DATA", "data/pa3_dados"))
    p.add_argument("--teacher", default="bioclip2")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--out", default="results/part1")
    args = p.parse_args(argv)

    root = find_root(args.data)
    classes = read_classes(root)
    K = len(classes)
    test = load_split(root, "test_public", classes)
    c = torch.load(Path(args.ckpt_dir) / f"teacher_{args.teacher}.pt", weights_only=True)
    pred = zero_shot_logits(c["test"].float(), c["text"].float(), c["scale"]).argmax(1).numpy()
    y = np.array([i.label for i in test])
    assert len(pred) == len(y), "embeddings do professor não batem com o test_public atual"

    st = [foto_stats(i.path) for i in test]
    df = pd.DataFrame({"y": [classes[k] for k in y], "pred": [classes[k] for k in pred], "acerto": pred == y,
                       "brilho": [s[0] for s in st], "cinza": [s[1] for s in st]})
    df["pequeno"] = df.y.isin(PEQUENOS)

    linhas = [grupo(df, "todas")]
    linhas += [grupo(df[df.cinza], "noite (cinza)"), grupo(df[~df.cinza], "dia (colorida)")]
    q = df.brilho.quantile([1 / 3, 2 / 3]).tolist()
    linhas += [grupo(df[df.brilho <= q[0]], "terço mais escuro"),
               grupo(df[(df.brilho > q[0]) & (df.brilho <= q[1])], "terço médio"),
               grupo(df[df.brilho > q[1]], "terço mais claro")]
    linhas += [grupo(df[df.pequeno], "classes pequenas"), grupo(df[~df.pequeno], "classes grandes")]
    tab = pd.DataFrame(linhas)
    print("Professor zero-shot no test_public — acurácia por grupo")
    print(tab.to_string(index=False))

    cm = confusion_matrix(y, pred, labels=range(K))
    off = [(classes[i], classes[j], int(cm[i, j]), round(float(cm[i, j] / cm[i].sum()), 2))
           for i in range(K) for j in range(K) if i != j and cm[i, j] > 0]
    top = pd.DataFrame(sorted(off, key=lambda r: -r[2])[:10], columns=["verdadeira", "prevista", "n", "fracao_da_classe"])
    print("\nPares mais confundidos (verdadeira -> prevista)")
    print(top.to_string(index=False))

    out = Path(args.out) / args.teacher
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "erros_por_grupo.csv", index=False)
    top.to_csv(out / "pares_confundidos.csv", index=False)
    return tab


if __name__ == "__main__":
    main()
