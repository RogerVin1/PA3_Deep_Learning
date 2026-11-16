from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pa3 import ORCAMENTO_PADRAO  # noqa: E402
from pa3.calibrate import calibrate_teacher  # noqa: E402
from pa3.data import find_root, load_split, load_image, read_classes  # noqa: E402
from pa3.teacher import zero_shot_logits  # noqa: E402


def describe(items, label=""):
    """Uma linha de estatísticas para uma lista de Item."""
    w, h, bright, gray = [], [], [], []
    for it in items:
        with Image.open(it.path) as im:
            w.append(im.width), h.append(im.height)
        a = np.asarray(load_image(it.path, 64), dtype=np.float32)
        bright.append(a.mean())
        gray.append(float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() < 6.0))
    sizes = pd.Series([f"{a}x{b}" for a, b in zip(w, h)]).value_counts(normalize=True)
    return {"grupo": label, "n": len(items), "brilho_medio": round(float(np.mean(bright)), 1),
            "cinza(%)": round(100 * float(np.mean(gray))), "tamanho_mais_comum": sizes.index[0],
            "tamanho(%)": round(100 * float(sizes.iloc[0])), "n_tamanhos": len(sizes)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.environ.get("PA3_DATA", "data/pa3_dados"))
    p.add_argument("--classe", default="eastern_cottontail")
    p.add_argument("--budget", type=int, default=ORCAMENTO_PADRAO)
    p.add_argument("--teacher", default="bioclip2")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--max-por-grupo", type=int, default=400)
    p.add_argument("--out", default="results/part1/diagnostico")
    args = p.parse_args(argv)

    root = find_root(args.data)
    classes = read_classes(root)
    K, k = len(classes), classes.index(args.classe)
    train, labeled = load_split(root, "train", classes), load_split(root, f"rotulos_{args.budget}", classes)
    val, test = load_split(root, "val", classes), load_split(root, "test_public", classes)

    c = torch.load(Path(args.ckpt_dir) / f"teacher_{args.teacher}.pt", weights_only=True)
    zs = zero_shot_logits(c["train"].float(), c["text"].float(), c["scale"])
    pos = {Path(i.image_id).stem: n for n, i in enumerate(train)}
    lab_idx = np.array([pos[Path(i.image_id).stem] for i in labeled])
    cal, _ = calibrate_teacher(zs[lab_idx], torch.as_tensor([i.label for i in labeled]), K)
    pred = cal.apply(zs).argmax(1).numpy()
    for n, i in zip(lab_idx, labeled):
        pred[n] = i.label  # rótulo verdadeiro onde existe

    rng = np.random.default_rng(0)

    def sample(items):
        items = list(items)
        return [items[j] for j in rng.permutation(len(items))[: args.max_por_grupo]]

    grupos = {
        f"train ({args.classe}, pseudo)": sample(it for it, q in zip(train, pred) if q == k),
        f"val ({args.classe}, real)": [it for it in val if it.label == k],
        f"test ({args.classe}, real)": [it for it in test if it.label == k],
        "train (outras classes)": sample(it for it, q in zip(train, pred) if q != k),
        "test (outras classes)": sample(it for it in test if it.label != k),
    }
    df = pd.DataFrame([describe(v, g) for g, v in grupos.items()])
    print(df.to_string(index=False))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / f"{args.classe}.csv", index=False)
    return df


if __name__ == "__main__":
    main()
