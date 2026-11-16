"""Parte 1, Trilha C — passo 2: o aluno (ResNet-18) aprende dos pseudo-rótulos do professor.

Pré-requisito: scripts/part1_teacher.py já rodou (usa os embeddings salvos em checkpoints/).
O aluno treina em todo o train: pseudo-rótulos do professor + rótulos verdadeiros do orçamento.
A época é escolhida no val; o test_public é avaliado uma única vez, no fim.

    python scripts/part1_student.py --data /caminho/pa3_dados --budget 5 --seed 0 --targets hard
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pa3 import F1_TETO_MONITOR, ORCAMENTO_PADRAO, ORCAMENTOS  # noqa: E402
from pa3.data import (check_splits, find_root, ids_have_extension, load_split, make_eval_ds,  # noqa: E402
                      read_classes, train_transform)
from pa3.metrics import memorization_summary, plot_confusion, plot_history  # noqa: E402
from pa3.models import build_model, check_param_budget, save_checkpoint  # noqa: E402
from pa3.student import SoftDataset, build_targets, fit_soft  # noqa: E402
from pa3.teacher import zero_shot_logits  # noqa: E402
from pa3.train import FitConfig, evaluate  # noqa: E402
from pa3.utils import append_curve, get_device, git_commit, save_json  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.environ.get("PA3_DATA", "data/pa3_dados"))
    p.add_argument("--budget", type=int, default=ORCAMENTO_PADRAO, choices=ORCAMENTOS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher", default="bioclip2", help="nome usado em part1_teacher.py")
    p.add_argument("--targets", default="hard", choices=["hard", "soft"], help="pseudo-rótulos duros ou suaves")
    p.add_argument("--temperature", type=float, default=1.0, help="temperatura do softmax (só em --targets soft)")
    p.add_argument("--true-weight", type=float, default=5.0, help="peso das imagens com rótulo verdadeiro")
    p.add_argument("--variant", default=None, help="nome do método em curva_orcamento.csv (padrão: aluno_<targets>)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--load-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3, help="taxa da cabeça; o corpo usa lr*0.1 (igual ao piso)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-pretrained", action="store_true", help="(testes) não baixa pesos do ImageNet")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/part1/aluno")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--curve-csv", default="curva_orcamento.csv")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    variant = args.variant or f"aluno_{args.targets}"
    root = find_root(args.data)
    classes = read_classes(root)
    K = len(classes)
    device = get_device(args.device)

    train, labeled = load_split(root, "train", classes), load_split(root, f"rotulos_{args.budget}", classes)
    val_items, test_items = load_split(root, "val", classes), load_split(root, "test_public", classes)
    check_splits(train, labeled, val_items, test_items)

    # Embeddings do professor (calculados uma vez por part1_teacher.py); ordem = load_split("train").
    cache = Path(args.ckpt_dir) / f"teacher_{args.teacher}.pt"
    if not cache.exists():
        sys.exit(f"Não achei {cache}. Rode antes: python scripts/part1_teacher.py --teacher {args.teacher}")
    c = torch.load(cache, weights_only=True)
    t_logits = zero_shot_logits(c["train"].float(), c["text"].float(), c["scale"])
    assert len(t_logits) == len(train), "embeddings do professor não batem com o train atual"

    pos = {Path(i.image_id).stem: n for n, i in enumerate(train)}
    lab_idx = np.array([pos[Path(i.image_id).stem] for i in labeled])
    lab_y = np.array([i.label for i in labeled])
    targets, weights = build_targets(t_logits, args.targets, args.temperature, lab_idx, lab_y, args.true_weight)
    agree = float((t_logits.argmax(1).numpy() == targets.argmax(1).numpy()).mean())
    print(f"Aluno '{variant}' | alvos {args.targets} | dispositivo {device} | train {len(train)} "
          f"({len(labeled)} com rótulo verdadeiro, peso {args.true_weight}) | alvo = professor em {agree:.1%}")

    train_ds = SoftDataset(train, args.load_size, train_transform(args.img_size), targets, weights)
    val_ds, test_ds = make_eval_ds(val_items, args.img_size), make_eval_ds(test_items, args.img_size)

    model = build_model("resnet18", K, pretrained=not args.no_pretrained)
    n_params = check_param_budget(model)
    cfg = FitConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, backbone_lr_mult=0.1,
                    num_workers=args.workers)
    history, best_epoch = fit_soft(model, train_ds, val_ds, cfg, device, K, seed=args.seed)

    val, test = evaluate(model, val_ds, device, K), evaluate(model, test_ds, device, K)  # test: uma vez
    name = f"{variant}_b{args.budget}_s{args.seed}"
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    plot_history(history, out / "curva_treino_val.png", f"{variant} — resnet18", best_epoch)
    plot_confusion(test["confusion"], classes, out / "matriz_confusao_test_public.png",
                   f"{variant}: F1 macro {test['f1_macro']:.3f} / acc {test['acc']:.3f} (test_public)")

    # Aluno x professor por classe (o professor foi medido em part1_teacher.py).
    t_json = Path("results/part1") / args.teacher / "metrics.json"
    t_f1 = json.loads(t_json.read_text())["f1_por_classe_test"] if t_json.exists() else {}
    s_f1 = dict(zip(classes, test["f1_per_class"]))
    summary = {
        "variant": variant, "params": n_params, "budget": args.budget, "seed": args.seed,
        "best_epoch": best_epoch, "memorizacao": memorization_summary(history),
        "val": {k: v for k, v in val.items() if k != "logits"},
        "test_public": {k: v for k, v in test.items() if k != "logits"},
        "f1_por_classe_aluno": s_f1, "f1_por_classe_professor": t_f1, "args": vars(args), "commit": git_commit(),
    }
    save_json(summary, out / "metrics.json")
    save_checkpoint(Path(args.ckpt_dir) / f"{name}.pt", model, "resnet18", classes, args.img_size,
                    ids_have_extension(labeled), extra={"variant": variant, "budget": args.budget, "seed": args.seed})
    append_curve(args.curve_csv, variant, args.budget, args.seed, test["f1_macro"], test["acc"])

    print(f"\n=== Aluno '{variant}': época {best_epoch} | val F1 {val['f1_macro']:.3f} | "
          f"test_public F1 macro {test['f1_macro']:.3f} | acurácia {test['acc']:.3f} "
          f"(teto do monitor {F1_TETO_MONITOR:.3f})")
    if t_f1:
        print(f"\n{'classe':<24}{'professor':>10}{'aluno':>8}{'aluno-prof':>12}")
        for k in classes:
            d = s_f1[k] - t_f1[k]
            print(f"{k:<24}{t_f1[k]:>10.2f}{s_f1[k]:>8.2f}{d:>+12.2f}" + ("  <- aluno melhor" if d > 0.05 else ""))
        print(f"\nProfessor: F1 macro {np.mean(list(t_f1.values())):.3f} | Aluno: {test['f1_macro']:.3f}")
    return summary


if __name__ == "__main__":
    main()
