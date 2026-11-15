from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from pa3 import F1_TETO_MONITOR, ORCAMENTO_PADRAO, ORCAMENTOS  # noqa: E402
from pa3.data import (check_splits, find_root, ids_have_extension, load_split, make_eval_ds,  # noqa: E402
                      make_train_ds, read_classes)
from pa3.metrics import memorization_summary, plot_confusion, plot_history  # noqa: E402
from pa3.models import build_model, check_param_budget, save_checkpoint  # noqa: E402
from pa3.train import FitConfig, evaluate, fit  # noqa: E402
from pa3.utils import append_curve, get_device, git_commit, save_json  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.environ.get("PA3_DATA", "data/pa3_dados"))
    p.add_argument("--budget", type=int, default=ORCAMENTO_PADRAO, choices=ORCAMENTOS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--load-size", type=int, default=256, help="resolução em RAM do treino (>= img-size)")
    p.add_argument("--epochs-scratch", type=int, default=60)
    p.add_argument("--epochs-ft", type=int, default=30)
    p.add_argument("--lr-scratch", type=float, default=2e-3)
    p.add_argument("--lr-ft", type=float, default=1e-3, help="taxa da cabeça; o corpo usa lr*0.1")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--skip-scratch", action="store_true", help="roda só a versão (b)")
    p.add_argument("--no-pretrained", action="store_true", help="(testes) não baixa pesos do ImageNet")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/part0")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--curve-csv", default="curva_orcamento.csv")
    return p.parse_args(argv)


def run_one(tag: str, arch: str, pretrained: bool, epochs: int, lr: float, mult: float, args, ctx) -> dict:
    classes, labeled, val_ds, test_ds, device, id_ext = (
        ctx["classes"], ctx["labeled"], ctx["val_ds"], ctx["test_ds"], ctx["device"], ctx["id_ext"])
    K = len(classes)
    name = f"{tag}_b{args.budget}_s{args.seed}"
    out = Path(args.out) / name
    print(f"\n=== {tag}: {arch} ({'ImageNet' if pretrained else 'do zero'}), {epochs} épocas ===")

    model = build_model(arch, K, pretrained=pretrained)
    n_params = check_param_budget(model)
    train_ds = make_train_ds(labeled, args.img_size, args.load_size)
    cfg = FitConfig(epochs=epochs, batch_size=args.batch_size, lr=lr, backbone_lr_mult=mult,
                    num_workers=args.workers)
    history, best_epoch = fit(model, train_ds, val_ds, cfg, device, K, seed=args.seed)

    # Seleção feita; agora sim, uma única avaliação no test_public.
    val = evaluate(model, val_ds, device, K)
    test = evaluate(model, test_ds, device, K)
    mem = memorization_summary(history)

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    plot_history(history, out / "curva_treino_val.png", f"{tag} — {arch}", best_epoch)
    plot_confusion(test["confusion"], classes, out / "matriz_confusao_test_public.png",
                   f"{tag}: F1 macro {test['f1_macro']:.3f} / acc {test['acc']:.3f} (test_public)")
    summary = {
        "tag": tag, "arch": arch, "pretrained": pretrained, "params": n_params,
        "budget": args.budget, "seed": args.seed, "best_epoch": best_epoch, "memorizacao": mem,
        "val": {k: v for k, v in val.items() if k != "logits"},
        "test_public": {k: v for k, v in test.items() if k != "logits"},
        "f1_por_classe_test": dict(zip(classes, test["f1_per_class"])),
        "args": vars(args), "commit": git_commit(),
    }
    save_json(summary, out / "metrics.json")
    thr = mem["epoch_train_acc_thr"]
    print(f"  -> época escolhida (val): {best_epoch} | perda de val mínima na época {mem['epoch_min_val_loss']} | "
          f"acurácia de treino >= {mem['train_acc_thr']:.0%} " + (f"na época {thr}" if thr else "nunca"))
    save_checkpoint(Path(args.ckpt_dir) / f"{name}.pt", model, arch, classes, args.img_size, id_ext,
                    extra={"tag": tag, "budget": args.budget, "seed": args.seed, "best_epoch": best_epoch})
    if tag == "piso_b":  # a versão (b) é "o piso" da curva de orçamento (Parte 2)
        append_curve(args.curve_csv, "piso", args.budget, args.seed, test["f1_macro"], test["acc"])
    return summary


def main(argv=None) -> list[dict]:
    args = parse_args(argv)
    root = find_root(args.data)
    classes = read_classes(root)
    device = get_device(args.device)
    print(f"Dados: {root} | {len(classes)} classes | dispositivo: {device}")

    labeled = load_split(root, f"rotulos_{args.budget}", classes)
    val_items, test_items = load_split(root, "val", classes), load_split(root, "test_public", classes)
    check_splits(load_split(root, "train", classes), labeled, val_items, test_items)
    print(f"rotulos_{args.budget}: {len(labeled)} imagens | val: {len(val_items)} | test_public: {len(test_items)}")

    ctx = {
        "classes": classes, "labeled": labeled, "device": device, "id_ext": ids_have_extension(labeled),
        "val_ds": make_eval_ds(val_items, args.img_size), "test_ds": make_eval_ds(test_items, args.img_size),
    }
    results = []
    if not args.skip_scratch:
        results.append(run_one("piso_a", "small_cnn", False, args.epochs_scratch, args.lr_scratch, 1.0, args, ctx))
    results.append(run_one("piso_b", "resnet18", not args.no_pretrained, args.epochs_ft, args.lr_ft, 0.1, args, ctx))

    print("\n=== Resumo (F1 macro e acurácia no test_public; época escolhida no val) ===")
    print(f"{'versão':<8}{'params':>12}{'época':>7}{'val F1':>9}{'test F1':>9}{'test acc':>10}")
    for r in results:
        print(f"{r['tag']:<8}{r['params']:>12,}{r['best_epoch']:>7}{r['val']['f1_macro']:>9.3f}"
              f"{r['test_public']['f1_macro']:>9.3f}{r['test_public']['acc']:>10.3f}")
    print(f"(teto do monitor: {F1_TETO_MONITOR:.3f} de F1 macro no test_public)")
    return results


if __name__ == "__main__":
    main()
