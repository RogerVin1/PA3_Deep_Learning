from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pa3 import ORCAMENTO_PADRAO, ORCAMENTOS  # noqa: E402
from pa3.data import find_root, load_split, read_classes  # noqa: E402
from pa3.metrics import plot_confusion, summarize  # noqa: E402
from pa3.teacher import (class_prompts, encode_classes, encode_images, load_teacher,  # noqa: E402
                         softmax_confidence, squash_transform, zero_shot_logits)
from pa3.utils import get_device, git_commit, save_json  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.environ.get("PA3_DATA", "data/pa3_dados"))
    p.add_argument("--teacher", default="bioclip2", help="bioclip2 | bioclip | id do open_clip")
    p.add_argument("--budget", type=int, default=ORCAMENTO_PADRAO, choices=ORCAMENTOS, help="orçamento cujos rótulos servem de diagnóstico do professor")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/part1")
    p.add_argument("--cache-dir", default="checkpoints")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    root = find_root(args.data)
    classes = read_classes(root)
    K = len(classes)
    device = get_device(args.device)
    tag = args.teacher.replace("hf-hub:", "").replace("/", "_")
    out = Path(args.out) / tag
    cache = Path(args.cache_dir) / f"teacher_{tag}.pt"

    # O val NÃO é carregado em nenhum ponto deste script (regra da Trilha C).
    train, test = load_split(root, "train", classes), load_split(root, "test_public", classes)
    labeled = load_split(root, f"rotulos_{args.budget}", classes)
    print(f"Professor: {args.teacher} | dispositivo: {device} | train {len(train)} | test_public {len(test)}")

    if cache.exists():
        c = torch.load(cache, weights_only=True)
        f_train, f_test, f_txt, scale = c["train"].float(), c["test"].float(), c["text"].float(), c["scale"]
        print(f"Embeddings lidos de {cache}")
    else:
        model, preprocess, tokenizer = load_teacher(args.teacher, device)
        tf = squash_transform(model, preprocess)
        scale = float(model.logit_scale.exp().detach())
        f_txt = encode_classes(model, tokenizer, class_prompts(classes), device)
        f_train = encode_images(model, train, tf, device, args.batch_size, args.workers)
        f_test = encode_images(model, test, tf, device, args.batch_size, args.workers)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"train": f_train.half(), "test": f_test.half(), "text": f_txt.half(), "scale": scale}, cache)
        print(f"Embeddings salvos em {cache}")

    # --- desempenho zero-shot do professor no test_public (linha de referência) ---
    logits_test = zero_shot_logits(f_test, f_txt, scale)
    y_test = np.array([i.label for i in test])
    m = summarize(y_test, logits_test.argmax(1).numpy(), K)

    # --- pseudo-rótulos do train + diagnóstico nos rótulos do orçamento (permitido: são do pacote) ---
    logits_train = zero_shot_logits(f_train, f_txt, scale)
    pred, conf = softmax_confidence(logits_train)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": [i.image_id for i in train], "teacher_label": [classes[p] for p in pred],
                  "confidence": conf.round(4)}).to_csv(out / "pseudo_rotulos_train.csv", index=False)

    pos = {Path(i.image_id).stem: n for n, i in enumerate(train)}  # stem: com ou sem ".jpg" no CSV
    idx = np.array([pos[Path(i.image_id).stem] for i in labeled])
    y_lab = np.array([i.label for i in labeled])
    m_lab = summarize(y_lab, pred[idx], K)

    plot_confusion(m["confusion"], classes, out / "matriz_confusao_professor.png",
                   f"Professor {tag} zero-shot: F1 macro {m['f1_macro']:.3f} / acc {m['acc']:.3f} (test_public)")
    summary = {
        "teacher": args.teacher, "scale": scale, "prompts": dict(zip(classes, class_prompts(classes))),
        "test_public": m, f"rotulos_{args.budget}": {k: v for k, v in m_lab.items() if k != "confusion"},
        "f1_por_classe_test": dict(zip(classes, m["f1_per_class"])),
        "pseudo_rotulos_por_classe": dict(zip(classes, np.bincount(pred, minlength=K).tolist())),
        "confianca_media_train": float(conf.mean()), "args": vars(args), "commit": git_commit(),
    }
    save_json(summary, out / "metrics.json")

    print(f"\n=== Professor zero-shot no test_public: F1 macro {m['f1_macro']:.3f} | acurácia {m['acc']:.3f}")
    print(f"=== Mesmo professor nas {len(labeled)} imagens rotuladas (rotulos_{args.budget}): "
          f"F1 macro {m_lab['f1_macro']:.3f} | acurácia {m_lab['acc']:.3f}")
    print(f"\n{'classe':<24}{'F1 test':>8}{'pseudo-rótulos no train':>26}")
    for k, name in enumerate(classes):
        print(f"{name:<24}{m['f1_per_class'][k]:>8.2f}{int((pred == k).sum()):>26}")
    return summary


if __name__ == "__main__":
    main()
