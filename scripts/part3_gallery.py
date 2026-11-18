"""Parte 3: galeria de falhas e acertos do modelo final, com Grad-CAM.

Usa os pesos já treinados (weights/modelo_final.pt), sem treinar nada. Gera em results/part3/:
  - erros.png / acertos.png: imagem, classe verdadeira, prevista, confiança e Grad-CAM
  - galeria.csv: as imagens escolhidas
Erros: os 3 mais confiantes, de classes verdadeiras diferentes.
Acertos: 5 sorteados (seed 0), de classes diferentes.

Uso: python scripts/part3_gallery.py --data pa3_dados
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pa3.data import find_root, load_split, make_eval_ds, read_classes  # noqa: E402
from pa3.metrics import summarize  # noqa: E402
from pa3.models import load_checkpoint  # noqa: E402
from pa3.train import predict_logits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="pa3_dados")
p.add_argument("--weights", default="weights/modelo_final.pt")
p.add_argument("--out", default="results/part3")
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

model, meta = load_checkpoint(args.weights)
root = find_root(args.data)
classes = read_classes(root)
items = load_split(root, "test_public", classes)
ds = make_eval_ds(items, meta["img_size"])

probs = predict_logits(model, ds, "cpu").softmax(1)
conf, pred = probs.max(1)
y = ds.labels
print("F1 macro test_public:", round(summarize(y.numpy(), pred.numpy(), len(classes))["f1_macro"], 4))

# Escolha das imagens
erros = [i for i in conf.argsort(descending=True).tolist() if pred[i] != y[i]]
escolha_erros, vistos = [], set()
for i in erros:
    if int(y[i]) not in vistos:
        escolha_erros.append(i)
        vistos.add(int(y[i]))
    if len(escolha_erros) == 3:
        break
rng = np.random.default_rng(0)
acertos = rng.permutation(np.where((pred == y).numpy())[0]).tolist()
escolha_acertos, vistos = [], set()
for i in acertos:
    if int(y[i]) not in vistos:
        escolha_acertos.append(i)
        vistos.add(int(y[i]))
    if len(escolha_acertos) == 5:
        break


# Grad-CAM na última camada convolucional (layer4), para a classe prevista
def grad_cam(x):
    feats = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: feats.update(a=o))
    logits = model(x[None])
    h.remove()
    a = feats["a"]
    g = torch.autograd.grad(logits[0, logits.argmax()], a)[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1))[0]
    cam = F.interpolate(cam[None, None], size=x.shape[1:], mode="bilinear")[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().numpy()


def figura(indices, nome, titulo):
    fig, axs = plt.subplots(2, len(indices), figsize=(3.2 * len(indices), 6.6), squeeze=False)
    for c, i in enumerate(indices):
        img = np.asarray(ds.images[i]) / 255.0  # já em 224x224, como o modelo vê
        cam = grad_cam(ds[i][0])
        axs[0, c].imshow(img)
        axs[0, c].set_title(f"{items[i].image_id}\nverdadeira: {classes[y[i]]}\nprevista: {classes[pred[i]]} "
                            f"({conf[i]:.2f})", fontsize=8)
        axs[1, c].imshow(img)
        axs[1, c].imshow(cam, cmap="jet", alpha=0.45)
        axs[1, c].set_title("Grad-CAM (classe prevista)", fontsize=8)
        for a in axs[:, c]:
            a.axis("off")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(out / nome, dpi=130)
    plt.close(fig)
    print("salvo", out / nome)


figura(escolha_erros, "erros.png", "Três erros mais confiantes do modelo final (test_public)")
figura(escolha_acertos, "acertos.png", "Cinco acertos sorteados (test_public)")

linhas = [{"tipo": t, "image_id": items[i].image_id, "verdadeira": classes[y[i]], "prevista": classes[pred[i]],
           "confianca": round(float(conf[i]), 3), "brilho_medio": round(float(np.asarray(ds.images[i]).mean()), 1),
           "cinza": bool(np.ptp(np.asarray(ds.images[i]).astype(int), axis=2).mean() < 5)}
          for t, idx in [("erro", escolha_erros), ("acerto", escolha_acertos)] for i in idx]
pd.DataFrame(linhas).to_csv(out / "galeria.csv", index=False)
print(pd.DataFrame(linhas).to_string(index=False))
