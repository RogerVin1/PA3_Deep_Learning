"""Parte 2: curva de orçamento e ganho por classe no orçamento de 5.

Lê curva_orcamento.csv e os metrics.json do piso e do aluno e gera, em results/part2/:
  - curva_orcamento.png: F1 macro (test_public) x rótulos por classe (eixo x log)
  - resumo.csv: média ± desvio por método e orçamento
  - ganho_por_classe.csv: F1 por classe do aluno menos o do piso (orçamento 5, média das seeds)

Uso: python scripts/part2_curve.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pa3 import F1_TETO_MONITOR  # noqa: E402

ALUNO = "aluno_soft_cal"
NOMES = {"piso": "Piso (ResNet-18, só rótulos)", ALUNO: "Aluno (soft + professor calibrado)",
         "professor_cal": "Professor calibrado (viés + T)", "professor": "Professor zero-shot (BioCLIP 2)"}
OUT = Path("results/part2")
OUT.mkdir(parents=True, exist_ok=True)

curva = pd.read_csv("curva_orcamento.csv")
curva = curva[curva.metodo.isin(NOMES)]
resumo = (curva.groupby(["metodo", "orcamento"])
          .agg(f1_media=("f1_macro", "mean"), f1_desvio=("f1_macro", "std"),
               acc_media=("acuracia", "mean"), acc_desvio=("acuracia", "std"), seeds=("seed", "count"))
          .round(4).reset_index())
resumo.to_csv(OUT / "resumo.csv", index=False)
print(resumo.to_string(index=False))

# Gráfico
fig, ax = plt.subplots(figsize=(7, 4.5))
for metodo, estilo in [("piso", "o-"), (ALUNO, "s-"), ("professor_cal", "^--")]:
    r = resumo[resumo.metodo == metodo].sort_values("orcamento")
    ax.errorbar(r.orcamento, r.f1_media, yerr=r.f1_desvio.fillna(0), fmt=estilo, capsize=4, label=NOMES[metodo])
zs = resumo[resumo.metodo == "professor"].f1_media.mean()  # zero-shot não usa rótulos: linha horizontal
ax.axhline(zs, color="gray", ls=":", label=f"{NOMES['professor']} = {zs:.3f}")
ax.axhline(F1_TETO_MONITOR, color="black", ls="--", label=f"Teto (monitor, todos os rótulos) = {F1_TETO_MONITOR}")
ax.set_xscale("log")
ax.set_xticks([1, 5, 20], ["1", "5", "20"])
ax.set_xlabel("rótulos por classe (escala log)")
ax.set_ylabel("F1 macro (test_public)")
ax.set_title("Curva de orçamento (orçamento 5: média ± desvio de 3 seeds)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "curva_orcamento.png", dpi=150)
print("salvo", OUT / "curva_orcamento.png")

# Ganho por classe no orçamento de 5 (média das seeds disponíveis)
seeds = sorted(curva[(curva.metodo == ALUNO) & (curva.orcamento == 5)].seed)
piso = pd.DataFrame({s: json.load(open(f"results/part0/piso_b_b5_s{s}/metrics.json", encoding="utf-8"))
                     ["f1_por_classe_test"] for s in seeds}).mean(axis=1)
aluno = pd.DataFrame({s: json.load(open(f"results/part1/aluno/{ALUNO}_b5_s{s}/metrics.json", encoding="utf-8"))
                      ["f1_por_classe_aluno"] for s in seeds}).mean(axis=1)
ganho = pd.DataFrame({"piso": piso, "aluno": aluno, "ganho": aluno - piso}).sort_values("ganho", ascending=False)
ganho.round(3).to_csv(OUT / "ganho_por_classe.csv", index_label="classe")
print(f"\nGanho por classe (orçamento 5, seeds {seeds}):")
print(ganho.round(3).to_string())
