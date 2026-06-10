# -*- coding: utf-8 -*-
"""Figura comparativa do artigo — CONSISTENTE com a Tabela 2 (desenho fatorial)
+ a barra da campea (Tabela 3, clinico one-hot no classificador).
Saida: resultados/comparativo_pipelines.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, VERDE, CINZA = "#1B4480", "#6FA539", "#9AA7B4"

# 4 variantes do desenho fatorial (= Tabela 2, exatamente) + campea (= Tabela 3)
labels = ["Multimodal\n(clinico na rede)", "Visao isolada",
          "Multimodal\n+ balanco", "Visao isolada\n+ balanco",
          "* CAMPEA\n+ clinico one-hot"]
auc = [0.763, 0.815, 0.868, 0.853, 0.883]
f1  = [0.290, 0.313, 0.402, 0.373, 0.409]
cores = [CINZA, AZUL, AZUL, AZUL, VERDE]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
x = np.arange(len(labels))

for k, (eixo, vals, titulo, ylab, lo, hi) in enumerate([
    (ax[0], auc, "AUC OVR por pipeline (teste por lesao)", "AUC OVR", 0.70, 0.92),
    (ax[1], f1,  "F1-Macro por pipeline (teste por lesao)", "F1-Macro", 0.25, 0.45),
]):
    bars = eixo.bar(x, vals, color=cores, width=0.62)
    eixo.set_ylim(lo, hi); eixo.set_ylabel(ylab); eixo.set_title(titulo)
    eixo.set_xticks(x); eixo.set_xticklabels(labels, fontsize=7)
    eixo.axvline(3.5, color="gray", ls=":", lw=0.8)   # separa fatorial (Tab.2) da campea (Tab.3)
    for b, v in zip(bars, vals):
        eixo.text(b.get_x()+b.get_width()/2, v+(hi-lo)*0.012, f"{v:.3f}", ha="center", fontsize=8)
    eixo.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = Path("resultados/comparativo_pipelines.png")
plt.savefig(out, dpi=150)
print("salvo:", out, "(+ copia na apresentacao)")
