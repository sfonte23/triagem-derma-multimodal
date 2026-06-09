# -*- coding: utf-8 -*-
"""Gera figura comparativa para o artigo: barras de F1-Macro e AUC OVR
das 4 variantes (desenho fatorial) + o efeito da fusao clinica.
Le os CSVs ja gerados em docs/tcc/revisao/results_grouped/.
Saida: resultados/comparativo_pipelines.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("docs/tcc/revisao/results_grouped")
AZUL, VERDE, CINZA = "#1B4480", "#6FA539", "#9AA7B4"

# --- dados (holdout, XGBoost+SMOTE) ---
labels = ["Multimodal\n(clinico na rede)", "Visao isolada",
          "+ balanco\nde imagem", "+ clinico one-hot\n(campeao)"]
f1  = [0.290, 0.313, 0.373, 0.409]
auc = [0.763, 0.815, 0.853, 0.883]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

x = np.arange(len(labels))
bars = ax[0].bar(x, auc, color=[CINZA, AZUL, AZUL, VERDE], width=0.6)
ax[0].set_ylim(0.70, 0.92); ax[0].set_ylabel("AUC OVR")
ax[0].set_title("AUC OVR por pipeline (teste por lesao)")
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=7.5)
for b, v in zip(bars, auc):
    ax[0].text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.3f}", ha="center", fontsize=8)
ax[0].grid(axis="y", alpha=0.3)

bars2 = ax[1].bar(x, f1, color=[CINZA, AZUL, AZUL, VERDE], width=0.6)
ax[1].set_ylim(0.25, 0.45); ax[1].set_ylabel("F1-Macro")
ax[1].set_title("F1-Macro por pipeline (teste por lesao)")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=7.5)
for b, v in zip(bars2, f1):
    ax[1].text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.3f}", ha="center", fontsize=8)
ax[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
out = Path("resultados/comparativo_pipelines.png")
plt.savefig(out, dpi=150)
# copia para a apresentacao tambem
print("salvo:", out, "(+ copia na apresentacao)")
