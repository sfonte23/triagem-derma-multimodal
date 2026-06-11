# -*- coding: utf-8 -*-
"""Diagrama da arquitetura do pipeline (inicio ao fim) para o artigo.
Extrator visual (CNN, 1x em GPU) -> embedding; Decisor classico (CPU) que
funde o clinico one-hot e calibra o limiar de triagem.
Saida: docs/tcc/final/artigo/images/arquitetura_pipeline.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

AZUL, VERDE, CINZA, ROXO = "#1B4480", "#6FA539", "#9AA7B4", "#7A4FA3"
FUNDO_CNN, FUNDO_DEC = "#EAF0F8", "#EFF4E8"

fig, ax = plt.subplots(figsize=(12, 5.2))
ax.set_xlim(0, 24); ax.set_ylim(0, 11); ax.axis("off")

def box(x, y, w, h, text, fc, tc="white", fs=9, ls="-", ec=None):
    ec = ec or fc
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
                       linewidth=1.4, edgecolor=ec, facecolor=fc, linestyle=ls)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", color=tc,
            fontsize=fs, weight="bold", zorder=5)
    return (x, y, w, h)

def arrow(p1, p2, color="#33414F", ls="-", rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                        lw=1.6, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=4)
    ax.add_patch(a)

def right(b): x,y,w,h=b; return (x+w, y+h/2)
def left(b):  x,y,w,h=b; return (x, y+h/2)
def top(b):   x,y,w,h=b; return (x+w/2, y+h)
def bottom(b):x,y,w,h=b; return (x+w/2, y)

# ---- faixas de fundo ----
ax.add_patch(FancyBboxPatch((0.2, 6.6), 23.6, 4.0, boxstyle="round,pad=0.1,rounding_size=0.2",
             fc=FUNDO_CNN, ec="#B9CBE3", lw=1.2, zorder=0))
ax.text(0.55, 10.2, "EXTRATOR VISUAL  (CNN — treinada 1× em GPU)", color=AZUL, fontsize=10, weight="bold")
ax.add_patch(FancyBboxPatch((0.2, 0.4), 23.6, 5.6, boxstyle="round,pad=0.1,rounding_size=0.2",
             fc=FUNDO_DEC, ec="#C5D6AE", lw=1.2, zorder=0))
ax.text(0.55, 5.55, "DECISOR CLÁSSICO  (CPU — sem re-treinar a rede)", color="#4C6B22", fontsize=10, weight="bold")

# ---- spine da CNN (topo): ... -> Dense128 -> EMBEDDING256 -> [Dense128 -> Softmax7] (descartados) ----
y1, h = 7.7, 1.5
b_foto = box(0.5,  y1, 2.5, h, "Foto\n320×320×3", AZUL)
b_eff  = box(3.4,  y1, 3.1, h, "EfficientNetB3\n(ImageNet)", AZUL)
b_gap  = box(6.9,  y1, 2.1, h, "GAP\n1536", AZUL)
b_den  = box(9.4,  y1, 2.1, h, "Dense\n128", AZUL)
b_emb  = box(11.9, y1, 2.9, h, "EMBEDDING\n256", ROXO)
b_den2 = box(15.7, y1, 2.1, h, "Dense\n128", CINZA, tc="#3a3a3a", ls="--")
b_soft = box(18.2, y1, 2.4, h, "Softmax\n7", CINZA, tc="#3a3a3a", ls="--")
for a, bx in [(b_foto, b_eff), (b_eff, b_gap), (b_gap, b_den), (b_den, b_emb)]:
    arrow(right(a), left(bx))
arrow(right(b_emb), left(b_den2), color=CINZA, ls="--")
arrow(right(b_den2), left(b_soft), color=CINZA, ls="--")
ax.text((15.7+20.6)/2, 9.45, "descartados após o treino", ha="center", va="bottom",
        fontsize=8, style="italic", color="#5a5a5a")

# anotacao do balanceamento na imagem (dentro da faixa, abaixo da Foto)
ax.annotate("balanceamento no espaço de imagem\n(antes do treino da CNN)",
            xy=bottom(b_foto), xytext=(3.6, 7.05), ha="center", va="center",
            fontsize=7, color="#4C6B22",
            arrowprops=dict(arrowstyle="-", color="#8Fae5e", lw=1))

# ---- decisor (base) ----
y2 = 2.0
b_clin = box(0.6, y2, 4.4, 1.7, "Metadados clínicos\nidade (Z), sexo, local\n(one-hot)", VERDE, fs=8.5)
c_cat = Circle((6.6, 3.4), 0.52, fc="white", ec="#33414F", lw=1.6, zorder=5)
ax.add_patch(c_cat); ax.text(6.6, 3.4, "+", ha="center", va="center", fontsize=15, weight="bold")
b_smote = box(8.1,  y2+0.35, 2.6, 1.0, "SMOTE\n(só treino)", CINZA, tc="#3a3a3a", fs=8)
b_xgb   = box(11.4, y2+0.15, 2.8, 1.4, "XGBoost", VERDE)
b_prob  = box(15.0, y2+0.15, 3.0, 1.4, "P(7 classes)", VERDE)
b_lim   = box(18.8, y2+0.15, 2.0, 1.4, "Limiar τ", ROXO)
b_dec   = box(21.1, y2+0.05, 2.6, 1.6, "Encaminhar?\n(sim / não)", AZUL, fs=8.5)

# embedding desce para o concat
arrow(bottom(b_emb), (6.6, 3.92), rad=-0.15)
arrow(right(b_clin), left((6.08,3.4,0,0)) if False else (6.08, 3.4))
arrow((7.12, 3.4), left(b_smote))
arrow(right(b_smote), left(b_xgb))
arrow(right(b_xgb), left(b_prob))
arrow(right(b_prob), left(b_lim))
arrow(right(b_lim), left(b_dec))

ax.text(12.0, 0.75, "SMOTE atua nos embeddings (espaço latente); o clínico entra one-hot só aqui, no decisor.",
        fontsize=7.5, color="#4C6B22", ha="center", style="italic")

plt.tight_layout()
out = Path("docs/tcc/final/artigo/images/arquitetura_pipeline.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("salvo:", out)
