# -*- coding: utf-8 -*-
"""Projecao t-SNE dos embeddings de 256 dimensoes (conjunto de teste, 1.973 amostras).
Usa embeddings do pipeline imageonlybalanced (melhor para discriminacao visual).
Saida: docs/tcc/final/artigo/images/tsne_embeddings.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NOMES   = {"akiec": "Ceratose actínica", "bcc": "Carc. basocelular",
           "bkl": "Ceratose benigna",   "df": "Dermatofibroma",
           "mel": "Melanoma",           "nv": "Nevo melanocítico",
           "vasc": "Lesão vascular"}
MALIGNAS = {"akiec", "bcc", "mel"}

# paleta: benignas em tons frios/neutros, malignas em tons quentes
CORES = {
    "bkl":   "#7BB3D1",   # azul claro
    "df":    "#B5A46A",   # ocre
    "nv":    "#4A7FB5",   # azul médio (dominante)
    "vasc":  "#9B82C2",   # lilás
    "akiec": "#E8901A",   # laranja (maligna)
    "bcc":   "#CC3333",   # vermelho (maligna)
    "mel":   "#7B0000",   # bordô (maligna)
}
MARCADORES = {dx: ("D" if dx in MALIGNAS else "o") for dx in CLASSES}
TAMANHOS   = {dx: (28 if dx in MALIGNAS else 16) for dx in CLASSES}

B   = Path("data/embeddings_cache/grouped/imageonlybalanced")
OUT = Path("docs/tcc/final/artigo/images"); OUT.mkdir(parents=True, exist_ok=True)

X = np.load(B / "X_test_imgfused.npy")          # (1973, 256)
y = np.load(B / "y_test.npy")                   # (1973,) — indices 0-6

print(f"Rodando t-SNE sobre {len(X)} amostras × {X.shape[1]} dimensoes...")
tsne = TSNE(n_components=2, perplexity=35, learning_rate="auto",
            max_iter=1200, random_state=42, verbose=1)
Z = tsne.fit_transform(X)
print("t-SNE concluido.")

fig, ax = plt.subplots(figsize=(7, 5.5))
# benignas por baixo, malignas por cima
ordem = [dx for dx in CLASSES if dx not in MALIGNAS] + \
        [dx for dx in CLASSES if dx in MALIGNAS]

for dx in ordem:
    idx = CLASSES.index(dx)
    mask = (y == idx)
    ax.scatter(Z[mask, 0], Z[mask, 1],
               c=CORES[dx], marker=MARCADORES[dx], s=TAMANHOS[dx],
               label=f"{NOMES[dx]} ({mask.sum()})",
               alpha=0.75 if dx not in MALIGNAS else 0.90,
               edgecolors="none" if dx not in MALIGNAS else "white",
               linewidths=0.4, zorder=3 if dx in MALIGNAS else 2)

ax.set_xlabel("t-SNE 1", fontsize=9)
ax.set_ylabel("t-SNE 2", fontsize=9)
ax.set_title("Projeção t-SNE — embeddings de 256 dim. (teste por lesão)", fontsize=9.5)
ax.tick_params(labelsize=8)
ax.grid(alpha=0.2)

leg = ax.legend(loc="upper right", fontsize=7.5, markerscale=1.3,
                framealpha=0.9, edgecolor="#cccccc",
                title="Classes (alto risco: ◆)", title_fontsize=7.5)

plt.tight_layout()
out = OUT / "tsne_embeddings.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Salvo: {out}")
