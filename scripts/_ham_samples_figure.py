# -*- coding: utf-8 -*-
"""Painel com 1 imagem representativa de cada uma das 7 classes do HAM10000,
em linha, com a sigla e a contagem da classe embaixo (ilustra tambem o
desbalanceamento). Le do cache local do kagglehub (nao baixa nada).
Saida: docs/tcc/final/artigo/images/ham_classes.png
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CACHE = Path(os.path.expanduser(
    "~/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2"))
IMG_DIRS = [CACHE/"HAM10000_images_part_1", CACHE/"HAM10000_images_part_2"]
# benignas primeiro; pré-maligna (akiec) antes das malignas (bcc, mel)
ORDER = ["bkl", "df", "nv", "vasc", "akiec", "bcc", "mel"]
MALIGNAS    = {"bcc", "mel"}     # malignas no sentido estrito
PRE_MALIGNA = {"akiec"}          # pré-maligna / carcinoma in situ
NOMES = {"akiec": "Ceratose actínica/in situ", "bcc": "Carc. basocelular",
         "bkl": "Ceratose benigna", "df": "Dermatofibroma",
         "mel": "Melanoma", "nv": "Nevo melanocítico", "vasc": "Lesão vascular"}

COR_TITULO = {dx: ("#B23A48" if dx in MALIGNAS else
                   ("#C96A00" if dx in PRE_MALIGNA else "#33414F"))
              for dx in ORDER}
COR_BORDA  = {dx: ("#B23A48" if dx in MALIGNAS else
                   ("#C96A00" if dx in PRE_MALIGNA else "#33414F"))
              for dx in ORDER}
SUFIXO     = {dx: ("  (maligna)" if dx in MALIGNAS else
                   ("  (pré-maligna)" if dx in PRE_MALIGNA else ""))
              for dx in ORDER}

meta = pd.read_csv(CACHE/"HAM10000_metadata.csv")
counts = meta["dx"].value_counts()

def find_img(image_id):
    for d in IMG_DIRS:
        p = d/f"{image_id}.jpg"
        if p.exists():
            return p
    return None

fig, axes = plt.subplots(1, 7, figsize=(14, 2.5))
for ax, dx in zip(axes, ORDER):
    # primeira imagem da classe (deterministico)
    image_id = meta.loc[meta["dx"] == dx, "image_id"].sort_values().iloc[0]
    img = mpimg.imread(find_img(image_id))
    h, w = img.shape[:2]
    s = min(h, w)                       # recorte central quadrado
    img = img[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(COR_BORDA[dx]); sp.set_linewidth(1.5)
    ax.set_title(NOMES[dx], fontsize=8, color=COR_TITULO[dx], pad=3)
    ax.set_xlabel(f"{dx}  (n={counts[dx]}){SUFIXO[dx]}", fontsize=9,
                  weight="bold", color=COR_TITULO[dx])

plt.tight_layout()
out = Path("docs/tcc/final/artigo/images/ham_classes.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("salvo:", out)
