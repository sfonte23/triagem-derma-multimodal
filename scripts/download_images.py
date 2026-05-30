"""
download_images.py — Baixa as imagens da nova base de avaliação (Etapa 2 ICV/UFPI)

Estratégia:
  - Imagens com prefixo 'ham_ISIC_': são imagens HAM10000 referenciadas pelo Derm7pt.
    Strip do prefixo 'ham_' → busca no cache kagglehub do HAM10000 → copia.
  - Imagens com prefixo 'derm7pt_': são imagens nativas do Derm7pt.
    Tenta baixar via Kaggle (com aceitação de regras) → fallback para instrução manual.

Pré-requisito: rodar prepare_nova_base.py primeiro.
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path

import kagglehub
import pandas as pd
import requests

ROOT          = Path(__file__).resolve().parent.parent
NOVA_BASE_CSV = ROOT / "data" / "nova_base" / "metadata_curada.csv"
OUTPUT_IMAGES = ROOT / "data" / "nova_base" / "images"
DERM7PT_CACHE = ROOT / "data" / "derm7pt_raw"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
DERM7PT_CACHE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Carregar lista de imagens necessárias
# ---------------------------------------------------------------------------
if not NOVA_BASE_CSV.exists():
    print("[ERRO] Execute prepare_nova_base.py primeiro.")
    sys.exit(1)

df = pd.read_csv(NOVA_BASE_CSV)
ham_ids   = [x for x in df["image_id"] if str(x).startswith("ham_ISIC_")]
derm7_ids = [x for x in df["image_id"] if str(x).startswith("derm7pt_")]

print(f"Imagens a baixar: {len(ham_ids)} HAM10000  +  {len(derm7_ids)} Derm7pt nativas")
print("=" * 60)

# ---------------------------------------------------------------------------
# PARTE 1: imagens HAM10000 (prefixo ham_ISIC_)
# ---------------------------------------------------------------------------
print(f"\n[1/2] Copiando {len(ham_ids)} imagens HAM10000 do cache kagglehub...")

path_ham = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
data_dir  = Path(path_ham)
img_dirs  = [data_dir / "HAM10000_images_part_1", data_dir / "HAM10000_images_part_2"]

# Construir dicionário de todas as imagens disponíveis no cache
img_dict = {}
for d in img_dirs:
    if d.exists():
        for f in os.listdir(d):
            if f.lower().endswith(".jpg"):
                img_dict[f[:-4]] = d / f  # key: ISIC_XXXXXXX

copied_ham = 0
missing_ham = []
for full_id in ham_ids:
    isic_id = full_id.replace("ham_", "", 1)   # ham_ISIC_0024450 → ISIC_0024450
    dst     = OUTPUT_IMAGES / f"{full_id}.jpg"

    if dst.exists():
        copied_ham += 1
        continue

    src = img_dict.get(isic_id)
    if src and src.exists():
        shutil.copy2(src, dst)
        copied_ham += 1
    else:
        missing_ham.append(full_id)

print(f"  Copiadas: {copied_ham}/{len(ham_ids)}")
if missing_ham:
    print(f"  [AVISO] Não encontradas: {missing_ham}")

# ---------------------------------------------------------------------------
# PARTE 2: imagens Derm7pt nativas (prefixo derm7pt_)
# ---------------------------------------------------------------------------
print(f"\n[2/2] Baixando {len(derm7_ids)} imagens Derm7pt nativas...")

# Os IDs na base são como 'derm7pt_Ael455' — no dataset Derm7pt o arquivo
# é 'Ael455_derm.jpg' (dermoscopia) ou 'Ael455.jpg'

def strip_derm7pt(full_id):
    return full_id.replace("derm7pt_", "", 1)   # derm7pt_Ael455 → Ael455

# Tentativa 1: kagglehub (requer aceitar regras no site Kaggle)
derm7pt_path = None
KAGGLE_SLUGS = [
    "jeremykawahara/derm7pt",
    "mariaherrero/derm7pt",
    "drscarlat/derm7pt",
]

for slug in KAGGLE_SLUGS:
    try:
        print(f"  Tentando kagglehub: {slug} ...")
        derm7pt_path = Path(kagglehub.dataset_download(slug))
        print(f"  Kaggle OK: {derm7pt_path}")
        break
    except Exception as e:
        print(f"  Falhou ({slug}): {e}")

copied_derm7 = 0
missing_derm7 = []

if derm7pt_path and derm7pt_path.exists():
    # Indexar todas as imagens disponíveis no dataset Derm7pt baixado
    derm7_img_index = {}
    for img_file in derm7pt_path.rglob("*.jpg"):
        derm7_img_index[img_file.stem.lower()] = img_file
    for img_file in derm7pt_path.rglob("*.png"):
        derm7_img_index[img_file.stem.lower()] = img_file

    for full_id in derm7_ids:
        case_id = strip_derm7pt(full_id)       # ex: Ael455
        dst     = OUTPUT_IMAGES / f"{full_id}.jpg"

        if dst.exists():
            copied_derm7 += 1
            continue

        # Tentar variações de nome: Ael455_derm, Ael455, ael455_derm, etc.
        candidates = [
            case_id.lower() + "_derm",
            case_id.lower(),
            case_id.lower() + "_dermoscopy",
        ]
        found = None
        for cand in candidates:
            if cand in derm7_img_index:
                found = derm7_img_index[cand]
                break

        if found:
            shutil.copy2(found, dst)
            copied_derm7 += 1
        else:
            missing_derm7.append(full_id)
            print(f"  [AVISO] Não encontrado no dataset: {full_id} (buscou: {candidates})")

    print(f"  Copiadas: {copied_derm7}/{len(derm7_ids)}")

else:
    missing_derm7 = list(derm7_ids)
    print(
        "\n  ============================================================\n"
        "  DOWNLOAD MANUAL NECESSARIO -- Derm7pt (64 imagens)\n"
        "  ============================================================\n"
        "  O Derm7pt requer aceitar termos de uso antes de baixar.\n\n"
        "  OPCAO A -- Kaggle (recomendado):\n"
        "    1. Acesse: https://www.kaggle.com/datasets/jeremykawahara/derm7pt\n"
        "    2. Clique em 'I understand and accept' (aceitar regras)\n"
        "    3. Rode este script novamente -- o download sera automatico.\n\n"
        "  OPCAO B -- Site oficial:\n"
        "    1. Acesse: http://derm.cs.sfu.ca\n"
        "    2. Preencha o formulario de acesso\n"
        "    3. Baixe o zip e extraia\n"
        "    4. Copie pasta images/derm/ para data/nova_base/images/\n"
        "       Renomeando: Ael455_derm.jpg -> derm7pt_Ael455.jpg\n\n"
        "  IDs necessarios (ver data/nova_base/derm7pt_needed.txt):"
    )
    for iid in derm7_ids[:15]:
        print(f"    {iid}")
    if len(derm7_ids) > 15:
        print(f"    ... e mais {len(derm7_ids) - 15}")
    print("  ============================================================")

# ---------------------------------------------------------------------------
# Sumário final
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
total_found = sum(
    1 for row in df.itertuples()
    if any((OUTPUT_IMAGES / f"{row.image_id}{ext}").exists()
           for ext in [".jpg", ".jpeg", ".png"])
)
print(f"RESULTADO: {total_found}/{len(df)} imagens disponíveis em data/nova_base/images/")
print(f"  HAM10000 (ham_ISIC_): {copied_ham}/{len(ham_ids)}")
print(f"  Derm7pt nativas:      {copied_derm7}/{len(derm7_ids)}")

if total_found == len(df):
    print("\n  Todas as imagens disponíveis!")
    print("  Execute: python scripts/evaluate_nova_base.py")
elif total_found > 0:
    print(f"\n  {len(df) - total_found} imagens ainda faltando (Derm7pt manual).")
    print("  É possível rodar evaluate_nova_base.py com as imagens disponíveis.")
    print("  Execute: python scripts/evaluate_nova_base.py")
else:
    print("\n  Nenhuma imagem disponível ainda.")
print("=" * 60)
