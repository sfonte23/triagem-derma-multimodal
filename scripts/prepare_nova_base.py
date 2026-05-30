"""
prepare_nova_base.py — Etapa 2 ICV/UFPI

Carrega medical-staff/metadata.xlsx, filtra apenas as linhas validadas pelo
médico, identifica a fonte de cada imagem (HAM10000 ou Derm7pt), baixa as
imagens HAM10000 automaticamente via kagglehub, e orienta o download manual
das imagens Derm7pt. Gera data/nova_base/metadata_curada.csv pronto para
evaluate_nova_base.py.
"""

import os
import sys
import shutil
import warnings
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH      = ROOT / "medical-staff" / "metadata.xlsx"
OUTPUT_DIR     = ROOT / "data" / "nova_base"
OUTPUT_CSV     = OUTPUT_DIR / "metadata_curada.csv"
OUTPUT_IMAGES  = OUTPUT_DIR / "images"
DERM7PT_LIST   = OUTPUT_DIR / "derm7pt_needed.txt"

HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ---------------------------------------------------------------------------
# 1. Carregar e limpar metadata.xlsx
# ---------------------------------------------------------------------------
print("=" * 60)
print("ETAPA 1 — Carregando medical-staff/metadata.xlsx")
print("=" * 60)

if not XLSX_PATH.exists():
    print(f"[ERRO] Arquivo não encontrado: {XLSX_PATH}")
    sys.exit(1)

df_raw = pd.read_excel(XLSX_PATH)
print(f"  Linhas brutas: {len(df_raw)}, Colunas: {list(df_raw.columns)}")

# Remover colunas Unnamed (geradas por células vazias no Excel)
df_raw = df_raw[[c for c in df_raw.columns if not str(c).startswith("Unnamed")]]
print(f"  Colunas após limpeza: {list(df_raw.columns)}")

# ---------------------------------------------------------------------------
# 2. Filtrar apenas linhas preenchidas pelo médico
# ---------------------------------------------------------------------------
print("\nETAPA 2 — Filtrando linhas preenchidas pelo médico")

col_diag  = "DIAGNÓSTICO"
col_conf  = "NÍVEL DE CONFIANÇA"

mask = pd.Series([False] * len(df_raw))
if col_diag in df_raw.columns:
    mask = mask | df_raw[col_diag].notna()
if col_conf in df_raw.columns:
    mask = mask | df_raw[col_conf].notna()

df = df_raw[mask].reset_index(drop=True)
print(f"  Linhas com validação médica: {len(df)}")

if len(df) == 0:
    print("[ERRO] Nenhuma linha com DIAGNÓSTICO ou NÍVEL DE CONFIANÇA preenchido.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Identificar fonte por image_id
# ---------------------------------------------------------------------------
print("\nETAPA 3 — Identificando fonte de cada imagem")

df["dataset_source"] = df["image_id"].apply(
    lambda x: "HAM10000" if str(x).startswith("ISIC_") else "Derm7pt"
)
counts = df["dataset_source"].value_counts()
print(f"  HAM10000: {counts.get('HAM10000', 0)} imagens")
print(f"  Derm7pt:  {counts.get('Derm7pt', 0)} imagens")

# ---------------------------------------------------------------------------
# 4. Validar coluna dx
# ---------------------------------------------------------------------------
print("\nETAPA 4 — Validando coluna dx")

invalid_dx = df[~df["dx"].isin(HAM10000_CLASSES)]
if len(invalid_dx) > 0:
    print(f"  [AVISO] {len(invalid_dx)} linha(s) com dx fora do padrão HAM10000:")
    for _, row in invalid_dx.iterrows():
        print(f"    image_id={row['image_id']}  dx='{row['dx']}'")
else:
    print(f"  [OK] Todos os valores de dx estão nas 7 classes HAM10000.")

# ---------------------------------------------------------------------------
# 5. Baixar imagens HAM10000 (automático via kagglehub)
# ---------------------------------------------------------------------------
print("\nETAPA 5 — Baixando imagens HAM10000")

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)

ham_ids = set(df[df["dataset_source"] == "HAM10000"]["image_id"])
ham_copied = 0
ham_missing = []

if ham_ids:
    print(f"  Conectando ao kagglehub para {len(ham_ids)} imagens HAM10000...")
    try:
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        data_dir = Path(path)
        img_dirs = [data_dir / "HAM10000_images_part_1", data_dir / "HAM10000_images_part_2"]

        img_dict = {}
        for d in img_dirs:
            if d.exists():
                for img_file in os.listdir(d):
                    if img_file.lower().endswith(".jpg"):
                        img_dict[img_file[:-4]] = d / img_file

        for image_id in ham_ids:
            src = img_dict.get(image_id)
            dst = OUTPUT_IMAGES / f"{image_id}.jpg"
            if dst.exists():
                ham_copied += 1
                continue
            if src and src.exists():
                shutil.copy2(src, dst)
                ham_copied += 1
            else:
                ham_missing.append(image_id)
    except Exception as e:
        print(f"  [ERRO] Falha no download kagglehub: {e}")
        ham_missing = list(ham_ids)

    print(f"  HAM10000 copiadas: {ham_copied}/{len(ham_ids)}")
    if ham_missing:
        print(f"  [AVISO] Não encontradas no kagglehub: {ham_missing}")
else:
    print("  Nenhuma imagem HAM10000 na seleção do médico.")

# ---------------------------------------------------------------------------
# 6. Imagens Derm7pt — orientação para download manual
# ---------------------------------------------------------------------------
derm7_ids = list(df[df["dataset_source"] == "Derm7pt"]["image_id"])

if derm7_ids:
    print("\nETAPA 6 — Imagens Derm7pt (download manual necessário)")
    DERM7PT_LIST.write_text("\n".join(derm7_ids), encoding="utf-8")

    print("""
============================================================
 AÇÃO MANUAL NECESSÁRIA — Derm7pt Images
============================================================
 1. Registre-se em: http://derm.cs.sfu.ca
 2. Baixe o dataset Derm7pt e extraia as imagens.
 3. Copie os arquivos listados abaixo para:
    data/nova_base/images/

 A lista completa de image_ids necessários foi salva em:
    data/nova_base/derm7pt_needed.txt
============================================================""")
    print(f"\n  IDs Derm7pt necessários ({len(derm7_ids)} imagens):")
    for iid in derm7_ids[:20]:
        print(f"    {iid}")
    if len(derm7_ids) > 20:
        print(f"    ... e mais {len(derm7_ids) - 20} (ver derm7pt_needed.txt)")
else:
    print("\nETAPA 6 — Nenhuma imagem Derm7pt na seleção.")

# ---------------------------------------------------------------------------
# 7. Exportar CSV curado
# ---------------------------------------------------------------------------
print("\nETAPA 7 — Exportando metadata_curada.csv")

cols_export = ["image_id", "dx", "age", "sex", "localization", "dataset_source"]
if col_diag in df.columns:
    cols_export.append(col_diag)
if col_conf in df.columns:
    cols_export.append(col_conf)

df[cols_export].to_csv(OUTPUT_CSV, index=False)
print(f"  Salvo em: {OUTPUT_CSV}")
print(f"  Linhas: {len(df)}")

# ---------------------------------------------------------------------------
# 8. Validação final
# ---------------------------------------------------------------------------
print("\nETAPA 8 — Validação final")

exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
found_imgs = 0
missing_imgs = []

for _, row in df.iterrows():
    found = any((OUTPUT_IMAGES / f"{row['image_id']}{ext}").exists() for ext in exts)
    if found:
        found_imgs += 1
    else:
        missing_imgs.append((row["image_id"], row["dataset_source"]))

missing_ham   = [x for x in missing_imgs if x[1] == "HAM10000"]
missing_derm7 = [x for x in missing_imgs if x[1] == "Derm7pt"]
invalid_count = len(df[~df["dx"].isin(HAM10000_CLASSES)])

print(f"\n  {'[OK]' if len(df) > 0 else '[ERRO]'} Linhas carregadas: {len(df)}")
print(f"  {'[OK]' if not invalid_count else '[AVISO]'} Valores dx inválidos: {invalid_count}")
print(f"  {'[OK]' if found_imgs > 0 else '[AVISO]'} Imagens encontradas: {found_imgs}/{len(df)}")
if missing_ham:
    print(f"  [AVISO] HAM10000 faltando: {len(missing_ham)}")
if missing_derm7:
    print(f"  [INFO]  Derm7pt aguardando download manual: {len(missing_derm7)}")

if missing_imgs:
    print(f"\n  Distribuição das imagens encontradas:")
    print(f"    HAM10000: {counts.get('HAM10000', 0) - len(missing_ham)}/{counts.get('HAM10000', 0)}")
    print(f"    Derm7pt:  {counts.get('Derm7pt', 0) - len(missing_derm7)}/{counts.get('Derm7pt', 0)} (manual pendente)")

print("\n" + "=" * 60)
print("Pronto! Próximo passo:")
print("  1. [Se Derm7pt pendente] Copie as imagens para data/nova_base/images/")
print("  2. Execute: python scripts/evaluate_nova_base.py")
print("=" * 60)
