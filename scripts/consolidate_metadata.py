"""
Fase 3 — Consolidacao do metadata_base_nacional.csv.
Filtra ai_inferred.csv mantendo apenas as 283 imagens HAM10000-compativeis
e organiza no schema final compativel com o pipeline v1.1.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "data" / "base_nacional" / "ai_inferred.csv"
RAW  = ROOT / "data" / "base_nacional" / "raw_extraction.csv"
OUT  = ROOT / "data" / "base_nacional" / "metadata_base_nacional.csv"
EXCL = ROOT / "data" / "base_nacional" / "metadata_excluidas.csv"

HAM_CLASSES = {"akiec","bcc","bkl","df","mel","nv","vasc"}

# Carrega raw_extraction para buscar caption original
with RAW.open(encoding="utf-8") as f:
    raw_by_id = {r["image_id"]: r for r in csv.DictReader(f)}

# Carrega classificacoes
with SRC.open(encoding="utf-8") as f:
    classified = list(csv.DictReader(f))

incluidos = []
excluidos = []

for r in classified:
    image_id = r["image_id"]
    dx = r["dx_inferido"].strip()
    raw = raw_by_id.get(image_id, {})
    raw_caption = raw.get("nearby_caption", "")

    base_row = {
        "image_id":      image_id,
        "dx":            dx if dx in HAM_CLASSES else "",
        "age":           r["idade"].strip(),
        "sex":           r["sexo"].strip(),
        "localization":  r["localizacao"].strip(),
        "dataset_source": "AzulayBook" if r["source_book"] == "Azulay" else "GlossarioBook",
        "source_book":   r["source_book"],
        "source_page":   r["source_page"],
        "raw_caption":   raw_caption[:400],
        "nivel_confianca": r["confianca"],
        "fototipo":      r["fototipo"],
        "excluir":       0,  # padrao; medico marca 1 na curadoria
    }

    if dx in HAM_CLASSES:
        incluidos.append(base_row)
    else:
        # excluidos: tudo que nao e HAM (outros ou vazio)
        base_row["motivo_exclusao"] = "outros" if dx == "outros" else "sem_dx_claro"
        excluidos.append(base_row)

# Salvar incluidos
fieldnames = ["image_id","dx","age","sex","localization","dataset_source",
              "source_book","source_page","raw_caption","nivel_confianca",
              "fototipo","excluir"]
with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(incluidos)

# Salvar excluidos (para auditoria)
fn_ex = fieldnames + ["motivo_exclusao"]
with EXCL.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fn_ex)
    writer.writeheader()
    writer.writerows(excluidos)

# Estatisticas finais
from collections import Counter
dx_dist = Counter(r["dx"] for r in incluidos)
book_dist = Counter(r["dataset_source"] for r in incluidos)
foto_count = sum(1 for r in incluidos if r["fototipo"])
loc_count = sum(1 for r in incluidos if r["localization"])
age_count = sum(1 for r in incluidos if r["age"])
sex_count = sum(1 for r in incluidos if r["sex"])

print(f"=== metadata_base_nacional.csv ===")
print(f"Total incluidos: {len(incluidos)}")
print(f"\nDistribuicao por classe HAM10000:")
for dx in ["akiec","bcc","bkl","df","mel","nv","vasc"]:
    print(f"  {dx:6s}: {dx_dist[dx]:3d}")
print(f"\nDistribuicao por livro:")
for book, c in book_dist.items():
    print(f"  {book:15s}: {c}")
print(f"\nMetadados clinicos capturados:")
print(f"  localizacao: {loc_count}/{len(incluidos)} ({100*loc_count/len(incluidos):.1f}%)")
print(f"  idade:       {age_count}/{len(incluidos)} ({100*age_count/len(incluidos):.1f}%)")
print(f"  sexo:        {sex_count}/{len(incluidos)} ({100*sex_count/len(incluidos):.1f}%)")
print(f"  fototipo:    {foto_count}/{len(incluidos)} ({100*foto_count/len(incluidos):.1f}%)")
print(f"\nExcluidos (auditoria): {len(excluidos)} -> {EXCL.name}")
print(f"\nCSV final: {OUT}")
