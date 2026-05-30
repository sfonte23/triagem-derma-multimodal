"""
filter_book_images.py — Fase 1 do v2.0
Sinaliza imagens-lixo (pequenas, uniformes, escala de cinza, proporções extremas)
adicionando colunas booleanas ao raw_extraction.csv. NÃO exclui — só sinaliza.
"""
import sys
import csv
from pathlib import Path
import numpy as np
import cv2

sys.stdout.reconfigure(line_buffering=True)

ROOT     = Path(__file__).resolve().parent.parent
IMG_DIR  = ROOT / "data" / "base_nacional" / "images_raw"
CSV_PATH = ROOT / "data" / "base_nacional" / "raw_extraction.csv"

# Limiares
MIN_DIM            = 100   # < 100px em qualquer lado = pequena
UNIFORM_STD_THRESH = 12.0  # desvio padrão de luminância < 12 = uniforme
GRAYSCALE_CORR     = 0.99  # correlação entre canais > 0.99 = grayscale/diagrama
ASPECT_MIN, ASPECT_MAX = 0.3, 3.0

def analyze_image(img_path: Path) -> dict:
    """Retorna dict de flags + score."""
    img = cv2.imread(str(img_path))
    if img is None:
        return {"lixo_pequena": True, "lixo_uniforme": True,
                "lixo_escala_cinza": False, "lixo_proporcao_extrema": False,
                "lixo_score": 4, "_corrompida": True}

    h, w = img.shape[:2]
    pequena = w < MIN_DIM or h < MIN_DIM

    # std de luminância
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    uniforme = float(gray.std()) < UNIFORM_STD_THRESH

    # correlação entre canais (grayscale disfarçado de RGB)
    b, g, r = cv2.split(img)
    try:
        corr_bg = np.corrcoef(b.flatten(), g.flatten())[0, 1]
        corr_br = np.corrcoef(b.flatten(), r.flatten())[0, 1]
        corr_gr = np.corrcoef(g.flatten(), r.flatten())[0, 1]
        min_corr = min(corr_bg, corr_br, corr_gr)
        escala_cinza = min_corr > GRAYSCALE_CORR
    except Exception:
        escala_cinza = False

    # aspect ratio
    aspect = w / h if h > 0 else 0
    proporcao_extrema = aspect < ASPECT_MIN or aspect > ASPECT_MAX

    flags = {
        "lixo_pequena":            bool(pequena),
        "lixo_uniforme":           bool(uniforme),
        "lixo_escala_cinza":       bool(escala_cinza),
        "lixo_proporcao_extrema":  bool(proporcao_extrema),
    }
    score = sum(flags.values())
    flags["lixo_score"] = score
    flags["_corrompida"] = False
    return flags

def main():
    if not CSV_PATH.exists():
        print(f"[ERRO] {CSV_PATH} não encontrado. Rode extract_book_images.py primeiro.")
        sys.exit(1)

    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    print(f"Analisando {len(rows)} imagens...")

    new_fields = ["lixo_pequena", "lixo_uniforme", "lixo_escala_cinza",
                  "lixo_proporcao_extrema", "lixo_score"]
    for nf in new_fields:
        if nf not in fieldnames:
            fieldnames.append(nf)

    counters = {f: 0 for f in new_fields}
    score_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for i, row in enumerate(rows):
        img_path = IMG_DIR / f"{row['image_id']}.png"
        if not img_path.exists():
            print(f"  [warn] imagem não encontrada: {img_path.name}")
            continue
        flags = analyze_image(img_path)
        for nf in new_fields:
            row[nf] = flags[nf]
            if nf != "lixo_score" and flags[nf]:
                counters[nf] += 1
        score_dist[flags["lixo_score"]] = score_dist.get(flags["lixo_score"], 0) + 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSinalizações:")
    for nf, c in counters.items():
        print(f"  {nf}: {c} ({100*c/len(rows):.1f}%)")
    print(f"\nDistribuição de lixo_score:")
    for s, c in sorted(score_dist.items()):
        print(f"  score={s}: {c} imagens ({100*c/len(rows):.1f}%)")
    print(f"\nCSV atualizado: {CSV_PATH}")
    print(f"Imagens com lixo_score == 0 (candidatas prioritárias): {score_dist.get(0, 0)}")

if __name__ == "__main__":
    main()
