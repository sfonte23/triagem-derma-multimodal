"""
Classifica os 595 candidatos tumorais aplicando o mapeamento permissivo
(documentado em docs/entregas/2.0_artigo_multimodal_base_nacional/mapeamento_permissivo.md).

Classificador deterministico baseado em busca textual em caption + nearby_text.
Cada linha gera: dx_inferido, dx_original_texto, localizacao, idade, sexo,
fototipo, confianca, justificativa.

Output: data/base_nacional/ai_inferred.csv (append do piloto + 595 candidatos)
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "data" / "base_nacional" / "candidatos_tumorais.csv"
DEST = ROOT / "data" / "base_nacional" / "ai_inferred.csv"
PILOTO = ROOT / "data" / "base_nacional" / "ai_inferred.csv"  # piloto ja salvo

# =============================================================================
# Mapeamento permissivo: termos -> classe HAM10000
# Ordem importa: termos mais especificos primeiro. Score = confianca.
# =============================================================================
RULES = [
    # === MEL — Melanoma (qualquer subtipo) ===
    ("mel", 0.95, [r"melanoma\s+maligno", r"melanoma\s+lentig", r"melanoma\s+nodular",
                   r"melanoma\s+superficial", r"melanoma\s+acral", r"melanoma\s+amelan",
                   r"melanoma\s+in\s+situ", r"melanoma\s+espitz", r"\bmelanoma\b"]),
    ("mel", 0.85, [r"lentigo\s+maligno", r"\bLMM\b", r"\bMLN\b"]),
    # === BCC — Carcinoma Basocelular ===
    ("bcc", 0.95, [r"carcinoma\s+basocelular", r"\bCBC\b", r"epitelioma\s+basocelular",
                   r"\bbasalioma\b", r"ulcus\s+rodens", r"basocelular\s+nodular",
                   r"basocelular\s+ulcerativ", r"basocelular\s+pigmentad",
                   r"basocelular\s+esclerod", r"basocelular\s+superficial"]),
    # === AKIEC — Ceratose Actinica + Carcinoma intraepitelial ===
    ("akiec", 0.95, [r"cerat[oô]se\s+act[íi]nica", r"cerat[oô]se\s+solar",
                     r"doen[çc]a\s+de\s+bowen", r"\bbowen\b",
                     r"carcinoma\s+intraep", r"carcinoma\s+in\s+situ",
                     r"eritroplasia\s+de\s+queyrat", r"queilite\s+act[íi]nica",
                     r"cerat[oô]se\s+pr[éeé]-?canceros", r"cerat[oô]se\s+pr[éeé]-?malign"]),
    # === BKL — Lesoes Benignas Queratoticas ===
    ("bkl", 0.95, [r"cerat[oô]se\s+seborreica", r"querat[oô]se\s+seborreica"]),
    ("bkl", 0.85, [r"\blentigo\s+sol", r"\blentigo\s+senil", r"\blentigo\s+simples",
                   r"dermatose\s+papulosa\s+nigra", r"acantoma\s+de\s+c[éeé]lulas\s+claras"]),
    # === DF — Dermatofibroma ===
    ("df", 0.95, [r"\bdermatofibroma\b", r"histiocitoma\s+fibroso"]),
    # === NV — Nevo Melanocitico ===
    ("nv", 0.95, [r"nevo\s+melanoc[íi]tico", r"nevo\s+cong[êe]nito",
                  r"nevo\s+azul", r"nevo\s+de\s+spitz", r"nevo\s+de\s+reed",
                  r"nevo\s+de\s+sutton", r"\bnevo\s+displ", r"\bnevo\s+at[íi]pico",
                  r"\bnevo\s+comum", r"\bnevo\s+dermal", r"\bnevo\s+composto",
                  r"\bnevo\s+juncional"]),
    ("nv", 0.7,  [r"\bnevo\b(?!\s+verrugoso)(?!\s+epid)(?!\s+sebac)"]),  # nevo generico
    # === VASC — Vasculares ===
    ("vasc", 0.95, [r"\bhemangioma\b", r"angioma\s+rubi", r"angioma\s+senil",
                    r"angioma\s+estelar", r"angioma\s+plano", r"angioma\s+cavernoso",
                    r"granuloma\s+piog[êe]nico", r"lago\s+venoso",
                    r"telangiectasia\s+focal"]),
    ("vasc", 0.6,  [r"\bangioma\b", r"malforma[çc][ãa]o\s+vascular"]),  # generico

    # === OUTROS — termos explicitos que NAO mapeiam ===
    # (priorizamos esses pra evitar falsos positivos)
    ("outros", 0.95, [
        r"carcinoma\s+espinocelular", r"\bCEC\b", r"carcinoma\s+de\s+c[éeé]lulas\s+escamosas",
        r"queratoacantoma", r"cerat[oô]acantoma",
        r"sarcoma\s+de\s+kaposi", r"linfoma\s+cut[âa]neo",
        r"cilindroma", r"tricoepitelioma", r"tricoblastoma", r"siringoma",
        r"pilomatrixoma", r"poroma", r"hidradenoma",
        r"\bcisto\s+epid", r"\bcisto\s+sebac", r"\blipoma\b",
        r"queratose\s+folicular",
        r"micose\s+fungoid", r"linfoma\s+B\b",
        r"sebaceous", r"\bxantoma\b", r"\bxantelasma",
    ]),
    # Doencas nao tumorais explicitas
    ("outros", 0.9, [
        r"psor[íi]ase", r"vitiligo", r"dermatite", r"eczema",
        r"hansen[íi]ase", r"hansen", r"\blepra\b",
        r"leishmaniose", r"esporotricose", r"cromoblastom",
        r"esca?biose", r"larva\s+migrans", r"prurigo",
        r"pênfigo", r"penfigoide", r"epiderm[óo]lise\s+bolhosa",
        r"s[íi]filis", r"treponema", r"condiloma",
        r"verruga", r"herpes", r"\bzoster\b",
        r"tinha", r"\btinea\b", r"candid[íi]ase", r"piti?r[íi]ase",
        r"acne", r"ros[áa]cea",
        r"queloide", r"cicatriz",
        r"alopecia", r"calvic", r"telog[êe]nico",
        r"\bcromon[íi]quia", r"ungueal\s+", r"\bonic[oh]",
        r"queilite(?!\s+act)", r"glossite",
        r"granuloma\s+anular", r"granuloma\s+facial",
        r"exantema", r"urticaria", r"eritema\s+nodos",
        r"l[úu]pus", r"esclerodermia", r"dermatomiosite",
        r"acantose\s+nigricans",
        r"micobacterio", r"tuberculose",
        r"telangiectasia\s+hemorr",
        r"queratodermia", r"hiperqueratose",
        r"epiderm",
        r"intertrigo", r"gangrena", r"fournier",
        r"acroquerat", r"poroquerat",
        r"\bestria\b", r"estrias\b",
        r"mal\s+perfurante", r"neuropat",
        r"hist[ió]l", r"histopat",
        r"\bfibroma\s+ungueal", r"fibroma\s+pendul",
        r"queilit", r"mucosite",
    ]),
]

# Heuristicas para extracao opcional de age/sex/fototipo
RX_AGE = re.compile(r"\b(\d{1,3})\s*(?:anos|a\.|ano)\b", re.IGNORECASE)
RX_FOTO = re.compile(r"\bfototipo[s]?\s*(I{1,3}|IV|V|VI|[1-6])\b", re.IGNORECASE)
RX_SEX = re.compile(r"\b(homem|masculin[oa]|sex[oa]\s+m[áa]?sc|paciente\s+masc|"
                    r"mulher|feminin[oa]|sex[oa]\s+fem|paciente\s+fem)", re.IGNORECASE)

# Localizacao anatomica em PT (mapeamento livre, nao categoriza ainda)
LOC_KEYWORDS = [
    "face", "rosto", "nariz", "orelha", "couro cabeludo", "fronte", "queixo",
    "pesco[çc]o", "t[óo]rax", "dorso", "abdome", "ombro",
    "bra[çc]o", "antebra[çc]o", "cotovelo", "m[ãa]o", "punho",
    "dedo", "palma", "unha",
    "perna", "coxa", "joelho", "tornozelo", "p[ée]", "planta", "calcanhar",
    "gl[úu]te", "n[áa]dega",
    "regi[ãa]o anterior", "regi[ãa]o posterior",
    "membro superior", "membro inferior",
]
RX_LOC = re.compile(r"\b(" + "|".join(LOC_KEYWORDS) + r")\b", re.IGNORECASE)

def classify(caption: str, nearby_text: str):
    """Retorna (dx, dx_original, localizacao, idade, sexo, fototipo, conf, justif)."""
    combined = (caption + " | " + nearby_text).lower()

    # Aplicar regras em ordem
    for dx_class, base_conf, patterns in RULES:
        for pat in patterns:
            m = re.search(pat, combined)
            if m:
                hit = m.group(0)
                # ajusta confianca: hit completo da caption (vs nearby_text)
                in_caption = hit in caption.lower()
                conf = base_conf if in_caption else max(base_conf - 0.1, 0.4)

                # Extracao opcional
                loc_m   = RX_LOC.search(caption + " " + nearby_text)
                age_m   = RX_AGE.search(caption + " " + nearby_text)
                foto_m  = RX_FOTO.search(caption + " " + nearby_text)
                sex_m   = RX_SEX.search(caption + " " + nearby_text)

                loc  = loc_m.group(1) if loc_m else ""
                age  = age_m.group(1) if age_m else ""
                foto = foto_m.group(1).upper() if foto_m else ""
                if sex_m:
                    s = sex_m.group(1).lower()
                    sex_norm = "male" if ("masc" in s or "homem" in s) else "female"
                else:
                    sex_norm = ""

                justif = f"Match regex '{pat}' -> '{hit}' (caption' if in_caption else 'context)"
                return (dx_class, hit, loc, age, sex_norm, foto, conf,
                        justif.replace("'", ""))

    # Nenhuma regra casou
    return ("", "", "", "", "", "", 0.0, "Nenhum termo conhecido casou")

def main():
    # Carregar piloto ja existente
    existing = {}
    if DEST.exists():
        with DEST.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[r["image_id"]] = r
        print(f"Piloto ja classificado: {len(existing)} linhas")

    # Carregar candidatos tumorais
    with SRC.open(encoding="utf-8") as f:
        cands = list(csv.DictReader(f))
    print(f"Candidatos tumorais: {len(cands)}")

    novos = 0
    for r in cands:
        if r["image_id"] in existing:
            continue
        dx, dx_orig, loc, age, sex, foto, conf, justif = classify(
            r["nearby_caption"], r["nearby_text"]
        )
        existing[r["image_id"]] = {
            "image_id": r["image_id"], "source_book": r["source_book"],
            "source_page": r["source_page"],
            "dx_inferido": dx, "dx_original_texto": dx_orig,
            "localizacao": loc, "idade": age, "sexo": sex, "fototipo": foto,
            "confianca": f"{conf:.2f}", "justificativa": justif,
        }
        novos += 1

    print(f"Novos classificados: {novos}")

    fieldnames = ["image_id","source_book","source_page","dx_inferido",
                  "dx_original_texto","localizacao","idade","sexo","fototipo",
                  "confianca","justificativa"]

    with DEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing.values():
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Estatisticas
    from collections import Counter
    all_rows = list(existing.values())
    dx_dist = Counter(r["dx_inferido"] for r in all_rows)
    print(f"\n=== Distribuicao FINAL (N={len(all_rows)}) ===")
    for k, v in sorted(dx_dist.items(), key=lambda x: -x[1]):
        print(f"  {k or '(vazio)':10s} {v:4d} ({100*v/len(all_rows):.1f}%)")

    ham_classes = {"akiec","bcc","bkl","df","mel","nv","vasc"}
    ham_count = sum(v for k, v in dx_dist.items() if k in ham_classes)
    print(f"\nTotal HAM10000-compatibvel: {ham_count}/{len(all_rows)} = {100*ham_count/len(all_rows):.1f}%")

    print(f"\nCSV: {DEST}")

if __name__ == "__main__":
    main()
