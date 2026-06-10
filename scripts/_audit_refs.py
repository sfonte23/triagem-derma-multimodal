import re, sys
from pathlib import Path
art = Path("docs/tcc/final/artigo")
tex = (art / "main.tex").read_text(encoding="utf-8")
bib = (art / "references.bib").read_text(encoding="utf-8")

cited = set()
for m in re.findall(r"\\cite\{([^}]+)\}", tex):
    for k in m.split(","):
        cited.add(k.strip())
defined = set(re.findall(r"@\w+\{([^,]+),", bib))

print("CITADAS no texto :", len(cited))
print("DEFINIDAS no bib :", len(defined))
print()
orfas = sorted(defined - cited)
quebradas = sorted(cited - defined)
print("Definidas mas NUNCA citadas (orfas):", orfas or "NENHUMA")
print("Citadas mas SEM entrada no bib (quebradas):", quebradas or "NENHUMA")
print()
print("Lista citada:", sorted(cited))
