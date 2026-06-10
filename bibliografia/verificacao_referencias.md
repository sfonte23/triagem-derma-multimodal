# Verificação das Referências — Pente Fino

> Auditoria de **todas** as referências citadas no artigo (`docs/tcc/final/artigo/references.bib`), verificadas via **Crossref** (busca por título, independente do `.bib`) e busca na web (jun/2026). Motivada por uma citação fabricada detectada na revisão.

## Resumo

- **12 referências verificadas e corretas.**
- **1 fabricada** (`marcano2021`) → **removida** do artigo e do `.bib`.
- **1 com metadados errados** (`daneshjou2022`) → **corrigida** (revista/título/DOI).
- Após a correção: **13 referências, todas reais e citadas** (auditoria `scripts/_audit_refs.py`: 13 citadas = 13 definidas, 0 órfãs, 0 quebradas).

---

## Tabela de verificação

| Chave | Status | Verificação (real) | DOI / fonte |
|---|---|---|---|
| esteva2017 | ✅ correta | Nature 542(7639):115–118, 2017 | 10.1038/nature21056 |
| tschandl2018 | ✅ correta | Scientific Data 5:180161, 2018 (HAM10000) | 10.1038/sdata.2018.161 |
| tan2019 | ✅ correta | EfficientNet, ICML 2019 | arXiv:1905.11946 |
| lin2017 | ✅ correta | Focal Loss, ICCV 2017 | arXiv:1708.02002 |
| chawla2002 | ✅ correta | SMOTE, JAIR 16:321–357, 2002 | 10.1613/jair.953 |
| johnson2019 | ✅ correta | J. Big Data 6(1), 2019 | 10.1186/s40537-019-0192-5 |
| gessert2020 | ✅ correta | MethodsX 7:100864, 2020 | 10.1016/j.mex.2020.100864 |
| tang2022 | ✅ correta | FusionM4Net, Medical Image Analysis 76:102307, 2022 | 10.1016/j.media.2021.102307 |
| **marcano2021** | ❌ **FABRICADA** | Nenhum paper do Marcano-Cedeño com esse título existe no Crossref. O DOI derivado do vol/nº/artigo declarado (Sensors 21(9):3230 → 10.3390/s21093230) aponta para um artigo de **cerâmica piezoelétrica de cimento** (Ding et al.), nada a ver. | — (removida) |
| **daneshjou2022** | ⚠️ **corrigida** | Era citada como "NPJ Digital Medicine, *Disparate performance of AI dermatology tools…*". O real é **Science Advances** 8(32):eabq6147, 2022, *Disparities in dermatology AI performance on a diverse, curated clinical image set* | 10.1126/sciadv.abq6147 |
| adamson2018 | ✅ correta | JAMA Dermatology 154(11):1247–1248, 2018 | 10.1001/jamadermatol.2018.2348 (PubMed 30073260) |
| chen2016 | ✅ correta | XGBoost, ACM SIGKDD 2016:785–794 | 10.1145/2939672.2939785 |
| breiman2001 | ✅ correta | Random Forests, Machine Learning 45(1):5–32, 2001 | 10.1023/A:1010933404324 |
| inca2023 | ✅ real | INCA, *Estimativa 2023: Incidência de Câncer no Brasil*, Rio de Janeiro | relatório institucional (inca.gov.br) |

---

## Detalhe das duas correções

### ❌ marcano2021 — referência fabricada (removida)
- **Como foi detectada:** o `.bib` afirmava *Sensors 21(9):3230, 2021*. A MDPI usa DOI determinístico (`10.3390/s` + volume + nº com 2 dígitos + nº do artigo) → `10.3390/s21093230`. Esse DOI, no Crossref, é "Cement-Based Piezoelectric Ceramic Composites for Sensing Elements" (Ding et al.) — não é dermatologia.
- **Confirmação independente:** busca por título no Crossref (`query.bibliographic`) não retorna nenhum artigo do Marcano-Cedeño sobre CDSS de lesões de pele; os melhores matches são de sepse/anestesiologia (irrelevantes).
- **Ação:** removida do `references.bib` e as duas citações no `main.tex` (Introdução e Trabalhos Relacionados) foram ajustadas — a frase agora cita apenas `gessert2020` e `tang2022` (ambos verificados).

### ⚠️ daneshjou2022 — metadados errados (corrigida)
- O trabalho de Daneshjou sobre disparidade de desempenho de IA em peles de cor é **real e relevante**, mas o `.bib` tinha **título e revista errados** (constava "NPJ Digital Medicine"). 
- Corrigido para: *Disparities in dermatology AI performance on a diverse, curated clinical image set*, **Science Advances** 8(32):eabq6147, 2022, DOI 10.1126/sciadv.abq6147.

---

## Como reproduzir esta auditoria

```bash
python scripts/_audit_refs.py        # confere citadas vs definidas (órfãs/quebradas)
# verificação por título no Crossref (exemplo):
#   https://api.crossref.org/works?query.bibliographic=<titulo>&rows=1
```

## Acesso aos PDFs (validação local)

Os PDFs foram baixados para `docs/tcc/bibliografia/pdfs/` **apenas para validação local** das afirmações do artigo. A pasta é **gitignored** (`**/bibliografia/pdfs/`) — os arquivos **não** vão para os repositórios públicos (direitos autorais). Verificou-se o **título da 1ª página** de cada PDF para garantir que é o artigo certo (dois downloads iniciais vieram de PMC IDs errados que eu havia chutado, e foram **descartados** após essa checagem — reforçando a necessidade do pente fino).

| Chave | PDF local | Fonte |
|---|---|---|
| tschandl2018 | ✔ baixado | Nature Scientific Data (OA) |
| gessert2020 | ✔ baixado | Europe PMC PMC7150512 (OA) |
| johnson2019 | ✔ baixado | SpringerOpen (OA) |
| chawla2002 | ✔ baixado | arXiv 1106.1813 |
| tan2019 | ✔ baixado | arXiv 1905.11946 |
| lin2017 | ✔ baixado | arXiv 1708.02002 |
| chen2016 | ✔ baixado | arXiv 1603.02754 |
| breiman2001 | ✔ baixado | stat.berkeley.edu (cópia do autor) |
| daneshjou2022 | ✔ baixado | Europe PMC PMC9374341 (OA) |
| inca2023 | ✔ baixado | inca.gov.br (relatório público) |
| esteva2017 | 🔗 link | Nature (pago); PMC8382232 é o artigo certo, mas o download é bloqueado — DOI 10.1038/nature21056 |
| adamson2018 | 🔗 link | JAMA Dermatology (pago) — DOI 10.1001/jamadermatol.2018.2348 |
| tang2022 | 🔗 link | Elsevier, Medical Image Analysis (pago) — DOI 10.1016/j.media.2021.102307 |

**10 de 13 disponíveis localmente; 3 restritos** (acessar via CAPES/instituição com os DOIs acima).
