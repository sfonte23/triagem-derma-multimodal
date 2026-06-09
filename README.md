# Triagem Dermatológica Multimodal — Pacote de Auditoria

Pacote **mínimo e auto-suficiente** para auditar **todos os números** do TCC *"Abordagem Multimodal baseada em Aprendizado Profundo para Apoio ao Diagnóstico Dermatológico na Atenção Básica"* (CEAD/UFPI), na sua versão **revisada** (correção de *data leakage* por lesão).

Qualquer pessoa pode, a partir deste repositório, **reproduzir as tabelas e figuras do artigo** rodando dois scripts em CPU — sem GPU e sem baixar o HAM10000 — porque os *embeddings* já extraídos estão incluídos.

---

## ⚡ Reproduzir todos os números (CPU, ~5 min)

```bash
pip install -r requirements.txt
python scripts/evaluate_grouped.py     # desenho fatorial 2x2, CV agrupada, bootstrap, ablação, fusão clínica
python scripts/operating_point.py      # ponto de operação clínico (qualquer maligna) + curva + calibração
python scripts/_comparison_figure.py   # figura comparativa dos pipelines
```
Saídas em `resultados/` (CSVs + PNGs).

---

## 🗂️ Estrutura

```
.
├── scripts/                     # análise (CPU) + geradores dos notebooks
│   ├── evaluate_grouped.py      #   ★ reproduz o 2x2, CV, bootstrap, ablação, fusão clínica
│   ├── operating_point.py       #   ★ ponto de operação clínico + curva + calibração
│   ├── _comparison_figure.py    #   figura comparativa dos 4 pipelines
│   ├── _threshold_tuning.py     #   varredura de limiar (prévia)
│   └── _build_colab_notebook*.py#   geradores dos 4 notebooks
├── notebooks/                   # 4 notebooks Colab que treinam as CNNs (split por lesão)
│   ├── 01_CNN_Training_Multimodal_GROUPED.ipynb
│   ├── 02_CNN_ImageOnly_GROUPED.ipynb
│   ├── 03_CNN_ImgBalanced_GROUPED.ipynb
│   └── 04_CNN_ImageOnly_Balanced_GROUPED.ipynb
├── data/embeddings_cache/grouped/   # embeddings .npy (256-d e 1536-d) + rótulos + lesion_id + manifestos
│   ├── multimodal/  imageonly/  imgbalanced/  imageonlybalanced/
├── resultados/                  # CSVs e figuras de referência
└── requirements.txt
```

> **Modelos `.keras` não estão no Git** (≈800 MB). Os *embeddings* incluídos reproduzem **todos os números reportados**. Para re-extrair os *embeddings* a partir das imagens (reprodução de ponta a ponta), rode os notebooks no Colab (T4) — eles baixam o HAM10000, treinam as CNNs sob a partição por lesão e exportam os `.npy` aqui presentes. Modelos treinados disponíveis sob solicitação.

---

## 🔬 O que está sendo auditado

A versão revisada corrige um **vazamento de dados (*data leakage*)**: o HAM10000 tem ~7.470 lesões para 10.015 imagens; a divisão por imagem deixava fotos da mesma lesão em treino e teste. Aqui a divisão e a validação cruzada são **agrupadas por `lesion_id`** (`GroupShuffleSplit` + `StratifiedGroupKFold`). Cada notebook prova ausência de vazamento via `assert` e registra em `split_manifest*.json` (com `leakage_check` zerado).

**Desenho fatorial** (visão isolada × multimodal, com/sem balanceamento na imagem) + onde fundir o metadado (rede vs. classificador) + ponto de operação para triagem.

### Principais resultados (teste *held-out* por lesão, XGBoost+SMOTE)

| Pipeline | F1-Macro | AUC OVR |
|---|---|---|
| Multimodal (clínico na rede) | 0,290 | 0,763 |
| Visão isolada | 0,313 | 0,815 |
| + balanço de imagem | 0,373 | 0,853 |
| **+ clínico one-hot no classificador (campeão)** | **0,409** | **0,883** |

**Ponto de operação (qualquer maligna: mel+bcc+akiec):** a 85% de sensibilidade → especificidade 71%, PPV 41% (recall mel 78%, bcc 92%, akiec 95%).

---

## ✅ Verificação de integridade do split

Os `data/embeddings_cache/grouped/*/split_manifest*.json` contêm `leakage_check` (interseções de lesão entre treino/val/teste, todas **zero**) e `dist_teste` (idêntica entre os 4 modelos → mesma partição → comparação justa).

```bash
python -c "import json,glob; [print(p, json.load(open(p))['leakage_check']) for p in glob.glob('data/embeddings_cache/grouped/*/split_manifest*.json')]"
```
