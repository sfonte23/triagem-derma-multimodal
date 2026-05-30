# ICV — Projeto de Iniciação Científica Voluntária (UFPI)

Pesquisa multimodal de aprendizado profundo para apoio ao diagnóstico dermatológico na Atenção Básica. Combina CNN (EfficientNetB3) com metadados clínicos para classificar lesões cutâneas em 7 categorias (HAM10000).

---

## Estrutura do Projeto

```
ICV/
├── models/                          # Modelo treinado (.keras, 76 MB; gitignored)
├── data/                            # Datasets e cache (gitignored)
│   ├── embeddings_cache/            # Embeddings .npy do HAM10000 (regeneráveis)
│   ├── nova_base/                   # Base curada v1.1 (N=144, 55,6% HAM10000)
│   └── base_nacional/               # Base v2.0 extraída de livros (N=283 HAM-compat)
├── notebooks/
│   ├── colab/                       # Execução com GPU T4/L4 no Google Colab
│   └── local/                       # Execução local (CPU/GPU)
├── scripts/
│   ├── train_code.py                # Pipeline completo de treinamento
│   ├── evaluate_local.py            # Avaliação oficial HAM10000
│   ├── evaluate_nova_base.py        # Avaliação v1.1 (nova base curada)
│   ├── extract_book_images.py       # v2.0: extrai imagens dos PDFs (PyMuPDF)
│   ├── filter_book_images.py        # v2.0: sinaliza lixo (não exclui)
│   ├── classify_tumor_candidates.py # v2.0: classificador determinístico HAM
│   ├── consolidate_metadata.py      # v2.0: CSV final compatível v1.1
│   ├── _analise_pesos_*.py          # v2.0: análise dos pesos da branch clínica
│   ├── _teste_pesos_clinicos.py     # v2.0: experimento de re-priorização
│   └── convert_to_word.py           # Converte MD → DOCX (formato ABNT)
├── results/                         # Métricas e gráficos do HAM10000
├── medical-staff/                   # Metadados curados pelo estagiário médico (gitignored)
│   └── metadata.xlsx
├── docs/
│   ├── entregas/                    # Artigos, relatório, apresentação
│   └── reunioes/                    # Atas de reunião (formato AAAAMMDD.md)
└── venv/                            # Ambiente virtual Python (não versionar)
```

---

## Ambiente e Execução

### Setup local (Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Treinar modelo
```bash
python scripts/train_code.py
```
O script baixa o HAM10000 via `kagglehub`. Requer credenciais Kaggle em `~/.kaggle/kaggle.json`.

### Avaliar modelo (gera métricas e plots em `results/`)
```bash
python scripts/evaluate_local.py
```

### Notebooks no Colab
1. Abrir `notebooks/colab/01_CNN_Training_Multimodal.ipynb` no Google Colab
2. Configurar GPU: `Runtime → Change runtime type → T4 GPU`
3. Montar Google Drive para salvar o modelo
4. Rodar `02_Evaluation_and_Comparison.ipynb` após o treinamento

---

## Arquitetura do Modelo

**Dual-branch multimodal:**
- **Branch de imagem:** EfficientNetB3 pré-treinado (ImageNet), entrada 320×320
- **Branch clínica:** Dense layers sobre 3 atributos: `age` (Z-score), `sex` (0/1/2), `localization` (categórico)
- **Fusão:** Concatenação das embeddings → camadas densas → softmax 7 classes

**Função de perda:** Categorical Focal Loss (γ=2.0, α=0.75) — combate desbalanceamento de classes.

**Classes HAM10000:** `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`

---

## Resultados Oficiais (HAM10000 — Etapa 1)

| Algoritmo       | Accuracy | F1 Macro | AUC OVR |
|:----------------|:--------:|:--------:|:-------:|
| Random Forest   | 68.95%   | 0.291    | 0.723   |
| XGBoost         | 68.25%   | 0.290    | **0.768** |
| SVM             | 68.50%   | 0.159    | 0.689   |
| Multimodal CNN  | 66.95%   | 0.120    | 0.730   |
| Naive Bayes     | 16.67%   | 0.113    | —       |

**Achado crítico:** A CNN colapsou para predição da classe majoritária (Nevo). Classificadores clássicos sobre embeddings recuperaram até 24% de recall em Melanoma.

---

## Etapa 2 — v1.1 (Artigo CBIS'26) — ESTADO

Base curada usada no artigo v1.1 (`data/nova_base/metadata_curada.csv`, N=144). Scripts:
- `scripts/prepare_nova_base.py` — prepara a base curada
- `scripts/download_images.py` — baixa/copia imagens para `data/nova_base/images/`
- `scripts/evaluate_nova_base.py` — inferência, métricas, plots (Fix 1 + Fix 2 embutidos)
- `scripts/experimentos_revisao.py` + `experimentos_revisao_continuacao.py` — Ablation/SMOTE/CV/Bootstrap/t-SNE

**Resultados (pós-ajuste Fix 1+2):** XGBoost 44,44% Acc / 0,390 F1 Macro na nova base vs 68,25% / 0,290 no HAM10000.

**Limitação:** 55,6% das imagens da base v1.1 são do próprio HAM10000 (vide seção abaixo). Artigo v2.0 endereça essa limitação.

## Entregas (docs/entregas/)

Reorganizado por versão de artigo. Cada pasta tem README.md próprio explicando status e conteúdo.

| Pasta | Conteúdo | Status |
|---|---|---|
| `1.0_artigo_abordagem_multimodal/` | Artigo v1 original | 🔴 Rejeitado JEMS #24749 |
| `1.1_revisao_cibis2026/` | Artigo revisado para CBIS'26 | 🟡 Pronto p/ submissão |
| `2.0_artigo_multimodal_base_nacional/` | Validação em base nacional (Azulay + Glossário) | 🟡 Fases 0-3 concluídas, aguardando curadoria médica |
| `apresentacao_grupo_ufpi/` | Slides Beamer + roteiro do orador | ✅ Pronto |
| `protocolo_curadoria/` | Protocolos médicos (transversal) | ✅ Estável |
| `relatorio_parcial_ufpi/` | Relatório parcial PIBIC | ✅ Entregue |
| `revisao_bibliometrica/` | Revisão bibliométrica (Keylla) | 🟡 Em curso |

Embeddings cacheados em `data/embeddings_cache/` (gitignorado) — reaproveita extração para futuros experimentos.

---

## Artigo v2.0 — Base Nacional Brasileira (ESTADO ATUAL)

Construindo dataset brasileiro a partir de 2 livros didáticos (Azulay 2015 + Glossário Ibero-Latino-Americano) e avaliando o modelo HAM10000 nesse domínio.

**Estado em 2026-05-30:**
- ✅ **Fase 1 — Extração:** 3.732 imagens extraídas dos PDFs via PyMuPDF (`scripts/extract_book_images.py` + `filter_book_images.py` que sinaliza lixo sem excluir)
- ✅ **Fase 2 — Classificação:** **283 imagens HAM-compatíveis** identificadas via mineração dirigida por capítulos tumorais + classificador determinístico (`scripts/classify_tumor_candidates.py`). Taxa de aproveitamento: 44,8% (vs 16% em amostragem aleatória)
- ✅ **Fase 3 — Consolidação:** `data/base_nacional/metadata_base_nacional.csv` com schema compatível v1.1
- ⏳ **Fase 3.5 — Curadoria médica:** médico revisa as 283 linhas (validar `dx`, preencher metadados clínicos, marcar exclusões)
- ⏸ **Fase 4 — Avaliação:** após curadoria
- ⏸ **Fases 5-6:** comparação HAM vs Brasil + rascunho do artigo

**Distribuição das 283 imagens:**
mel 101, bcc 63, nv 48, bkl 39, akiec 15, df 14, vasc 3 (todas as 7 classes HAM10000 presentes).

**Detalhes em** [`docs/entregas/2.0_artigo_multimodal_base_nacional/README.md`](docs/entregas/2.0_artigo_multimodal_base_nacional/README.md) e [`progresso_fases_0a3.md`](docs/entregas/2.0_artigo_multimodal_base_nacional/progresso_fases_0a3.md).

**Decisões metodológicas importantes:**
- Copyright dos livros → base permanentemente gitignored em `data/base_nacional/`
- Sem retreinamento — modelo HAM10000 usado só como extrator de embeddings
- Curadoria assistida pelo Claude Code (sem API key separada)
- Imputação de faltantes só no pipeline final (CSV preserva vazios)
- `source_page` preservada no CSV → médico abre o PDF na página exata para validar

---

## Análise dos Pesos da Branch Clínica (achado para iterações futuras)

Análise feita com o modelo HAM10000 original. Scripts em `scripts/_analise_pesos_clinicos.py`, `_analise_pesos_finais.py`, `_teste_pesos_clinicos.py`. Plots em `docs/entregas/2.0_artigo_multimodal_base_nacional/results/`.

### Importância efetiva das 3 features clínicas (Jacobiano numérico)

| Feature | Jacobiano \|dY/dX\| | Ranking |
|---|---|---|
| **age** | **2,58** | 1º |
| sex | 1,58 | 2º |
| localization | 1,18 | 3º |

`age` é **2,18× mais importante** que `localization` no output da branch clínica.

### Camada de fusão (Visual 128 dims vs Clínica 8 dims)

| | L2 total | L2 por dimensão |
|---|---|---|
| Visual | 12,91 | 1,14 |
| Clínica | 3,14 | 1,11 |
| Razão V/C | 4,1× | **1,03×** |

**Cada feature clínica tem peso individual equivalente a cada feature visual.** A diferença total vem só do volume (16× mais features visuais).

### Experimento: re-priorizar localization sem retreinar (FALHOU)

Hipótese testada: aumentar peso de `localization` (2×) e reduzir `age`/`sex` (0,5×) melhoraria desempenho. **Todas as configurações degradaram**:

| Config | AUC OVR | ΔAUC vs Baseline |
|---|---|---|
| Baseline (1,1,1) | 0,730 | — |
| Pedido (0.5, 0.5, 2.0) | 0,576 | −15,4 pp |
| Sem clínica (0, 0, 0) | 0,474 | −25,6 pp (**abaixo de chute!**) |

**Conclusão:** o modelo encontrou um equilíbrio fino durante o treino que não tolera reweighting pós-treino. Para realmente re-priorizar features precisaria de retreinamento (one-hot encoding, gradient regularization, etc.).

**Implicação para o v2.0:** a base brasileira terá ~99% das imagens sem `age` extraível → equivale a "Sem age" no experimento (−5pp AUC). É a degradação esperada por ausência de metadados (sem contar viés étnico-fenotípico, que é adicional).

---

## ⚠️ Limitação reconhecida da base v1.1 (medical-staff/metadata.xlsx)

A base curada do v1.1 (`data/nova_base/metadata_curada.csv`, N=144) tem **55,6% de imagens do próprio HAM10000** (`ham_ISIC_*`). O modelo já "viu" essas durante o treino → contamina a avaliação de generalização.

**Solução em curso:** o artigo v2.0 endereça essa limitação usando dataset 100% externo (livros brasileiros, N=283 candidatos).

---

## Outros candidatos para validação independente (não usados ainda)

### Dataset UFES/Dermay (investigado por Arthur)
- **Prós:** Imagens brasileiras, fototipos III-VI Fitzpatrick
- **Contras:** Provavelmente sem metadados clínicos estruturados (`age`, `sex`, `localization`). Branch clínica seria desabilitada → testa só CNN
- **Verificar com Arthur** o schema disponível

### Derm7pt nativas (parcialmente no v1.1)
- 64 imagens nativas Derm7pt (não-HAM) já integradas
- Expandir é o caminho mais rápido para ampliar N sem contaminação

### Fotos clínicas próprias (coleta prospectiva)
- Mais robusto, mas requer aprovação de CEP. Protocolo em `docs/entregas/protocolo_curadoria/`

---

## Convenções

- **random_state=42** em todos os splits e classificadores
- Split estratificado 80/20 realizado ANTES da extração de features (sem data leakage)
- Normalização Z-score de `age` idêntica entre treino e teste
- Atas de reunião: `docs/reunioes/DDMMAAAA.md`
- Arquivos Word gerados automaticamente via `scripts/convert_to_word.py` (formato ABNT)

---

## Equipe

- **Sergio Fonte** — desenvolvedor principal, pipeline CNN, scripts
- **Arthur** — investigação de datasets brasileiros (Dermay/UFES)
- **Keylla** — revisão bibliométrica, submissão para periódico
- **Supervisora** — Prof. responsável pelo protocolo de curadoria e orientação metodológica
