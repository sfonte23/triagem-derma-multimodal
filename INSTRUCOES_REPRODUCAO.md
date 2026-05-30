# Instruções de Reprodução

> Este documento descreve como reproduzir cada etapa do projeto a partir do código fornecido.
> Dependendo da etapa, podem ser necessários **arquivos externos** (modelo treinado, PDFs dos livros) que não foram incluídos por restrições de tamanho ou copyright.

## ⚙️ Setup inicial (necessário para qualquer etapa)

### 1. Criar ambiente Python (Python 3.10–3.12)

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configurar credencial Kaggle

Para baixar o HAM10000 automaticamente via `kagglehub`:

1. Acesse https://kaggle.com → Account → "Create New API Token"
2. Baixe o `kaggle.json`
3. Coloque em `~/.kaggle/kaggle.json` (Linux/Mac) ou `C:\Users\<seu_usuario>\.kaggle\kaggle.json` (Windows)

## 🎯 Reprodução por etapa

### Etapa 0 — Treinamento da CNN multimodal (v1.0)

**Pré-requisitos:**
- GPU NVIDIA com ≥ 8 GB VRAM (testado em T4 do Colab)
- Tempo: 4–8 horas

```bash
python scripts/train_code.py
```

**Outputs:**
- `models/modelo_multimodal_final.keras` (~76 MB)
- Métricas em `results/`

**Atalho:** se você só quer auditar a metodologia sem treinar, posso te enviar o `.keras` pré-treinado por e-mail (76 MB).

### Etapa 1 — Avaliação oficial no HAM10000 (v1.0)

**Pré-requisitos:**
- Modelo treinado (`models/modelo_multimodal_final.keras`)
- HAM10000 já baixado em cache via `kagglehub`

```bash
python scripts/evaluate_local.py
```

**Outputs esperados** (já presentes em `resultados/v1.0_ham10000/`):
- `metricas_oficiais_pibic.csv`
- 5 matrizes de confusão (XGBoost, RF, SVM, NB, CNN multimodal)

### Etapa 2 — Artigo v1.1 (revisão CBIS'26)

#### 2.1 Preparar base curada (Derm7pt + HAM)

**Pré-requisitos:**
- `medical-staff/metadata.xlsx` (não incluído — dados sensíveis LGPD)
- Imagens Derm7pt baixadas localmente (instrução manual ou via Kaggle)

```bash
python scripts/prepare_nova_base.py
python scripts/download_images.py
```

#### 2.2 Avaliação + experimentos de reforço metodológico

```bash
# Avaliação com Fix 1 (tradução de localização) + Fix 2 (Bayesian Prior Correction)
python scripts/evaluate_nova_base.py

# Experimentos: SMOTE / 5-fold CV / Bootstrap IC 95% / t-SNE
python scripts/experimentos_revisao.py
python scripts/experimentos_revisao_continuacao.py
```

**Outputs esperados** (já presentes em `resultados/v1.1_cbis2026/`):
- 12 figuras (matrizes, SMOTE comparativo, CV, bootstrap, t-SNE)
- CSVs: `smote_results.csv`, `cv_5fold_results.csv`, `bootstrap_ci_results.csv`

### Etapa 3 — Artigo v2.0 (base nacional brasileira)

#### 3.1 Extrair imagens dos livros

**Pré-requisitos:**
- PDFs dos livros em `material_base/`:
  - Azulay — Dermatologia 6ª ed. 2015 (78 MB)
  - Glossário Ibero-Latino-Americano de Dermatologia 6ª ed.
- **NÃO incluídos no pacote** (copyright). Use seus próprios PDFs.

```bash
python scripts/extract_book_images.py
python scripts/filter_book_images.py
```

**Outputs:**
- `data/base_nacional/images_raw/` — ~3.732 imagens PNG
- `data/base_nacional/raw_extraction.csv` — metadata + flags de lixo

#### 3.2 Classificar candidatos tumorais

```bash
# Filtra páginas dos capítulos tumorais (Azulay p580-700, Glossário p313-560)
# e aplica mapeamento permissivo via regex sobre legendas
python scripts/classify_tumor_candidates.py
python scripts/consolidate_metadata.py
```

**Output:**
- `data/base_nacional/metadata_base_nacional.csv` — N=283 imagens HAM-compatíveis

Veja `dados_anonimizados/v2.0_metadata_base_nacional_amostra20.csv` para o formato esperado.

#### 3.3 Análise dos pesos da branch clínica

**Pré-requisito:** modelo `.keras` treinado.

```bash
python scripts/_analise_pesos_clinicos.py  # heatmap 1ª camada
python scripts/_analise_pesos_finais.py    # Jacobiano + camada de fusão
python scripts/_teste_pesos_clinicos.py    # experimento de re-priorização
```

**Outputs** (já presentes em `resultados/v2.0_base_nacional/`):
- `heatmap_pesos_branch_clinica.png`
- `pesos_finais_branch_clinica.png`
- `teste_pesos_clinicos.png` + `teste_pesos_clinicos.csv`

## 🔍 O que você pode auditar sem rodar nenhum código

Mesmo sem o modelo treinado, você pode auditar:

1. **Metodologia completa** — ver `docs/v2.0_plano.md` e `docs/v1.1_plano_revisao_cbis.md`
2. **Decisões metodológicas** — ver `docs/PROJETO_GERAL.md` (seções "Convenções" e "Decisões metodológicas")
3. **Mapeamento de termos médicos** — ver `docs/v2.0_mapeamento_permissivo.md`
4. **Resultados quantitativos** — ver todos os PNGs e CSVs em `resultados/`
5. **Lógica de cada script** — código está comentado e organizado por fase
6. **Pareceres e revisão** — `docs/v1.1_plano_revisao_cbis.md` mapeia cada apontamento JEMS → mudança aplicada

## ⚠️ Limitações de reprodução

- **HAM10000**: dataset público, mas exige conta Kaggle gratuita
- **Modelo treinado**: ~76 MB, não cabe no GitHub free; pode ser solicitado
- **Livros (v2.0)**: copyright; uso interno acadêmico apenas
- **Imagens médicas curadas**: removidas do pacote por sensibilidade LGPD

## 📬 Em caso de dúvida

Sérgio Fonte — sfonte@axenya.com / sergio.fonte@ufpi.edu.br

Para acesso ao modelo treinado, ao dataset curado completo, ou para reproduzir a Etapa 3 com os PDFs originais, entre em contato diretamente.
