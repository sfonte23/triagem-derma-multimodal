# ICV — Auditoria do Projeto

> **Pacote consolidado para auditoria docente** do projeto de Iniciação Científica Voluntária (UFPI).
> Inclui código, plano metodológico, resultados quantitativos e amostras anonimizadas dos datasets.

---

## 📋 Sobre o projeto

Pesquisa multimodal de aprendizado profundo para apoio ao diagnóstico dermatológico na Atenção Básica. O modelo combina **CNN (EfficientNetB3) com metadados clínicos** (idade, sexo, localização) para classificar lesões cutâneas em 7 categorias do HAM10000.

**Autor:** Sérgio Fonte (UFPI)
**Coautores:** Martony Demes Silva, Keylla Maria Sá Urtiga Aita
**Orientação:** Prof. Responsável pelo protocolo de curadoria

## 🗂 Cronologia dos artigos

| Versão | Tema | Veículo | Status |
|---|---|---|---|
| **v1.0** | Abordagem multimodal CNN + classificadores clássicos sobre HAM10000 | MiNDS@SBCAS 2026 (JEMS #24749) | 🔴 Rejeitado |
| **v1.1** | Revisão respondendo reviewers + reforço metodológico (SMOTE, CV, Bootstrap) | CBIS'26 (XXI Congresso Brasileiro de Informática em Saúde) | 🟡 Pronto para submissão |
| **v2.0** | Validação em base brasileira (extraída dos livros Azulay 2015 + Glossário Ibero-Latino-Americano) | TBD após Fase 4 | 🟡 Fases 0-3 concluídas, aguardando curadoria médica |

## 📁 Estrutura do pacote

```
icv-auditoria/
├── README.md                          ← este arquivo
├── INSTRUCOES_REPRODUCAO.md           ← como reproduzir o que está documentado
├── requirements.txt                   ← dependências Python
├── .gitignore
│
├── docs/                              ← documentação completa
│   ├── PROJETO_GERAL.md               ← visão geral do projeto (CLAUDE.md adaptado)
│   ├── v1.0_README.md                 ← artigo v1.0 (JEMS rejeitado)
│   ├── v1.1_README.md                 ← artigo v1.1 (CBIS'26)
│   ├── v1.1_plano_revisao_cbis.md     ← checklist reviewer → mudança aplicada
│   ├── v2.0_README.md                 ← artigo v2.0 (base nacional)
│   ├── v2.0_plano.md                  ← plano detalhado em 6 fases
│   ├── v2.0_progresso.md              ← relatório executivo das fases 0-3
│   └── v2.0_mapeamento_permissivo.md  ← regras termo médico → 7 classes HAM10000
│
├── scripts/                           ← código-fonte completo
│   │── (treinamento + avaliação base HAM10000)
│   ├── train_code.py                  ← pipeline de treino CNN multimodal
│   ├── evaluate_local.py              ← avaliação oficial HAM10000
│   │── (artigo v1.1 — base curada Derm7pt+HAM)
│   ├── prepare_nova_base.py
│   ├── download_images.py
│   ├── evaluate_nova_base.py
│   ├── experimentos_revisao.py        ← SMOTE/CV/Bootstrap/t-SNE
│   ├── experimentos_revisao_continuacao.py
│   │── (artigo v2.0 — base nacional brasileira)
│   ├── extract_book_images.py         ← PyMuPDF extrai imagens dos PDFs
│   ├── filter_book_images.py          ← validador automático (sinaliza lixo)
│   ├── classify_tumor_candidates.py   ← classificador determinístico (mapeamento permissivo)
│   ├── consolidate_metadata.py        ← gera metadata_base_nacional.csv
│   │── (análises auxiliares)
│   ├── _analise_pesos_clinicos.py     ← heatmap dos pesos da 1ª camada
│   ├── _analise_pesos_finais.py       ← Jacobiano numérico + camada de fusão
│   └── _teste_pesos_clinicos.py       ← experimento de re-priorização
│
├── resultados/                        ← métricas, plots e tabelas
│   ├── v1.0_ham10000/                 ← matrizes, gráfico comparativo, CSV oficial
│   ├── v1.1_cbis2026/                 ← 12 figuras (SMOTE, CV, bootstrap, t-SNE) + CSVs
│   └── v2.0_base_nacional/            ← análise de pesos + experimento de re-priorização
│
└── dados_anonimizados/                ← amostras dos CSVs (sem reproduzir copyright)
    ├── v2.0_ai_inferred_amostra20.csv
    └── v2.0_metadata_base_nacional_amostra20.csv
```

## 🔬 O que está incluído neste pacote

✅ **Todo o código-fonte** (Python) — para auditoria de metodologia
✅ **Documentação completa** das decisões metodológicas e fases de trabalho
✅ **Resultados quantitativos** (CSVs e plots) de todas as 3 versões
✅ **Amostras anonimizadas** dos datasets construídos
✅ **Plano de revisão** mapeando cada apontamento dos reviewers JEMS para uma ação tomada
✅ **Análise dos pesos da rede** — evidência empírica sobre importância das features clínicas

## ❌ O que NÃO está incluído (e por quê)

- **Modelo `.keras` treinado (76 MB)** — excede limite do GitHub free. Pode ser regenerado via `scripts/train_code.py` (4–8h em GPU T4) ou solicitado por e-mail.
- **HAM10000 dataset (~3 GB)** — público, baixado automaticamente via `kagglehub` quando os scripts rodam.
- **PDFs dos livros (Azulay + Glossário)** — protegidos por copyright; uso interno acadêmico apenas. Professor pode usar os PDFs próprios para reproduzir a extração.
- **`medical-staff/metadata.xlsx`** — contém metadados clínicos curados por estagiário médico (dados sensíveis sob LGPD).
- **`data/base_nacional/images_raw/`** — 3.732 imagens extraídas dos livros (copyright).
- **Cache de embeddings `.npy`** — regenerável (~15 min em CPU).

## 📊 Resumo executivo dos resultados

### v1.0 — HAM10000 (Etapa 1)
| Modelo | Acurácia | F1 Macro | AUC OVR |
|---|---|---|---|
| Random Forest | 68,95% | 0,291 | 0,723 |
| **XGBoost** | 68,25% | 0,290 | **0,768** |
| SVM | 68,50% | 0,159 | 0,689 |
| CNN Multimodal | 66,95% | 0,120 | 0,730 |
| Naive Bayes | 16,67% | 0,113 | — |

**Achado:** CNN fim-a-fim colapsou para a classe majoritária (Nevus). Classificadores clássicos sobre embeddings recuperaram até 24% de recall em Melanoma.

### v1.1 — Reforço metodológico CBIS'26
- **SMOTE** sobre embeddings: recall Melanoma XGBoost 24,2% → 32,7%
- **5-fold CV estratificada**: XGBoost 0,685 ± 0,009 Acc / 0,797 ± 0,015 AUC
- **Bootstrap IC 95%**: XGBoost R_mel = 0,242 [0,186; 0,298]

### v2.0 — Base Nacional Brasileira
- **3.732 imagens** extraídas dos 2 livros
- **283 imagens HAM-compatíveis** distribuídas nas 7 classes (mineração dirigida por capítulos tumorais)
- Taxa de aproveitamento: **44,8%** (vs 16% em amostragem aleatória)

### Análise dos pesos da branch clínica (modelo HAM10000)
- Jacobiano numérico: **age** (2,58) > sex (1,58) > localization (1,18)
- Camada de fusão (peso por dim individual): Visual 1,14 ≈ Clínica 1,11 (razão 1,03×)
- **Conclusão:** o modelo dá peso individual equivalente a cada feature clínica e visual; a diferença total (4,1×) vem só do volume (16× mais features visuais).

## 🚀 Próximos passos do projeto

1. **Curadoria médica** das 283 imagens da base v2.0 (validar `dx`, preencher metadados clínicos quando possível, marcar exclusões)
2. **Fase 4**: avaliar o modelo HAM10000 sobre a base brasileira curada
3. **Fase 5**: comparativo HAM vs Brasil (degradação esperada por viés étnico-fenotípico)
4. **Fase 6**: rascunho do artigo v2.0 (venue TBD)

## 📬 Contato

- **Sérgio Fonte** — sfonte@axenya.com / sergio.fonte@ufpi.edu.br
- **Repositório de trabalho:** privado (este pacote é o subset compartilhado para auditoria)
- **Para reprodução completa**, ver `INSTRUCOES_REPRODUCAO.md`
