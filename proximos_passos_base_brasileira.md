# Próximos Passos — Validação em Base Brasileira

> Plano de **validação externa** do modelo treinado (HAM10000) em imagens brasileiras, para estimar a acurácia real de uso na Atenção Básica do SUS. Conecta-se ao "trabalho futuro" do artigo (validação com fototipos III–VI) e ao material arquivado da v2.0 (`docs/_arquivo/2.0_artigo_multimodal_base_nacional/`).

---

## 1. Por que fazer

O modelo foi treinado no **HAM10000** (Áustria/Austrália): fototipos I–II, imagens dermatoscópicas padronizadas. A população brasileira tem fototipos III–VI e, na Atenção Básica, as imagens muitas vezes são **fotos de celular**, não dermatoscopia. Há, portanto, um **desvio de domínio** (*domain shift*) esperado. Medir esse desvio é o que falta para sustentar (ou refutar) a viabilidade clínica no SUS — é o **padrão-ouro** de validação: testar numa população totalmente diferente da de treino, sem re-treinar (*zero-shot*).

## 2. O que encaixa direto

O pipeline campeão é **image-only** (visão isolada balanceada → *embeddings* → XGBoost). Ele **não precisa de metadados** — então uma base brasileira **só com imagens** já serve:

```
imagem BR → EfficientNetB3 (modelo nosso) → embedding 256-d → XGBoost (já treinado) → predição
```

Sem re-treinar nada. Se a base tiver metadados (idade/sexo/localização), dá para testar também a fusão clínica *one-hot* no classificador.

## 3. O que a base precisa ter

| Item | Necessário? | Para quê |
|---|---|---|
| Imagens (JPG/PNG) | ✅ obrigatório | extrair *embeddings* |
| Rótulo `dx` por imagem (validado por dermatologista) | ✅ **obrigatório para medir acurácia** | sem rótulo, só há predição, não avaliação |
| `lesion_id` (se houver várias fotos por lesão) | 🟡 desejável | evitar contagem inflada |
| Fototipo de Fitzpatrick (I–VI) | 🟡 desejável | estratificar I–II vs III–VI → testar viés étnico direto |
| Tipo de imagem (dermatoscópica vs foto de celular) | 🟡 anotar | interpretar a degradação |

**Formato sugerido do CSV de rótulos:** `image_id, dx[, lesion_id, fototipo, tipo_imagem]`, com `dx` em uma das 7 classes HAM (`akiec, bcc, bkl, df, mel, nv, vasc`).

## 4. Como rodar (script a criar: `scripts/evaluate_external.py`)

Esqueleto previsto (reaproveita `evaluate_grouped.py` e o modelo `.keras`):

1. Carregar `models/grouped/imageonlybalanced/modelo_imageonly_balanced_grouped.keras`.
2. Para cada imagem BR: redimensionar 320×320, normalizar [0,1], extrair *embedding* da camada `img_fused_256` (256-d).
3. Carregar o XGBoost **já treinado** sobre os *embeddings* do HAM (ou re-treinar o XGBoost só com os *embeddings* HAM e prever os BR).
4. Predizer e comparar com os rótulos BR:
   - acurácia, F1-Macro, **sensibilidade e especificidade por classe**;
   - aplicar o **ponto de operação** calibrado (limiar ~0,05 para "qualquer maligna") e ver se a sensibilidade de 85% se mantém;
   - se houver fototipo: **estratificar I–II vs III–VI**.
5. Gerar matriz de confusão BR + tabela comparativa HAM vs Brasil.

## 5. O que esperar e como interpretar

- **Cenário A — segura bem (sensibilidade ~85% nos brasileiros):** forte evidência de viabilidade como segunda opinião no SUS.
- **Cenário B — degrada (provável em pele escura / foto de celular):** quantifica o *gap* de domínio e justifica **fine-tuning** com dados nacionais. Também é resultado publicável e cientificamente honesto.
- **Cuidado com N pequeno:** algumas centenas de imagens dão resultado **indicativo**, não definitivo. Reportar intervalos de confiança (bootstrap).

## 6. Ética e dados

- Imagens de pacientes brasileiros são **dados sensíveis (LGPD)**; exigem base legal e, conforme a origem, parecer de **CEP** (Plataforma Brasil). Ver `SECURITY.md`.
- Não versionar imagens identificáveis no Git. Reportar apenas métricas agregadas.
- Se a base vier de livros/atlas (como a v2.0): **copyright** — uso acadêmico interno, sem redistribuição (vide `docs/_arquivo/2.0_...`).

## 7. Critério de "pronto"

- [ ] Base com ≥ 200 imagens rotuladas por dermatologista
- [ ] `scripts/evaluate_external.py` rodando e gerando métricas + matriz BR
- [ ] Tabela comparativa HAM10000 vs Brasil (Acc, F1, sensibilidade por classe maligna)
- [ ] (Se houver fototipo) análise estratificada I–II vs III–VI
- [ ] Discussão de degradação e recomendação (usar como está vs fine-tuning)
