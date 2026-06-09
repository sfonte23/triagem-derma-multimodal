# Guia Completo do Pipeline — Estudo para a Defesa

> Documento de estudo: explica **cada conceito, sigla e argumento** usado neste trabalho, do problema clínico ao último hiperparâmetro. Objetivo: dar a você **propriedade total** para apresentar e responder a banca. Leia na ordem; cada seção assume a anterior.

**Sumário**
1. O problema clínico
2. O dataset HAM10000
3. Visão geral do pipeline (o mapa)
4. Redes neurais e CNNs
5. A arquitetura do nosso modelo (peça por peça)
6. Treinamento: perda, otimizador e regularização
7. O colapso preditivo (o vilão do trabalho)
8. Pré-processamento e codificação dos dados
9. Data leakage por lesão e a correção
10. Por que desacoplar: CNN extratora + classificador clássico
11. Os classificadores clássicos
12. Desbalanceamento: SMOTE vs balanceamento na imagem
13. Onde fundir o metadado clínico
14. Métricas (cada uma, com interpretação clínica)
15. Validação estatística (CV e bootstrap)
16. Limiar de decisão e ponto de operação clínico
17. Calibração de probabilidade
18. O desenho fatorial 2×2
19. Como ler nossos resultados
20. Limitações e honestidade científica
21. Glossário-relâmpago de siglas

---

## 1. O problema clínico

**Câncer de pele** é a neoplasia (tumor) mais comum no Brasil. Há tipos benignos e malignos. Os que importam para triagem:
- **Melanoma (`mel`)** — o mais **letal**; origina-se dos melanócitos (células do pigmento). Minoritário em número, mas mata pela rápida metástase. Detecção precoce → sobrevida > 95%.
- **Carcinoma basocelular (`bcc`)** — maligno, comum, raramente metastático, mas precisa tratar.
- **Ceratose actínica (`akiec`)** — lesão **pré-maligna** (pode virar carcinoma); merece acompanhamento.
- Benignos: **Nevo (`nv`, "pinta")**, **Ceratose benigna (`bkl`)**, **Dermatofibroma (`df`)**, **Lesão vascular (`vasc`)**.

**Atenção Básica / SUS:** primeiro nível de atendimento (UBS — Unidade Básica de Saúde), com médicos **generalistas**, sem dermatoscópio nem treino dermatoscópico. Dermatologista é escasso fora das capitais.

**Triagem (*screening*):** decidir *quem encaminhar* ao especialista — não dar o diagnóstico final. Numa triagem, **falso negativo** (mandar um melanoma para casa) é catastrófico; **falso positivo** (encaminhar um benigno) é tolerável. Por isso priorizamos **sensibilidade**.

**SADC / CDSS:** Sistema de Apoio à Decisão Clínica — software que sugere, mas **não substitui** o médico. Nosso sistema é uma **segunda opinião**.

**Fototipos de Fitzpatrick (I–VI):** escala de cor da pele pela reação ao sol. I–II = pele clara (queima fácil); V–VI = pele negra. O HAM10000 tem majoritariamente I–II → **viés étnico** quando aplicado ao Brasil.

---

## 2. O dataset HAM10000

**HAM10000** = "Human Against Machine with 10000 training images". 10.015 imagens **dermatoscópicas** (tiradas com dermatoscópio: lente + luz polarizada que vê estruturas sob a pele), 7 classes, com metadados (idade, sexo, localização).

Dois fatos críticos:
1. **Desbalanceamento severo:** `nv` (Nevo) é ~67% das imagens; `df` e `vasc` somam < 3%. Razão de ~67:1 entre a maior e a menor classe.
2. **Múltiplas imagens por lesão:** ~7.470 **lesões** únicas para 10.015 imagens. A mesma lesão aparece em várias fotos (ângulos/momentos). Isso é a raiz do *data leakage* (Seção 9).

---

## 3. Visão geral do pipeline (o mapa)

```
                          [TREINO, uma vez]
imagem 320×320 ──► EfficientNetB3 ──► embedding (vetor de 256 números)
                                              │
metadado clínico (one-hot) ───────────────────┤  (fundido AQUI, no classificador)
                                              ▼
                                   XGBoost (+ SMOTE / balanceamento)
                                              │
                                              ▼
                              P(maligna) ≥ limiar?  ──► encaminhar ou não
```

Ideia central: a CNN **extrai características** (vira a imagem num vetor), e um **classificador clássico** toma a decisão. A CNN é treinada **uma vez** e nunca re-treinada para a decisão.

---

## 4. Redes neurais e CNNs

**Rede neural:** função matemática com milhões de "pesos" ajustáveis. Aprende ajustando os pesos para minimizar o erro nos exemplos de treino.

**CNN (Rede Neural Convolucional):** rede especializada em imagens. Usa **filtros (convoluções)** que deslizam pela imagem detectando padrões locais — primeiras camadas pegam bordas/cores, camadas profundas pegam formas e estruturas complexas (analogia: o olho detecta bordas → contornos → objeto).

**Por que pré-treinar (Transfer Learning):** treinar uma CNN do zero exige milhões de imagens. Em vez disso, parte-se de uma rede já treinada no **ImageNet** (14 milhões de imagens, 1.000 categorias do dia a dia). Ela já "sabe ver" bordas e texturas; só ajustamos para dermatoscopia. Isso é **aprendizado por transferência**.

**Embedding (representação latente):** a saída de uma camada intermediária da rede — um vetor de números (no nosso caso, 256) que resume o que a rede "entendeu" da imagem. Imagens parecidas têm *embeddings* próximos. É a "impressão digital" matemática da lesão.

---

## 5. A arquitetura do nosso modelo (peça por peça)

Arquitetura **multimodal de duas vias** (*dual-branch*), com **fusão tardia** (*late fusion* = cada modalidade é processada separada e só no fim se juntam):

**Via de imagem:**
- **EfficientNetB3** — CNN da família EfficientNet (Google, 2019). "B3" é o 3º nível de tamanho. A família usa *compound scaling* (escala profundidade, largura e resolução juntas) → alta performance com poucos parâmetros (~12 milhões). Pré-treinada no ImageNet.
- **GAP (Global Average Pooling):** transforma o "mapa" de saída da CNN (uma grade de ativações) num único vetor, tirando a média de cada canal. Saída: **1536 dimensões** (o "bruto" visual).
- **Dense(128) + Dropout:** camada densa reduz para 128; Dropout regulariza (ver §6).

**Via clínica:**
- **MLP (Perceptron Multicamadas):** rede simples de camadas densas (sem convolução), processa os 3 metadados (idade, sexo, localização) → vetor de 8 dimensões.

**Fusão:**
- Concatena (junta) a via de imagem com a via clínica → camada **`fused_dense_1`** de **256 dimensões** (é daqui que extraímos o *embedding* multimodal).
- Depois: Dense(128) → **Softmax(7)**.

**Softmax:** função final que transforma números brutos em **probabilidades que somam 1** (uma por classe). O modelo "escolhe" a classe de maior probabilidade. *Problema:* sob desbalanceamento, o Softmax aprende a sempre prever a classe majoritária (ver §7).

---

## 6. Treinamento: perda, otimizador e regularização

**Função de perda (*loss*):** mede o erro; o treino tenta minimizá-la.
- **Cross-entropy (entropia cruzada):** perda padrão para classificação — penaliza prever baixa probabilidade na classe certa.
- **Focal Loss** (a que usamos): variação da cross-entropy que **dá mais peso aos exemplos difíceis** e menos aos fáceis. Dois parâmetros:
  - **γ (gama) = 2,0:** fator de foco. γ alto faz exemplos fáceis (já bem classificados) quase não contarem → o treino "foca" nos difíceis.
  - **α (alfa) = 0,75:** peso de balanceamento entre classes.
  - *Por que usamos:* tentar combater o desbalanceamento já no treino. (Spoiler: não bastou — §7.)

**Otimizador Adam:** algoritmo que ajusta os pesos a cada passo. Combina momento + taxa adaptativa; é o padrão moderno.

**Learning rate (taxa de aprendizado) = 10⁻⁴:** tamanho do passo de ajuste. Muito alto → instável; muito baixo → lento. 10⁻⁴ é conservador, bom para *fine-tuning*.

**Batch (lote) = 32:** quantas imagens a rede vê por passo antes de atualizar os pesos.

**Época (*epoch*):** uma passada completa por todo o treino. Treinamos ~25–35 épocas.

**Early stopping (parada antecipada):** para o treino quando a métrica de validação para de melhorar (evita *overfitting* e desperdício). "Paciência" = quantas épocas esperar sem melhora antes de parar.

**Regularização (evitar *overfitting* = decorar o treino):**
- **Dropout (0,5):** durante o treino, "desliga" aleatoriamente 50% dos neurônios de uma camada a cada passo → força a rede a não depender de um único caminho.
- **Batch Normalization:** normaliza as ativações dentro da rede → treino mais estável e rápido.

**Data augmentation (aumento de dados):** transformações aleatórias na imagem de treino, *on-the-fly* (a cada época, uma versão diferente), para a rede ficar robusta. As que usamos: rotação, espelhamento (horizontal/vertical), variações de brilho/contraste e matiz/saturação, desfoque gaussiano, e oclusão (apagar retângulos aleatórios). **Importante:** augmentation só no **treino**; nunca na avaliação.

---

## 7. O colapso preditivo (o vilão do trabalho)

**Colapso preditivo (*predictive collapse*):** sob desbalanceamento extremo, a rede aprende o atalho de **prever sempre a classe majoritária** (`nv`). Resultado: **acurácia global alta** (acerta os 67% de `nv`) mas **recall ~0 nas classes raras** — exatamente as clinicamente críticas (melanoma).

**Por que acontece (explicação técnica):** durante o treino por retropropagação, o otimizador minimiza a perda média. Como `nv` domina o volume, os **gradientes** (sinais de ajuste) da classe majoritária dominam; os das minoritárias, mesmo amplificados pela Focal Loss, não são fortes o bastante para "deslocar" a fronteira de decisão da camada Softmax. A rede acomoda-se num **mínimo local** de alta acurácia e baixa sensibilidade.

Esse é o achado que motiva **desacoplar** a extração da decisão (§10).

---

## 8. Pré-processamento e codificação dos dados

**Imagem:** redimensionada para **320×320 pixels**, valores normalizados para o intervalo **[0,1]** (dividir por 255).

**Idade:** **normalização Z-Score** = (valor − média) / desvio-padrão → fica centrada em 0 com escala 1, evitando que a idade (escala 0–85) domine numericamente. Idades ausentes (< 1% da base) preenchidas com a **mediana**.

**Sexo:** codificado numericamente (masculino=0, feminino=1).

**Localização anatômica — o ponto delicado:**
- **Codificação ordinal** (`cat.codes`): cada local vira um inteiro (face=3, dorso=8…). **Problema:** sugere uma ordem/magnitude falsa (que dorso > face, e que a "distância" é 5) — sem sentido para variável categórica. Foi assim no ramo neural (uma limitação que reconhecemos).
- **Codificação one-hot** (a correta): cria uma coluna 0/1 por categoria (15 colunas, exatamente uma "1"). Não impõe ordem. Usamos one-hot na fusão pelo classificador → o metadado passou a **ajudar** (§13).

---

## 9. Data leakage por lesão e a correção

**Data leakage (vazamento de dados):** quando informação do conjunto de teste "vaza" para o treino, inflando as métricas (o modelo parece melhor do que é).

**Nosso caso:** o HAM tem várias imagens da **mesma lesão**. Se dividirmos treino/teste **por imagem** (sorteio aleatório de fotos), fotos de uma lesão caem no treino e outras da **mesma lesão** caem no teste. O modelo "reconhece" a lesão já vista → métrica inflada e enganosa.

**Correção:**
- **`lesion_id`:** identificador único da lesão (não da foto). Agrupamos por ele.
- **GroupShuffleSplit:** divide treino/teste **por grupo** (lesão), garantindo que **todas as fotos de uma lesão fiquem do mesmo lado**. Usamos 70% treino / 10% validação / 20% teste.
- **`random_state=42`:** semente aleatória fixa → a divisão é sempre a mesma (reprodutível).
- **StratifiedGroupKFold:** versão da validação cruzada (§15) que respeita os grupos (lesão) E tenta manter a proporção de classes.
- **Prova:** cada notebook checa com `assert` que **nenhuma lesão** aparece em mais de um conjunto, e registra `leakage_check: 0` no `split_manifest.json`.

**Efeito da correção:** as métricas caíram um pouco (recall de melanoma do XGBoost+SMOTE: 32,7% → ~27%), confirmando que o vazamento inflava. Honesto e esperado.

---

## 10. Por que desacoplar: CNN extratora + classificador clássico

A CNN "colapsa" na **camada de decisão (Softmax)**, mas **não na extração** — os *embeddings* ainda guardam informação que separa as classes. Prova: classificadores clássicos sobre os mesmos *embeddings* recuperam recall que a Softmax não conseguia.

- **Capacidade de representação** (extrair características) ≠ **capacidade de decisão** (traçar a fronteira).
- A Softmax traça **um hiperplano global rígido** que, sob desbalanceamento, é empurrado para "esmagar" as minoritárias.
- Árvores de decisão (XGBoost) fazem **partições locais** e conseguem isolar pequenos grupos de melanoma no espaço dos *embeddings*.

Por isso: CNN só como **extratora**; decisão com **clássico**.

---

## 11. Os classificadores clássicos

Todos operam sobre os *embeddings* (vetores de 256 ou 1536 números), não sobre pixels.

- **XGBoost (Extreme Gradient Boosting):** o nosso campeão. *Boosting* = treina muitas **árvores de decisão** em sequência, cada uma corrigindo os erros da anterior. Rápido, robusto, ótimo com tabelas de *features*. Argumento principal: **`n_estimators=100`** (100 árvores).
- **Random Forest (Floresta Aleatória):** *bagging* = muitas árvores independentes em subamostras aleatórias; decisão por **voto majoritário**. Estável.
- **SVM (Máquina de Vetores de Suporte):** acha o **hiperplano** que separa as classes com a maior **margem**. Com **kernel RBF** (função de base radial) lida com fronteiras curvas. `probability=True` faz calibração interna (lento). É o gargalo de tempo no nosso pipeline.
- **Naive Bayes (Gaussiano):** aplica o **Teorema de Bayes** assumindo independência entre *features* (hipótese "ingênua"). Simples e rápido; serve de *baseline* (sempre o pior aqui).

---

## 12. Desbalanceamento: SMOTE vs balanceamento na imagem

Duas formas de combater o 67:1, em **espaços diferentes**:

**SMOTE (Synthetic Minority Over-sampling Technique)** — no **espaço latente** (dos *embeddings*):
- Cria amostras sintéticas das classes minoritárias **interpolando** entre exemplos reais vizinhos (não copia: gera pontos no meio do caminho entre dois embeddings reais da mesma classe).
- Argumento **`k_neighbors=5`:** usa os 5 vizinhos mais próximos para interpolar.
- **Risco — "maldição da dimensionalidade":** em 256 dimensões, as distâncias ficam homogêneas e os dados, esparsos; o SMOTE pode criar pontos sintéticos em regiões "vazias", pouco representativas. Daí compararmos com a alternativa abaixo.

**Balanceamento no espaço de imagem** — na **origem**:
- Antes de treinar a CNN, **reamostra** o conjunto de treino para um **alvo comum por classe** (ex.: ~1000): **sobreamostra** as minoritárias (repete imagens) e **subamostra** a majoritária. Como o *augmentation* é *on-the-fly*, cada cópia da minoritária recebe uma transformação diferente → variabilidade real, não cópia idêntica.
- **Achado:** isso **supera** o SMOTE (AUC 0,82 → 0,85), porque corrige o desbalanceamento onde ele nasce (a representação aprendida), não depois.

---

## 13. Onde fundir o metadado clínico

Descoberta importante do trabalho: **o metadado não é inútil — o que importa é ONDE fundi-lo.**

- **Na rede (ramo clínico, ordinal):** sob desbalanceamento + codificação ordinal, o sinal clínico se **dissipa** (a dominância de gradiente da classe majoritária e o encoding ruim "estragam" a contribuição). Resultado: o multimodal fica **pior** que a visão isolada.
- **No classificador (one-hot):** concatenar o metadado **one-hot** ao *embedding* de imagem antes do XGBoost **agrega** (AUC 0,853 → 0,883; F1 0,373 → 0,409). Árvores tratam nativamente *features* heterogêneas (numéricas + categóricas) e preservam o sinal.
- O **clínico sozinho** (3 *features*) tem AUC 0,744 (acima do acaso 0,5) → confirma que **tem sinal**; o problema era o ponto/forma de fusão.

---

## 14. Métricas (cada uma, com interpretação clínica)

Definições com base na **matriz de confusão** (linhas = verdade, colunas = predito): VP (verdadeiro positivo), VN, FP (falso positivo), FN (falso negativo).

- **Acurácia** = (VP+VN)/total. *Enganosa em base desbalanceada:* prever sempre `nv` dá 67%. **Não use como métrica principal.**
- **Recall / Sensibilidade** = VP/(VP+FN). "Dos melanomas reais, quantos peguei?" **A métrica mais importante para triagem** (FN = melanoma perdido).
- **Especificidade** = VN/(VN+FP). "Dos benignos, quantos corretamente não encaminhei?" Mede a carga de falsos alarmes.
- **Precisão / PPV (Valor Preditivo Positivo)** = VP/(VP+FP). "Dos que sinalizei, quantos eram realmente câncer?" PPV baixo = muitos alarmes falsos.
- **F1-Macro:** média harmônica de precisão e recall, calculada **por classe** e depois média **simples** entre classes (cada classe pesa igual, mesmo `nv` sendo 30× maior que `df`). Penaliza ignorar minoritárias → boa métrica primária.
- **AUC OVR (Área sob a curva ROC, One-versus-Rest):** mede a **capacidade de ranqueamento** — quão bem o modelo separa cada classe das demais, **independente do limiar**. 0,5 = acaso; 1,0 = perfeito. "OVR" = calcula uma classe contra todas as outras e tira a média. Robusta ao desbalanceamento.
- **Matriz de confusão:** a tabela completa; a diagonal são acertos. Revela *onde* o modelo confunde (ex.: melanoma previsto como `nv`).

---

## 15. Validação estatística (CV e bootstrap)

**Validação cruzada k-fold (CV):** divide os dados em k partes (k=5); em cada rodada, 4 treinam e 1 testa; repete 5×. A métrica final é a **média ± desvio-padrão**. Desvio baixo (< 3 pp) = resultado **estável**, não acidente de uma divisão. Usamos **StratifiedGroupKFold** (respeita lesão + proporção de classes).

**Bootstrap (IC 95%):** reamostra **com reposição** o conjunto de teste 1.000 vezes, recalcula a métrica em cada → distribui 1.000 valores. O **intervalo de confiança de 95%** são os percentis 2,5% e 97,5%. Se os ICs de dois modelos **não se sobrepõem**, a diferença é estatisticamente significativa.

---

## 16. Limiar de decisão e ponto de operação clínico

**Limiar (*threshold*):** por padrão, o classificador decide "positivo" se a probabilidade ≥ **0,5**. Mas 0,5 é arbitrário e ruim para triagem desbalanceada.

**Calibrar o limiar = escolher o ponto de operação.** Baixando o limiar (ex.: sinalizar se P(maligna) ≥ 0,05), pegamos **mais** casos suspeitos → **sobe a sensibilidade**, **cai a especificidade**. É o trade-off **correto** em triagem (preferimos errar para o lado seguro).

**"Qualquer maligna":** em vez de só melanoma, sinalizamos se **P(mel)+P(bcc)+P(akiec) ≥ limiar**. Clinicamente mais completo (rastreia câncer de pele, não só melanoma) e estatisticamente melhor (BCC e akiec são mais fáceis → puxam a sensibilidade).

**Curva ROC:** plota sensibilidade (eixo y) vs. 1−especificidade (eixo x) para todos os limiares. A **AUC** é a área sob ela. Cada ponto da curva é um limiar possível — escolhemos o que dá ~85% de sensibilidade.

**Nosso ponto de operação (campeão):** limiar ~0,05 → **85% de sensibilidade** (recall mel 78%, bcc 92%, akiec 95%), especificidade 71%, PPV 41% (encaminha ~40% das lesões). A acurácia cai — **de propósito** (acurácia não é o objetivo em triagem).

---

## 17. Calibração de probabilidade

**Calibração (isotônica/Platt):** ajusta as probabilidades para que "0,08" signifique de fato 8% de chance real. Testamos a isotônica.
- **Achado honesto:** como é uma transformação **monotônica** (preserva a ordem), **não muda** o trade-off sensibilidade×especificidade nem a AUC — só **renomeia** o valor do limiar, tornando-o interpretável. Reportamos isso por transparência (não promete ganho que não existe). Calibração ≠ melhora de discriminação; discriminação vem do modelo (AUC).

---

## 18. O desenho fatorial 2×2

Para isolar o efeito de cada decisão, treinamos **4 variantes** sob a **mesma partição por lesão**:

| | sem balanço de imagem | com balanço de imagem |
|---|---|---|
| **com clínico (multimodal)** | nb01 | nb03 |
| **só imagem (visão isolada)** | nb02 | nb04 |

Isso permite ler **dois efeitos independentes**: (a) o balanço de imagem (colunas) e (b) a presença do clínico na rede (linhas) — e a interação entre eles. É a forma rigorosa de dizer "o que causou o quê".

---

## 19. Como ler nossos resultados

Números-âncora (teste *held-out* por lesão, XGBoost+SMOTE):

| Pipeline | F1-Macro | AUC OVR |
|---|---|---|
| Multimodal (clínico na rede) | 0,290 | 0,763 |
| Visão isolada | 0,313 | 0,815 |
| + balanço de imagem | 0,373 | 0,853 |
| **+ clínico one-hot no classificador (campeão)** | **0,409** | **0,883** |

Leitura em uma frase: *"corrigir o vazamento honesta os números; balancear na imagem é o que mais ajuda; o metadado só soma se fundido no classificador; e calibrando o limiar a ferramenta atinge 85% de sensibilidade para câncer de pele — útil como segunda opinião, não como diagnóstico autônomo."*

---

## 20. Limitações e honestidade científica

- **Recall de melanoma (78% no ponto de 85%)** é o menor entre as malignas → não substitui o especialista.
- **PPV 41%** → ~40% das lesões encaminhadas; carga alta para a regulação.
- **Viés étnico:** HAM é eurocêntrico (fototipos I–II) → pode degradar em pele negra (daí a base brasileira, §doc próximos passos).
- **Domínio:** dermatoscopia ≠ foto de celular da Atenção Básica.
- **Acurácia global é métrica enganosa** em base desbalanceada — repita isso na defesa; é o fio condutor do trabalho.

---

## 21. Glossário-relâmpago de siglas

| Sigla | Significado curto |
|---|---|
| CNN | Rede Neural Convolucional (vê imagens) |
| EfficientNetB3 | CNN pré-treinada usada como extratora |
| ImageNet | Base gigante de pré-treino (transfer learning) |
| GAP | Global Average Pooling (vira mapa em vetor 1536-d) |
| MLP | Perceptron Multicamadas (rede densa simples) |
| Embedding | Vetor que resume a imagem (256-d) |
| Softmax | Camada que vira números em probabilidades |
| Focal Loss | Perda que foca nos exemplos difíceis (γ, α) |
| Adam | Otimizador (ajusta os pesos) |
| Dropout / BatchNorm | Regularização / estabilização do treino |
| Data augmentation | Transformações aleatórias nas imagens de treino |
| Colapso preditivo | Prever sempre a classe majoritária |
| lesion_id | Identificador da lesão (agrupar para evitar vazamento) |
| GroupShuffleSplit | Divisão treino/teste por grupo (lesão) |
| StratifiedGroupKFold | Validação cruzada por grupo + estratificada |
| XGBoost / RF / SVM / NB | Classificadores clássicos |
| SMOTE | Sobreamostragem sintética (interpola minoritárias) |
| one-hot | Codificação categórica sem ordem (uma coluna por categoria) |
| Recall / Sensibilidade | % de positivos reais detectados |
| Especificidade | % de negativos reais corretamente liberados |
| PPV / Precisão | % de alarmes que eram realmente positivos |
| F1-Macro | Média (por classe) de precisão×recall |
| AUC OVR | Capacidade de separação, independente de limiar |
| ROC | Curva sensibilidade × (1−especificidade) |
| CV | Validação cruzada (média ± desvio) |
| Bootstrap / IC95% | Reamostragem para intervalo de confiança |
| Limiar / threshold | Corte de probabilidade para decidir |
| SADC / CDSS | Sistema de Apoio à Decisão Clínica |
| SUS / UBS | Sistema Único de Saúde / Unidade Básica |
| Fototipo (Fitzpatrick) | Escala I–VI de cor da pele |

---

> **Dica de defesa:** se travar, volte ao fio condutor — *"acurácia engana em base desbalanceada; por isso medimos sensibilidade; por isso corrigimos o vazamento, balanceamos na imagem, fundimos o clínico no lugar certo e calibramos o limiar."* Tudo no trabalho serve a essa frase.
