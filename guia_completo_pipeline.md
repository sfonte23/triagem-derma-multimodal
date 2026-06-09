# Guia Completo do Pipeline — Estudo para a Defesa

> Documento de estudo: explica **cada conceito, sigla e argumento** usado neste trabalho, do problema clínico ao último hiperparâmetro — sempre com **analogia do dia a dia + desenho + exemplo**, para qualquer leigo entender. Objetivo: dar a você **propriedade total** para apresentar e responder a banca. Leia na ordem; cada seção assume a anterior.

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
14. Métricas (cada uma, com exemplo numérico)
15. Validação estatística (CV e bootstrap)
16. Limiar de decisão e ponto de operação clínico
17. Calibração de probabilidade
18. O desenho fatorial 2×2
19. Como ler nossos resultados
20. Limitações e honestidade científica
21. Glossário-relâmpago de siglas

---

## 1. O problema clínico

**A ideia em uma frase:** ajudar uma UBS sem dermatologista a decidir **quem mandar para o especialista**, olhando a foto de uma lesão de pele.

**Câncer de pele** é o tumor mais comum no Brasil. Há tipos benignos (inofensivos) e malignos (perigosos):

```
   MALIGNAS (queremos PEGAR)              BENIGNAS (queremos LIBERAR)
   ├─ mel   Melanoma   ☠ o mais letal     ├─ nv    Nevo ("pinta")  — 67% dos casos
   ├─ bcc   Carcinoma basocelular         ├─ bkl   Ceratose benigna
   └─ akiec Ceratose actínica (pré-maligna)├─ df    Dermatofibroma
                                          └─ vasc  Lesão vascular
```

**Por que melanoma assusta:** é minoritário em número, mas mata rápido (metástase). Pego cedo → sobrevida > 95%. Pego tarde → fatal. Por isso o erro que mais tememos é **deixar passar um melanoma**.

**Analogia da triagem (fio condutor do trabalho):** pense num **detector de fumaça**. Você prefere que ele dispare às vezes sem incêndio (alarme falso, chato) do que **ficar quieto** num incêndio real (catastrófico). Na triagem de câncer é igual:
- **Falso negativo** (mandar um melanoma para casa) = o incêndio que o alarme não pegou → **inaceitável**.
- **Falso positivo** (encaminhar um benigno) = alarme falso → **tolerável**.
- Por isso priorizamos **sensibilidade** (pegar os doentes), não a acurácia total.

**Glossário clínico desta seção:**
- **Atenção Básica / SUS / UBS:** o primeiro nível de atendimento (postos de saúde), com médicos **generalistas**, sem dermatoscópio. É onde a maioria dos brasileiros entra no sistema.
- **Triagem (*screening*):** filtrar *quem* precisa do especialista — **não** dar o diagnóstico final.
- **SADC / CDSS:** Sistema de Apoio à Decisão Clínica — software que **sugere**, não substitui o médico. O nosso é uma **segunda opinião**.
- **Fototipos de Fitzpatrick (I–VI):** escala de cor da pele. I–II = clara; V–VI = negra. Guarde isso: nossa base de treino é quase toda I–II → **viés** ao aplicar no Brasil.

---

## 2. O dataset HAM10000

**O que é:** uma coleção pública de **10.015 fotos dermatoscópicas** (tiradas com **dermatoscópio** — uma lente com luz polarizada que enxerga estruturas sob a pele), rotuladas nas 7 classes acima, com 3 metadados por foto: **idade, sexo e localização** no corpo. Nome: *"Human Against Machine with 10000 training images"*.

**Dois fatos que mandam em tudo neste trabalho:**

**(a) Desbalanceamento severo.** Imagine 100 lesões da base:

```
nv   ███████████████████████████████████████████████████████████████████  ~67
bkl  ███████████                                                           ~11
mel  ███████                                                                ~7
bcc  █████                                                                  ~5
akiec███                                                                    ~3
vasc █                                                                      ~1
df   █                                                                      ~1
```
A maior classe (`nv`) é ~**67×** a menor (`df`). Um modelo preguiçoso que **chuta sempre "nv"** já acerta 67% — e é exatamente esse atalho perigoso que combatemos (§7).

**(b) Várias fotos da MESMA lesão.** Há ~**7.470 lesões** únicas para 10.015 imagens — a mesma verruga aparece em 2–3 ângulos/momentos:

```
lesão #A031  →  foto1.jpg   foto2.jpg   foto3.jpg     ← MESMA lesão, 3 fotos
lesão #A032  →  foto4.jpg
```
Guarde isso: é a origem do **vazamento de dados** (§9). Se as fotos da mesma lesão se espalharem entre treino e teste, o modelo "cola na prova".

---

## 3. Visão geral do pipeline (o mapa)

### A analogia que resume tudo

Imagine uma **triagem em duas pessoas**:

1. **Um técnico que só olha a foto** da lesão. Ele não conversa com o paciente, não vê a ficha. Ele apenas observa a imagem e preenche uma **planilha com 256 medidas** descrevendo o que viu (cor, textura, bordas, padrões…). Ele faz isso sempre do mesmo jeito, treinado uma vez. → **Essa é a CNN.** A planilha de 256 números é o **embedding**.
2. **Um médico decisor** que pega essa planilha de 256 medidas **e também** a ficha do paciente (idade, sexo, local da lesão) e, juntando tudo, decide: "suspeito → encaminhar" ou "tranquilo". → **Esse é o XGBoost** (o classificador).

O ponto-chave do trabalho: **o técnico (CNN) só olha a imagem**; os dados clínicos entram **só na mesa do médico decisor (XGBoost)**, não antes.

### O desenho do que fizemos (pipeline campeão)

```
   ┌─────────────────── TREINADO UMA VEZ ───────────────────┐
   │                                                         │
 FOTO 320×320 ─►  CNN (EfficientNetB3)  ─►  EMBEDDING         │   "o técnico que só vê a foto"
 (só pixels)        (só imagem!)            [256 números]     │
   └─────────────────────────────────────────────┬───────────┘
                                                  │
 FICHA CLÍNICA ──► vira colunas 0/1 (one-hot) ────┤  ← entra SÓ AQUI
 (idade, sexo,                                    │
  localização)                                    ▼
                                        XGBoost  (+ balanceamento)   "o médico decisor"
                                                  │
                                                  ▼
                                  P(maligna) ≥ limiar?  ─► encaminhar / não encaminhar
```

**Resumo em uma linha:** a CNN **transforma a imagem num vetor de 256 números** (extrai características); o clínico é anexado **depois**; e um **classificador clássico** decide. A CNN é treinada **uma única vez** e nunca re-treinada para a decisão.

> ⭐ **A campeã usa imagem E dados clínicos — ela é multimodal.** Foi a configuração que performou melhor (AUC 0,883 / F1 0,409). A diferença para a versão multimodal que falhou **não é** *se* usa o clínico, e sim **onde**: aqui o clínico entra na **mesa do decisor (XGBoost), em formato *one-hot***, e não dentro da rede. Sempre que este guia disser **"visão isolada"**, isso se refere **apenas à CNN** (que só olha a imagem) — o **pipeline completo continua usando imagem + clínico**.

---

### 3.1 A CNN trata dados clínicos? (Não — e isso foi de propósito)

No pipeline **final/campeão**, a CNN é usada **unicamente para extrair os embeddings da imagem**. Ela **nunca recebe** idade, sexo ou localização. Veja o que entra e sai de cada peça:

| Peça | O que ENTRA | O que SAI |
|---|---|---|
| **CNN** (EfficientNetB3) | só os **pixels** da foto (320×320×3) | **embedding**: 256 números |
| **XGBoost** (decisor) | embedding (256) **+** clínico *one-hot* (~18 colunas) | probabilidade de cada classe |

> **Por que separar?** Nós **testamos** a outra forma — uma CNN "multimodal" que processava a imagem **e** o clínico juntos, lá dentro da rede. Resultado: ficou **pior** (o sinal clínico se perdia sob o desbalanceamento). Quando movemos o clínico para **fora da rede**, direto no decisor, ele passou a **ajudar**. Por isso, no campeão, a CNN é 100% visual. (Detalhe completo na §13.)

Em uma frase para a banca: *"A CNN é só os olhos — ela vira a foto em números. Quem decide, juntando esses números com a ficha do paciente, é o XGBoost."*

---

### 3.2 O que é um *embedding*? (com exemplo concreto)

Um **embedding** é simplesmente uma **lista de números** (um *vetor*) que a CNN gera para **resumir** a imagem. No nosso caso, **256 números**. Cada número representa uma "característica visual" abstrata que a rede aprendeu sozinha (não tem nome legível como "borda escura" — é matemático), e o conjunto funciona como um **código de barras / impressão digital** da lesão.

A propriedade mágica: **lesões parecidas geram listas de números parecidas**; lesões diferentes geram listas distantes.

```
 Foto de um MELANOMA   ─► CNN ─►  [ 0.12, -1.45,  0.98, 0.03, ... , 0.07 ]   (256 números)
 Foto de OUTRO melanoma ─► CNN ─► [ 0.10, -1.38,  1.02, 0.05, ... , 0.09 ]   ← PARECIDO (perto)
 Foto de um NEVO (pinta) ─► CNN ─► [ -0.9,  0.20, -0.50, 1.30, ... ,-0.8 ]   ← DIFERENTE (longe)
```

**Três formas de entender (escolha a que te agrada):**
- **Ficha de medidas:** é como descrever uma pessoa por 256 medidas (altura, peso, etc.) — só que aqui as "medidas" são 256 traços visuais que a própria rede inventou.
- **Coordenadas num mapa:** pense num "mapa de lesões" de 256 dimensões. Cada foto vira um **ponto** nesse mapa. Melanomas caem numa região, nevos em outra. O decisor aprende a traçar as fronteiras desse mapa.
- **Código de barras:** cada lesão ganha um código de 256 números; fotos da mesma "cara" recebem códigos semelhantes.

**Por que isso é útil?** Porque trabalhar com 256 números é muito mais fácil (e leve) do que com a foto inteira (320×320×3 = 307.200 valores). A CNN faz o trabalho pesado de "ver" uma vez; depois, qualquer classificador simples decide rápido em cima dos 256 números — inclusive dá para anexar a ficha clínica e re-treinar o decisor **sem nunca mais tocar na rede pesada**.

---

## 4. Redes neurais e CNNs

**Rede neural — analogia:** imagine milhões de "botões" (os **pesos**) que controlam uma função. No começo estão aleatórios e a rede chuta tudo errado. O treino vai **girando os botões** aos pouquinhos para errar menos nos exemplos. É só isso: ajustar botões para minimizar erro.

**CNN (Rede Neural Convolucional) — como ela "vê":** é uma rede especializada em imagens. Ela usa **filtros** que deslizam pela foto procurando padrões, em camadas que vão do simples ao complexo:

```
 foto → [bordas, cores] → [texturas, cantos] → [formas: rede de vasos, pintas] → "isto parece melanoma"
        camadas rasas       camadas médias        camadas profundas
```
Igual ao olho humano: primeiro percebe contornos, depois junta em objetos.

**Transfer Learning (aprendizado por transferência) — analogia:** ninguém aprende a dirigir do zero ao trocar de carro — você já sabe dirigir, só se adapta ao carro novo. A nossa CNN já "aprendeu a ver" em **14 milhões de fotos do dia a dia (ImageNet)**: ela já sabe bordas, texturas, formas. Nós só a **adaptamos** para dermatoscopia (não precisamos de milhões de fotos de pele). Por isso ela funciona com "só" 10 mil imagens.

**Embedding (de novo, agora no contexto da rede):** é a saída de uma camada intermediária — o vetor de 256 números (§3.2) que resume o que a rede entendeu da imagem.

---

## 5. A arquitetura do nosso modelo (peça por peça)

A rede que **treinamos** tem **duas vias** (*dual-branch*) e **fusão tardia** (*late fusion* = cada via processa sua entrada separada, e elas só se juntam no fim). Desenho:

```
  VIA DE IMAGEM
  foto 320×320 ─► EfficientNetB3 ─► GAP (1536) ─► Dense(128) ─┐
                                                              ├─► concatena ─► fused_dense_1 (256) ─► Softmax(7)
  VIA CLÍNICA                                                 │            (EMBEDDING)
  idade/sexo/local ─► MLP ─► (8) ──────────────────────────────┘
```

Peça por peça:
- **EfficientNetB3:** a CNN (família EfficientNet, Google 2019). "B3" = 3º tamanho da família. Usa *compound scaling* (cresce profundidade, largura e resolução juntas) → muita performance com ~12 milhões de "botões". Vem pré-treinada no ImageNet.
- **GAP (Global Average Pooling):** a CNN cospe uma "grade" de ativações; o GAP tira a **média** de cada canal e devolve **um vetor de 1536 números** (o resumo visual bruto). Analogia: tirar a "nota média" de cada detector.
- **Dense(128):** camada que comprime 1536 → 128 (destila o essencial).
- **MLP (Perceptron Multicamadas):** rede simples (sem convolução) que processa os 3 metadados → 8 números.
- **fused_dense_1 (256):** junta imagem + clínico → **é daqui que sai o embedding de 256** que usamos depois.
- **Softmax(7):** a "boca" da rede, que dá uma probabilidade para cada uma das 7 classes (somam 100%). O modelo escolhe a maior.

> ⚠️ **Ligação com a campeã:** essa rede multimodal completa (com a via clínica dentro) é a que **falhou** mais (clínico se perde no treino). Na **campeã**, usamos a versão **só-imagem** dessa rede como extratora (pegamos o embedding visual) e levamos o clínico para o XGBoost. Mesma "espinha", clínico no lugar certo.

**Softmax — por que é o ponto fraco:** sob desbalanceamento, essa camada final aprende a sempre apostar em `nv` (§7). É por isso que trocamos a "boca" (Softmax) por um classificador clássico.

---

## 6. Treinamento: perda, otimizador e regularização

Treinar = **girar os botões** (§4) para minimizar o erro. Aqui estão as peças, cada uma com analogia:

**Função de perda (*loss*) — "o quanto erramos":** um número que mede a distância entre o que a rede previu e a verdade. O treino tenta **diminuí-lo**.
- **Cross-entropy:** a perda padrão — pune prever probabilidade baixa na classe certa.
- **Focal Loss** (a nossa): cross-entropy turbinada que **foca nos exemplos difíceis**. Analogia: um professor que **para de gastar tempo** com a matéria que o aluno já domina e **insiste** no que ele erra.
  - **γ (gama) = 2,0:** o "quanto focar". γ alto → exemplos fáceis quase não contam.
  - **α (alfa) = 0,75:** peso para equilibrar classes.
  - *Por que usamos:* tentar resolver o desbalanceamento já no treino. **Spoiler:** não bastou (§7) — daí o resto do trabalho.

**Otimizador Adam — "como giramos os botões":** o algoritmo que decide a direção e o tamanho do ajuste a cada passo. Analogia: **descer uma montanha na neblina** — você sente a inclinação sob os pés e dá um passo ladeira abaixo. Adam é uma forma esperta de dar esses passos.

**Learning rate (taxa de aprendizado) = 10⁻⁴ — "o tamanho do passo":** passo grande demais → você pula o vale (instável); pequeno demais → demora uma eternidade. 10⁻⁴ é um passo curto e seguro (bom para *fine-tuning*).

**Batch (lote) = 32:** quantas fotos a rede olha antes de girar os botões uma vez. Analogia: corrigir 32 provas e só então ajustar a aula.

**Época (*epoch*):** uma passada por **todas** as fotos de treino. Fizemos ~25–35. Analogia: revisar o livro inteiro uma vez.

**Early stopping (parada antecipada):** parar quando a nota na **validação** para de melhorar — evita "decorar" (overfitting) e poupa tempo. "Paciência" = quantas épocas esperar sem melhora antes de desistir.

**Regularização — evitar "decorar" (*overfitting*):** overfitting é quando o aluno **decora as respostas** em vez de aprender; vai bem no simulado e mal na prova real.
- **Dropout (0,5):** a cada passo, "desliga" 50% dos neurônios ao acaso. Analogia: um **grupo de estudo onde metade falta aleatoriamente** — força todos a saberem a matéria, ninguém depende de um só "gênio".
- **Batch Normalization:** mantém os números dentro da rede numa escala estável → treino mais rápido e suave.

**Data augmentation (aumento de dados):** mostrar a mesma foto **levemente alterada** a cada época (girada, espelhada, mais clara/escura, borrada, com um pedaço tapado). Analogia: estudar a mesma pessoa em **fotos com filtros e ângulos diferentes** para reconhecê-la em qualquer situação. Vale só no **treino** — nunca na prova (avaliação).

---

## 7. O colapso preditivo (o vilão do trabalho)

**Analogia:** numa prova de múltipla escolha onde **67% das respostas são "C"**, um aluno preguiçoso descobre que **marcar "C" em tudo** já garante 67%. Ele "passa" sem saber a matéria. A rede faz o mesmo: aprende a **chutar sempre `nv`** (a classe de 67%).

**Como isso aparece numa matriz de confusão (desenho):**

```
              PREVISTO
            nv    mel   ...
 R  nv   [ 1320    2  ... ]   ← acerta quase todo nevo
 E  mel  [  219    4  ... ]   ← dos 223 melanomas, acerta só 4 (!!)
 A  ...
 L
```
**Resultado:** acurácia global **alta** (acerta os 67% de `nv`) e recall de melanoma **perto de zero**. Clinicamente é um desastre — o que parece bom (acurácia) esconde o que importa (pegar melanoma).

**Por que acontece (técnico, em 1 parágrafo):** no treino, o otimizador minimiza a perda **média**. Como `nv` domina o volume, os **gradientes** (os "empurrões" que giram os botões) da classe gigante dominam; os empurrões das classes raras, mesmo amplificados pela Focal Loss, são fracos demais para mover a fronteira da Softmax. A rede "estaciona" num ponto cômodo de alta acurácia e baixa sensibilidade — um **mínimo local**.

➡️ É esse vilão que motiva tudo: **desacoplar** a extração da decisão (§10), **balancear** (§12) e **calibrar o limiar** (§16).

---

## 8. Pré-processamento e codificação dos dados

Antes de entrar no modelo, os dados são "arrumados":

**Imagem:** redimensionada para **320×320 pixels** e os valores de cor divididos por 255 → ficam entre **0 e 1** (redes treinam melhor com números pequenos e padronizados).

**Idade — normalização Z-Score:** transformamos a idade em **(idade − média) / desvio-padrão**. Resultado: idade média vira 0, e a escala fica comparável às outras variáveis. Analogia: converter "centímetros" e "quilos" para uma **régua comum**, senão a idade (0–85) "gritaria" mais alto que o sexo (0–1). Idades faltantes (<1%) preenchidas com a **mediana**.

**Sexo:** masculino = 0, feminino = 1.

**Localização anatômica — o detalhe que mais importou.** Há ~15 locais (face, dorso, mão…). Duas formas de virar número:

```
 ORDINAL (errada p/ a rede):   face=3,  dorso=8,  mão=11
   → a máquina "pensa": dorso(8) > face(3), e a distância face→dorso = 5  ❌ (não faz sentido!)

 ONE-HOT (correta):            face   dorso  mão  ...
   face   →                      1      0     0
   dorso  →                      0      1     0     ← uma coluna por local; exatamente um "1"
   mão    →                      0      0     1
```
- **Ordinal** (`cat.codes`): inventa uma **ordem/distância falsa** entre categorias. Foi assim **dentro da rede** (uma limitação que reconhecemos) — e ajudou a "estragar" o sinal clínico lá.
- **One-hot:** não impõe ordem nenhuma. Usamos one-hot **no classificador** → o clínico passou a **ajudar** (§13).

---

## 9. Data leakage por lesão e a correção

**Data leakage (vazamento de dados) — analogia:** é como **estudar com o gabarito da prova**. Se o modelo "viu" no treino algo que está no teste, a nota fica inflada e mentirosa.

**Nosso caso:** lembra que a mesma lesão tem várias fotos (§2)? Se sorteamos **por foto**, fotos de uma lesão caem no treino e **outras fotos da mesma lesão** caem no teste:

```
 ERRADO (por imagem):                     CERTO (por lesão):
 lesão A: foto1 → TREINO                  lesão A: foto1,foto2,foto3 → TREINO  (tudo junto)
          foto2 → TESTE  ❌ vazou!        lesão B: foto4,foto5       → TESTE
          foto3 → TREINO
```
No "errado", o modelo já "conhece" a lesão do teste → reconhece em vez de generalizar → métrica inflada.

**A correção (o ponto mais crítico do parecer da banca):**
- **`lesion_id`:** o identificador da **lesão** (não da foto). Agrupamos por ele.
- **GroupShuffleSplit:** sorteia **grupos inteiros (lesões)** para treino/validação/teste (70/10/20), garantindo que **todas as fotos de uma lesão fiquem do mesmo lado**.
- **`random_state=42`:** "trava" o sorteio → sempre o mesmo resultado (reprodutível).
- **StratifiedGroupKFold:** a validação cruzada (§15) que respeita os grupos (lesão) **e** tenta manter a proporção das classes.
- **Prova:** cada notebook tem um `assert` que **aborta** se alguma lesão aparecer em mais de um conjunto, e grava `leakage_check: 0` no `split_manifest.json`.

**Efeito:** ao corrigir, as métricas **caíram um pouco** (recall de melanoma do XGBoost+SMOTE: 32,7% → ~27% no limiar padrão). Isso é bom: significa que agora os números são **honestos**.

---

## 10. Por que desacoplar: CNN extratora + classificador clássico

**Analogia:** um perito com **ótima visão** (descreve a lesão perfeitamente) mas **péssimo em decidir** (sempre conclui "é benigno"). A solução não é trocar os olhos — é deixar **outra pessoa decidir** com base na descrição dele.

Foi o que descobrimos: a CNN **não falha em enxergar** (os embeddings guardam a informação que separa as classes); ela falha **na hora de decidir** (a camada Softmax colapsa). Prova: classificadores clássicos, usando **os mesmos embeddings**, recuperam o recall que a Softmax perdia.

- **Representação** (extrair características) **≠ Decisão** (traçar a fronteira).
- A **Softmax** traça **uma fronteira global rígida**; sob desbalanceamento, ela é empurrada para "esmagar" as classes raras.
- Árvores (XGBoost) fazem **recortes locais** e conseguem **cercar** pequenos grupos de melanoma no meio dos nevos.

➡️ Por isso: **CNN só como olhos (extratora); decisão com o clássico.**

---

## 11. Os classificadores clássicos

Todos decidem **sobre os embeddings** (256 ou 1536 números) — nunca sobre pixels. Analogias:

- **XGBoost (Extreme Gradient Boosting)** — *o nosso campeão.* É um **revezamento de árvores de decisão**: a 1ª árvore erra um pouco, a 2ª aprende a **corrigir o erro** da 1ª, a 3ª corrige o resto, e assim por diante (*boosting*). Analogia: uma **equipe onde cada pessoa conserta o erro da anterior**. Rápido e preciso com tabelas de números. Argumento principal: **`n_estimators=100`** (100 árvores no revezamento).
- **Random Forest (Floresta Aleatória)** — um **júri de muitas árvores independentes**, cada uma vendo uma amostra diferente; a decisão é por **voto da maioria** (*bagging*). Estável, difícil de "decorar".
- **SVM (Máquina de Vetores de Suporte)** — traça a **"rua mais larga" possível** entre dois bairros (classes), deixando a maior margem. Com **kernel RBF** consegue ruas curvas. `probability=True` faz um ajuste extra (lento — é o **gargalo de tempo** do pipeline).
- **Naive Bayes** — usa o **Teorema de Bayes** assumindo que as características são independentes (hipótese "ingênua", quase nunca verdadeira). Simples e rápido; serve de **chão de comparação** (*baseline*) — foi sempre o pior aqui.

---

## 12. Desbalanceamento: SMOTE vs balanceamento na imagem

O problema (67:1) pode ser atacado em **dois lugares diferentes**. Analogia geral: faltam exemplos das classes raras — como "criar" mais?

**SMOTE — inventa exemplos no espaço dos números (embeddings):**
- Pega dois melanomas reais **vizinhos** e cria um melanoma **sintético no meio do caminho** entre eles (interpola). Não copia — gera um ponto intermediário.
- Argumento **`k_neighbors=5`:** usa os 5 vizinhos mais próximos para interpolar.
- **Risco — "maldição da dimensionalidade":** em 256 dimensões o espaço é tão "vazio" que o "meio do caminho" pode cair numa região **sem sentido**, gerando exemplos ruidosos.

```
 SMOTE:    melanoma_real_1  ✕----------●----------✕  melanoma_real_2
                                  ↑ exemplo sintético "no meio"
```

**Balanceamento na imagem — atua na origem (antes da CNN):**
- Antes de treinar a CNN, **repete** as fotos das classes raras e **reduz** as da classe gigante, até todas terem ~o mesmo número (ex.: ~1000 cada). Como o *augmentation* (§6) é *on-the-fly*, **cada repetição vira uma foto um pouco diferente** (girada, mais clara…), gerando variedade **real**, não cópia idêntica.
- Analogia: em vez de **inventar** alunos (SMOTE), você **tira mais fotos, de ângulos diferentes**, dos alunos raros que já existem.
- **Achado:** isso **vence** o SMOTE (AUC 0,82 → 0,85), porque conserta o desbalanceamento **onde ele nasce** (na representação que a CNN aprende), não depois.

---

## 13. Onde fundir o metadado clínico

**A descoberta-chave:** o metadado clínico **não é inútil** — o que importa é **ONDE** você o junta com a imagem.

```
 ❌ DENTRO da rede (ordinal):   imagem + clínico → CNN → decisão     → clínico se PERDE (pior)
 ✅ NO classificador (one-hot): imagem → CNN → embedding + clínico → XGBoost → decisão (MELHOR)
```

- **Na rede (ramo clínico, ordinal):** sob desbalanceamento + codificação ordinal, o sinal clínico se **dissipa** (a classe gigante domina os gradientes e o encoding ruim atrapalha). O multimodal fica **pior** que a visão isolada.
- **No classificador (one-hot):** colar o metadado **one-hot** ao embedding antes do XGBoost **agrega** (AUC 0,853 → 0,883; F1 0,373 → 0,409). Árvores lidam naturalmente com misturas de números e categorias e **preservam** o sinal.
- **Prova de que o clínico tem sinal:** sozinho (só os 3 dados, sem imagem), atinge **AUC 0,744** — bem acima do acaso (0,5). Logo, o problema nunca foi o dado; era **onde** fundi-lo.

➡️ **Esta é a receita da campeã:** imagem pela CNN + clínico one-hot no XGBoost.

---

## 14. Métricas (cada uma, com exemplo numérico)

Tudo parte da **matriz de confusão**. Vamos com um **exemplo concreto**: 1000 lesões, sendo **200 malignas** e **800 benignas**. O modelo sinaliza algumas como "suspeitas":

```
                          PREVISTO
                     suspeita   liberada
 REAL  maligna  [   170 (VP)    30 (FN)  ]   ← pegou 170 dos 200 doentes
       benigna  [   232 (FP)   568 (VN)  ]   ← liberou 568 dos 800 sãos
```
VP=verdadeiro positivo, VN=verdadeiro negativo, FP=falso positivo (alarme falso), FN=falso negativo (doente perdido).

Agora cada métrica, com a conta:
- **Acurácia** = (VP+VN)/total = (170+568)/1000 = **74%**. *Enganosa:* chutar "tudo benigno" daria 80% aqui! **Não use como principal.**
- **Recall / Sensibilidade** = VP/(VP+FN) = 170/200 = **85%**. "Dos doentes, quantos peguei?" → **a métrica-rainha da triagem** (FN = melanoma perdido). Estes são os nossos ~85%.
- **Especificidade** = VN/(VN+FP) = 568/800 = **71%**. "Dos sãos, quantos liberei certo?" → mede alarme falso.
- **Precisão / PPV** = VP/(VP+FP) = 170/(170+232) = **42%**. "Dos que sinalizei, quantos eram doentes mesmo?" → PPV baixo = muito alarme falso (encaminha gente à toa).
- **F1-Macro:** média (harmônica) de precisão e recall, calculada **por classe** e depois média **simples** entre as 7 (cada classe pesa **igual**, mesmo `nv` sendo 67×). Por isso ignorar as raras **derruba** o F1-Macro → boa métrica primária.
- **AUC OVR:** mede a **capacidade de ranquear** — separar cada classe das outras, **sem depender do limiar**. 0,5 = moeda jogada pro alto; 1,0 = perfeito. "OVR" = uma classe contra todas, média no fim. É a nota mais robusta no desbalanceamento.

**Por que tantas métricas?** Porque **acurácia mente** em base desbalanceada. Na defesa, ancore em **sensibilidade + especificidade + F1-Macro + AUC**.

---

## 15. Validação estatística (CV e bootstrap)

Como saber se um resultado é **real** ou **sorte de uma divisão**? Duas ferramentas:

**Validação cruzada k-fold (CV) — "vários simulados":** divide os dados em **k=5** partes; faz 5 rodadas, cada uma testando numa parte diferente e treinando nas outras 4; a nota final é a **média ± desvio**. Desvio baixo (< 3 pontos) = resultado **estável**.

```
 Rodada 1: [TESTE][treino][treino][treino][treino]
 Rodada 2: [treino][TESTE][treino][treino][treino]
 ...                                              → média ± desvio
```
Usamos **StratifiedGroupKFold** (respeita lesão + proporção de classes — §9).

**Bootstrap (IC 95%) — "re-pesquisas de opinião":** reembaralha o conjunto de teste **com reposição** 1.000 vezes, recalcula a métrica em cada → 1.000 valores. O **intervalo de confiança de 95%** são os percentis 2,5% e 97,5%. Analogia: refazer a pesquisa eleitoral 1.000 vezes para ver a **margem de erro**. Se os intervalos de dois modelos **não se tocam**, a diferença é **significativa** (não foi sorte).

---

## 16. Limiar de decisão e ponto de operação clínico

**O que é o limiar (*threshold*):** o classificador dá uma **probabilidade** (ex.: "8% de chance de ser maligna"). O limiar é o **corte**: a partir de quanto eu chamo de "suspeito"? O padrão é **0,5 (50%)** — mas isso é arbitrário e **ruim para triagem**.

**Analogia do detector de fumaça (de novo):** o limiar é a **sensibilidade do botão**. Botão "duro" (limiar 0,5) → quase não dispara → perde incêndios. Botão "sensível" (limiar 0,05) → dispara mais → pega quase todos os incêndios, mas toca mais alarme falso. **Em câncer, queremos o botão sensível.**

```
 limiar ALTO (0,50): pega poucos doentes (sensib. baixa), poucos alarmes falsos
 limiar BAIXO (0,05): pega quase todos (sensib. ALTA), mais alarmes falsos  ← escolhemos este
```

**"Qualquer maligna":** em vez de mirar só melanoma, sinalizamos se **P(mel)+P(bcc)+P(akiec) ≥ limiar**. É um **rastreador de câncer de pele** (não só de melanoma), e dá números melhores (bcc e akiec são mais fáceis → puxam a sensibilidade pra cima).

**Curva ROC:** o gráfico de todos os limiares possíveis (sensibilidade no eixo Y, alarme falso no X). A **AUC** é a área sob ela. Escolhemos o **ponto** dessa curva que dá ~85% de sensibilidade.

**Nosso ponto de operação (campeã):** limiar ~**0,05** → **85% de sensibilidade** (recall mel 78%, bcc 92%, akiec 95%), especificidade 71%, PPV 41% (encaminha ~40% das lesões). A **acurácia cai de propósito** — porque em triagem o objetivo é **não perder doente**, não acertar a maioria.

---

## 17. Calibração de probabilidade

**O problema:** quando o modelo diz "8%", isso é **mesmo** 8% de chance real? Nem sempre — as probabilidades podem estar "torcidas".

**Calibração (isotônica/Platt):** reajusta os números para que "8%" signifique de fato 8%. Analogia: **reetiquetar um termômetro** descalibrado para que marque a temperatura certa — sem mudar a temperatura, só o rótulo.

**Achado honesto (importante na defesa):** como a calibração é uma transformação **monotônica** (preserva a ordem), ela **não muda** o trade-off sensibilidade×especificidade nem a AUC — apenas torna o **valor do limiar interpretável**. Ou seja: calibrar **não melhora a discriminação** (isso vem do modelo/AUC); só deixa o "8%" confiável. Reportamos isso **por transparência**, em vez de prometer um ganho que não existe.

---

## 18. O desenho fatorial 2×2

**O que é:** um **experimento controlado** que muda **um botão de cada vez** para saber **o que causou o quê**. Temos 2 botões: (1) usar clínico **na rede** ou não; (2) **balancear na imagem** ou não. Isso dá 4 combinações = 4 modelos, todos no **mesmo split por lesão**:

```
                       │ sem balanço de imagem │ com balanço de imagem
 ──────────────────────┼───────────────────────┼──────────────────────
 com clínico (na rede) │   nb01                 │   nb03
 só imagem             │   nb02                 │   nb04
```

Lendo a tabela: comparar **colunas** mostra o efeito do balanço; comparar **linhas** mostra o efeito do clínico-na-rede. Assim isolamos cada efeito de forma rigorosa (e não "achismo"). *(Lembre: a campeã pega a melhor CNN daqui — a só-imagem balanceada, nb04 — e adiciona o clínico no XGBoost.)*

---

## 19. Como ler nossos resultados

Números-âncora (teste *held-out* por lesão, XGBoost+SMOTE):

| Pipeline | F1-Macro | AUC OVR |
|---|---|---|
| Multimodal — clínico fundido na rede | 0,290 | 0,763 |
| CNN só-imagem (sem clínico) | 0,313 | 0,815 |
| CNN só-imagem + balanço de imagem | 0,373 | 0,853 |
| **★ CAMPEÃ: imagem + clínico (one-hot no XGBoost) + balanço** | **0,409** | **0,883** |

*(Todas as linhas usam a CNN como extratora; a campeã é a multimodal feita "do jeito certo" — imagem pela CNN e clínico no classificador.)*

Leitura em uma frase: *"corrigir o vazamento honesta os números; balancear na imagem é o que mais ajuda; o metadado só soma se fundido no classificador; e calibrando o limiar a ferramenta atinge 85% de sensibilidade para câncer de pele — útil como segunda opinião, não como diagnóstico autônomo."*

---

## 20. Limitações e honestidade científica

Ser honesto sobre os limites **fortalece** a defesa (mostra domínio):
- **Recall de melanoma (78% no ponto de 85%)** é o menor entre as malignas → **não substitui o especialista**.
- **PPV 41%** → ~40% das lesões encaminhadas; carga alta para a fila de regulação.
- **Viés étnico:** o HAM é eurocêntrico (fototipos I–II) → pode degradar em pele negra (daí o plano de **base brasileira** — ver doc de próximos passos).
- **Domínio:** dermatoscopia ≠ foto de celular da Atenção Básica.
- **Mantra:** *acurácia global é métrica enganosa em base desbalanceada* — é o fio condutor; repita na defesa.

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
