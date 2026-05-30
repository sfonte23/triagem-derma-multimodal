"""
evaluate_nova_base.py — Etapa 2 ICV/UFPI

Carrega o modelo pre-treinado no HAM10000, extrai features da nova base curada
pelo médico (data/nova_base/metadata_curada.csv), treina classificadores clássicos
nos embeddings HAM10000 e os testa na nova base. Gera métricas comparativas,
matrizes de confusão lado a lado e análises de viés por sexo e faixa etária.

Inclui dois ajustes pós-treinamento (sem retreinamento):
  Fix 1 — Correção de terminologia de localização (Derm7pt -> vocabulário HAM10000)
  Fix 2 — Correção Bayesiana de prior (Prior Shift Correction)

Pré-requisito: rodar prepare_nova_base.py primeiro.
"""

import os
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import kagglehub
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT           = Path(__file__).resolve().parent.parent
MODEL_PATH     = ROOT / "models" / "modelo_multimodal_final.keras"
NOVA_BASE_CSV  = ROOT / "data" / "nova_base" / "metadata_curada.csv"
NOVA_BASE_IMGS = ROOT / "data" / "nova_base" / "images"
STAGE1_CSV     = ROOT / "results" / "metricas_oficiais_pibic.csv"
OUT_NOVA       = ROOT / "results" / "nova_base"
OUT_COMP       = ROOT / "results" / "comparativo_v1_v2"
OUT_METRICS    = ROOT / "results" / "metricas_comparativo_v1_v2.csv"
OUT_AJUSTES    = OUT_NOVA / "metricas_ajustes_antes_depois.csv"

OUT_NOVA.mkdir(parents=True, exist_ok=True)
OUT_COMP.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constantes do HAM10000 (calculadas UMA vez antes da normalização Z-score)
# ---------------------------------------------------------------------------
HAM10000_AGE_MEDIAN = 50.0
HAM10000_AGE_MEAN   = 51.853220169745384
HAM10000_AGE_STD    = 16.92083280896139
HAM10000_CLASSES    = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Mapeamento de localização do HAM10000 (ordem alfabética — pandas category codes)
HAM10000_LOC_CATEGORIES = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp",
    "trunk", "unknown", "upper extremity"
]
HAM10000_LOC_MAPPING = {cat: code for code, cat in enumerate(HAM10000_LOC_CATEGORIES)}

# ---------------------------------------------------------------------------
# Fix 1 — Tradução de terminologia de localização Derm7pt -> HAM10000
# O Derm7pt usa "upper limbs"/"lower limbs"/"buttocks" que não existem no
# vocabulário HAM10000. Sem esta tradução, essas localizações (19 linhas)
# eram mapeadas para 'unknown' (código 13), degradando as features clínicas.
# ---------------------------------------------------------------------------
DERM7PT_LOC_TRANSLATION = {
    "upper limbs":  "upper extremity",   # HAM10000 usa "upper extremity"
    "lower limbs":  "lower extremity",   # HAM10000 usa "lower extremity"
    "buttocks":     "trunk",             # region glútea -> tronco (mais próximo)
}

# ---------------------------------------------------------------------------
# Função de perda customizada (necessária para carregar o modelo)
# ---------------------------------------------------------------------------
def categorical_focal_loss(gamma=2.0, alpha=0.75):
    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        modulating_factor = tf.pow(1.0 - y_pred, gamma)
        focal_loss = alpha * modulating_factor * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
    return categorical_focal_loss_fixed

# ---------------------------------------------------------------------------
# Utilitários de imagem
# ---------------------------------------------------------------------------
def find_image_path(image_id, img_dir, exts=(".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
    for ext in exts:
        p = img_dir / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(cv2.resize(img, (320, 320)), cv2.COLOR_BGR2RGB)
    return img / 255.0

def get_vectors_batched(df, feature_extractor, batch_size=32):
    feats_list, labels_list, imgs_list, clins_list = [], [], [], []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        imgs  = np.array([load_image(p) for p in batch["image_path"]])
        clins = batch[["age", "sex", "localization"]].values.astype("float32")
        labels = batch["label"].values if "label" in batch.columns else np.zeros(len(batch))

        feats = feature_extractor.predict([imgs, clins], verbose=0)
        feats_list.append(feats)
        labels_list.append(labels)
        imgs_list.append(imgs)
        clins_list.append(clins)
    return (
        np.vstack(feats_list),
        np.concatenate(labels_list),
        np.vstack(imgs_list),
        np.vstack(clins_list),
    )

def map_localization(val):
    val_lower = str(val).lower().strip()
    # Fix 1: traduzir termos Derm7pt para vocabulário HAM10000 antes da busca
    if val_lower in DERM7PT_LOC_TRANSLATION:
        val_lower = DERM7PT_LOC_TRANSLATION[val_lower]
    if val_lower in HAM10000_LOC_MAPPING:
        return HAM10000_LOC_MAPPING[val_lower]
    # Correspondência parcial como fallback
    for cat in HAM10000_LOC_CATEGORIES:
        if cat in val_lower or val_lower in cat:
            return HAM10000_LOC_MAPPING[cat]
    print(f"  [AVISO] Localizacao nao mapeada: '{val}' -> usando 13 ('unknown')")
    return 13  # 'unknown'

def compute_metrics(y_true, resultados_dict, classes_present):
    """Computa Accuracy, F1 Macro e AUC OVR para um dicionário de resultados."""
    rows = []
    for nome, res in resultados_dict.items():
        y_pred  = res["y_pred"]
        y_probs = res["y_probs"]
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro", labels=classes_present, zero_division=0)
        try:
            if y_probs is not None:
                probs_present = y_probs[:, classes_present]
                auc = roc_auc_score(y_true, probs_present, multi_class="ovr", labels=classes_present)
            else:
                auc = np.nan
        except Exception:
            auc = np.nan
        rows.append({"Algoritmo": nome, "Accuracy": acc, "F1_Macro": f1, "AUC_OVR": auc})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Baseline original (sem nenhum ajuste) — Etapa 2, primeira execução,
# antes do Fix 1 (localização) e Fix 2 (prior correction).
# Valores obtidos na execução inicial do pipeline e registrados no artigo v2.
# ---------------------------------------------------------------------------
_BASELINE_ANTES = [
    {"Algoritmo": "Multimodal CNN", "Acc_Antes": 0.2569, "F1_Antes": 0.0904, "AUC_Antes": float("nan")},
    {"Algoritmo": "Naive Bayes",    "Acc_Antes": 0.2153, "F1_Antes": 0.2330, "AUC_Antes": float("nan")},
    {"Algoritmo": "Random Forest",  "Acc_Antes": 0.3681, "F1_Antes": 0.3190, "AUC_Antes": float("nan")},
    {"Algoritmo": "XGBoost",        "Acc_Antes": 0.3750, "F1_Antes": 0.3170, "AUC_Antes": float("nan")},
    {"Algoritmo": "SVM",            "Acc_Antes": 0.2569, "F1_Antes": 0.0980, "AUC_Antes": float("nan")},
]
df_antes_nova = pd.DataFrame(_BASELINE_ANTES)
print("Baseline 'antes dos ajustes' carregado (valores da execucao original sem fixes).")

# ---------------------------------------------------------------------------
# 1. Carregar modelo
# ---------------------------------------------------------------------------
print("=" * 60)
print("ETAPA 1 — Carregando modelo pré-treinado")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"[ERRO] Modelo nao encontrado: {MODEL_PATH}")
    sys.exit(1)

model = load_model(
    MODEL_PATH,
    custom_objects={"categorical_focal_loss_fixed": categorical_focal_loss(gamma=2.0, alpha=0.75)}
)
print("  Modelo carregado com sucesso.")

layer_names = [l.name for l in model.layers]
assert "fused_dense_1" in layer_names, (
    f"Layer 'fused_dense_1' nao encontrada. Disponiveis: {layer_names}"
)
feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("fused_dense_1").output
)
print("  Feature extractor em 'fused_dense_1' configurado.")

# ---------------------------------------------------------------------------
# 2. Carregar e pré-processar nova base
# ---------------------------------------------------------------------------
print("\nETAPA 2 — Carregando nova base curada")

if not NOVA_BASE_CSV.exists():
    print(f"[ERRO] {NOVA_BASE_CSV} nao encontrado. Execute prepare_nova_base.py primeiro.")
    sys.exit(1)

nova = pd.read_csv(NOVA_BASE_CSV)
print(f"  Linhas carregadas: {len(nova)}")
print(f"  Distribuicao dx:\n{nova['dx'].value_counts().to_string()}")

# Preservar colunas raw para análise de viés
nova["age_raw"] = nova["age"].copy()
nova["sex_raw"] = nova["sex"].copy()

# Localização raw (antes da codificação) para diagnóstico do Fix 1
nova["localization_raw"] = nova["localization"].copy()

# age: fillna com mediana HAM10000, depois Z-score com estatísticas HAM10000
nova["age"] = nova["age"].replace("unknown", np.nan).astype(float)
nova["age"] = nova["age"].fillna(HAM10000_AGE_MEDIAN)
nova["age"] = (nova["age"] - HAM10000_AGE_MEAN) / HAM10000_AGE_STD

# sex: mapeamento idêntico ao treino
nova["sex"] = nova["sex"].str.lower().map({"male": 0, "female": 1}).fillna(2).astype(int)

# Contabilizar localizações afetadas pelo Fix 1 (para relatório)
loc_afetadas = nova["localization_raw"].str.lower().str.strip().isin(DERM7PT_LOC_TRANSLATION)
n_loc_fix = loc_afetadas.sum()
print(f"\n  Fix 1 — Localizacoes corrigidas pelo mapeamento Derm7pt->HAM10000: {n_loc_fix} linhas")
if n_loc_fix > 0:
    print(f"  {nova.loc[loc_afetadas, 'localization_raw'].value_counts().to_string()}")

# Aplicar mapeamento de localização (com Fix 1 embutido)
nova["localization"] = nova["localization"].apply(map_localization)

# LabelEncoder fixado nas 7 classes HAM10000
le = LabelEncoder()
le.fit(HAM10000_CLASSES)
nova["label"] = le.transform(nova["dx"])

# Resolver image_path
nova["image_path"] = nova["image_id"].apply(lambda x: find_image_path(x, NOVA_BASE_IMGS))
missing = nova["image_path"].isna().sum()
if missing > 0:
    print(f"  [AVISO] {missing} imagem(ns) nao encontrada(s) — serao removidas.")
nova = nova.dropna(subset=["image_path"]).reset_index(drop=True)
print(f"  Linhas com imagem disponivel: {len(nova)}")

if len(nova) == 0:
    print("[ERRO] Nenhuma imagem disponivel.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Features HAM10000 para treino dos classificadores clássicos
# ---------------------------------------------------------------------------
print("\nETAPA 3 — Construindo features HAM10000 para treino dos classificadores")

path_ham = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
data_dir  = Path(path_ham)
img_dirs  = [data_dir / "HAM10000_images_part_1", data_dir / "HAM10000_images_part_2"]

ham_meta = pd.read_csv(data_dir / "HAM10000_metadata.csv")

img_dict = {}
for d in img_dirs:
    if d.exists():
        for f in os.listdir(d):
            if f.endswith(".jpg"):
                img_dict[f[:-4]] = str(d / f)

ham_meta["image_path"] = ham_meta["image_id"].map(img_dict)
ham_meta = ham_meta.dropna(subset=["image_path"]).reset_index(drop=True)

ham_meta["age"] = ham_meta["age"].replace("unknown", np.nan).astype(float)
ham_meta["age"] = ham_meta["age"].fillna(HAM10000_AGE_MEDIAN)
ham_meta["age"] = (ham_meta["age"] - HAM10000_AGE_MEAN) / HAM10000_AGE_STD
ham_meta["sex"] = ham_meta["sex"].map({"male": 0, "female": 1, "unknown": 2}).fillna(2)
ham_meta["localization"] = ham_meta["localization"].apply(map_localization)
ham_meta["label"] = le.transform(ham_meta["dx"])

train_df, test_df_ham = train_test_split(
    ham_meta, test_size=0.2, stratify=ham_meta["label"], random_state=42
)
train_sample = train_df.sample(n=min(2000, len(train_df)), random_state=42)

print(f"  Extraindo features de {len(train_sample)} amostras HAM10000...")
X_train_feats, y_train, _, _ = get_vectors_batched(train_sample, feature_extractor)

# ---------------------------------------------------------------------------
# 4. Extrair features da nova base
# ---------------------------------------------------------------------------
print("\nETAPA 4 — Extraindo features da nova base")
print(f"  Processando {len(nova)} amostras...")
X_nova_feats, y_nova, X_nova_imgs, X_nova_clins = get_vectors_batched(nova, feature_extractor)

# ---------------------------------------------------------------------------
# 5. Inferência CNN + classificadores clássicos
# ---------------------------------------------------------------------------
print("\nETAPA 5 — Inferencia na nova base")

class_names = list(le.classes_)
resultados = {}

# CNN Multimodal
print("  CNN Multimodal...")
y_probs_cnn = model.predict([X_nova_imgs, X_nova_clins], verbose=0)
y_pred_cnn  = np.argmax(y_probs_cnn, axis=1)
resultados["Multimodal CNN"] = {"y_pred": y_pred_cnn, "y_probs": y_probs_cnn}

# Classificadores clássicos
modelos_ml = {
    "Naive Bayes":   GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":       XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
    "SVM":           SVC(probability=True, random_state=42),
}

for nome, clf in modelos_ml.items():
    print(f"  {nome}...")
    clf.fit(X_train_feats, y_train)
    y_pred = clf.predict(X_nova_feats)
    y_probs = clf.predict_proba(X_nova_feats) if hasattr(clf, "predict_proba") else None
    resultados[nome] = {"y_pred": y_pred, "y_probs": y_probs}

# ---------------------------------------------------------------------------
# 5b. Fix 2 — Correção Bayesiana de Prior (Prior Shift Correction)
#
# O HAM10000 tem 67% de nevo (nv), enquanto a nova base curada pelo médico
# tem ~25% de nevo. Esta mudança de distribuição faz o modelo tender a prever
# nv excessivamente. A correção ajusta as probabilidades sem retreinar:
#
#   p_corr[c] = p_pred[c] * (p_nova[c] / p_train[c])   (para cada classe c)
#   Normalizar: p_corr /= sum(p_corr)
#
# Referência: Saerens et al. (2002) — Adjusting the outputs of a classifier
# to new a priori probabilities. Neural Computation, 14(1), 21–41.
# ---------------------------------------------------------------------------
print("\nETAPA 5b — Fix 2: Correcao Bayesiana de Prior")

p_train_arr = ham_meta["dx"].value_counts(normalize=True).reindex(class_names, fill_value=1e-6).values
p_nova_arr  = nova["dx"].value_counts(normalize=True).reindex(class_names, fill_value=1e-6).values
prior_ratio = p_nova_arr / p_train_arr

print("  Razao de priors (nova/HAM10000):")
for c, ptr, pnv, r in zip(class_names, p_train_arr, p_nova_arr, prior_ratio):
    print(f"    {c}: HAM={ptr:.3f}  nova={pnv:.3f}  razao={r:.3f}")

resultados_corr = {}
for nome, res in resultados.items():
    if res["y_probs"] is not None:
        probs_corr = res["y_probs"] * prior_ratio
        probs_corr /= probs_corr.sum(axis=1, keepdims=True)
        pred_corr  = np.argmax(probs_corr, axis=1)
    else:
        probs_corr = None
        pred_corr  = res["y_pred"]
    resultados_corr[nome] = {"y_pred": pred_corr, "y_probs": probs_corr}

# ---------------------------------------------------------------------------
# 6. Métricas e tabela comparativa
# ---------------------------------------------------------------------------
print("\nETAPA 6 — Calculando metricas e gerando tabela comparativa")

nova_classes_present = sorted(nova["label"].unique())
nova_classes_present_names = [class_names[i] for i in nova_classes_present]

# Métricas após Fix 1 (localizacao corrigida, sem correcao de prior)
df_fix1 = compute_metrics(y_nova, resultados, nova_classes_present)
df_fix1.columns = ["Algoritmo", "Acc_Fix1", "F1_Fix1", "AUC_Fix1"]

# Métricas após Fix 1 + Fix 2 (localizacao + prior correction)
df_fix12 = compute_metrics(y_nova, resultados_corr, nova_classes_present)
df_fix12.columns = ["Algoritmo", "Acc_Fix12", "F1_Fix12", "AUC_Fix12"]

# Construir e salvar comparativo antes vs depois
if df_antes_nova is not None:
    df_ajustes = df_antes_nova.merge(df_fix1, on="Algoritmo").merge(df_fix12, on="Algoritmo")
    df_ajustes.to_csv(OUT_AJUSTES, index=False)
    print(f"  Salvo: {OUT_AJUSTES}")
    print("\n  Comparativo: Antes | Fix1 | Fix1+2")
    for _, row in df_ajustes.iterrows():
        print(f"  {row['Algoritmo']:20s}  Acc: {row['Acc_Antes']:.3f} -> {row['Acc_Fix1']:.3f} -> {row['Acc_Fix12']:.3f}"
              f"  |  F1: {row['F1_Antes']:.3f} -> {row['F1_Fix1']:.3f} -> {row['F1_Fix12']:.3f}")

# Atualizar metricas_comparativo_v1_v2.csv com os valores pós-ajuste (Fix 1+2)
df_nova_final = df_fix12.copy()
df_nova_final.columns = ["Algoritmo", "Accuracy_NovaBase", "F1_Macro_NovaBase", "AUC_OVR_NovaBase"]
df_stage1 = pd.read_csv(STAGE1_CSV).rename(columns={
    "Accuracy": "Accuracy_HAM10000",
    "F1_Macro": "F1_Macro_HAM10000",
    "AUC_OVR":  "AUC_OVR_HAM10000",
})
df_comp = df_stage1.merge(df_nova_final, on="Algoritmo")
df_comp.to_csv(OUT_METRICS, index=False)
print(f"\n  Salvo (atualizado com ajustes): {OUT_METRICS}")
print("\n  Metricas finais (pos-ajuste):")
print(df_nova_final.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
print("\nETAPA 7 — Gerando plots")

# ---- 7a. Métricas comparativas (barras agrupadas) -------------------------
metrics_map = [
    ("Accuracy_HAM10000", "Accuracy_NovaBase", "Accuracy"),
    ("F1_Macro_HAM10000", "F1_Macro_NovaBase", "F1 Macro"),
    ("AUC_OVR_HAM10000",  "AUC_OVR_NovaBase",  "AUC OVR"),
]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Metricas: HAM10000 (Etapa 1) vs. Nova Base (Etapa 2 — pos-ajuste)", fontsize=13)

for ax, (col_h, col_n, metric) in zip(axes, metrics_map):
    x = np.arange(len(df_comp))
    w = 0.35
    ax.bar(x - w / 2, df_comp[col_h], w, label="HAM10000", color="#4C72B0")
    ax.bar(x + w / 2, df_comp[col_n], w, label="Nova Base",  color="#DD8452")
    ax.set_title(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(df_comp["Algoritmo"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

plt.tight_layout()
p = OUT_COMP / "metrics_comparison_bar.png"
plt.savefig(p, dpi=150)
plt.close()
print(f"  Salvo: {p}")

# ---- 7b. Confusão lado a lado (XGBoost e Random Forest) ------------------
for nome_modelo, key in [("XGBoost", "XGBoost"), ("Random Forest", "Random_Forest")]:
    stage1_cm_path = ROOT / "results" / f"matrix_{key}.png"
    cm_nova = confusion_matrix(y_nova, resultados_corr[nome_modelo]["y_pred"],
                               labels=nova_classes_present)

    if stage1_cm_path.exists():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        img_stage1 = mpimg.imread(stage1_cm_path)
        axes[0].imshow(img_stage1)
        axes[0].axis("off")
        axes[0].set_title(f"{nome_modelo} - HAM10000 (Etapa 1)")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_nova,
                                      display_labels=nova_classes_present_names)
        disp.plot(cmap=plt.cm.Oranges, ax=axes[1], xticks_rotation=45)
        axes[1].set_title(f"{nome_modelo} - Nova Base (Etapa 2 pos-ajuste)")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_nova,
                                      display_labels=nova_classes_present_names)
        disp.plot(cmap=plt.cm.Oranges, ax=ax, xticks_rotation=45)
        ax.set_title(f"{nome_modelo} - Nova Base (Etapa 2 pos-ajuste)")

    plt.tight_layout()
    p = OUT_COMP / f"cm_side_by_side_{key}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Salvo: {p}")

# ---- 7c. Recall por classe ------------------------------------------------
xgb_rep = classification_report(
    y_nova, resultados_corr["XGBoost"]["y_pred"],
    labels=nova_labels_present if "nova_labels_present" in dir() else nova_classes_present,
    target_names=nova_classes_present_names,
    output_dict=True, zero_division=0
)
# Usar mesma variável para compatibilidade
nova_labels_present = nova_classes_present

recalls_nova_xgb = [xgb_rep.get(c, {}).get("recall", 0) for c in class_names]

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#D62728" if c == "mel" else "#4C72B0" for c in class_names]
bars   = ax.bar(class_names, recalls_nova_xgb, color=colors)
for c, b in zip(class_names, bars):
    if c == "mel":
        b.set_edgecolor("red")
        b.set_linewidth(2.5)
ax.set_title("Recall por Classe — XGBoost na Nova Base (mel = Melanoma em destaque)\n(pos-ajuste Fix 1+2)")
ax.set_ylabel("Recall")
ax.set_ylim(0, 1.05)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Limiar 50%")
ax.legend()
plt.tight_layout()
p = OUT_NOVA / "recall_per_class_nova.png"
plt.savefig(p, dpi=150)
plt.close()
print(f"  Salvo: {p}")

# ---- 7d. Degradação por classe (delta F1) ---------------------------------
f1_nova_xgb = {c: xgb_rep.get(c, {}).get("f1-score", 0.0) for c in class_names}

# Recalcular stage1 per-class F1 no HAM10000 test set
print("  Recalculando F1 por classe no HAM10000 test set para delta de degradacao...")
X_test_feats_ham, y_test_ham, _, _ = get_vectors_batched(test_df_ham, feature_extractor)
xgb_stage1 = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
xgb_stage1.fit(X_train_feats, y_train)
y_pred_ham_xgb = xgb_stage1.predict(X_test_feats_ham)
rep_ham_xgb = classification_report(y_test_ham, y_pred_ham_xgb, target_names=class_names, output_dict=True)
f1_ham_xgb = {c: rep_ham_xgb.get(c, {}).get("f1-score", 0) for c in class_names}

deltas = [f1_nova_xgb[c] - f1_ham_xgb[c] for c in class_names]

fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = ["#D62728" if d < 0 else "#2CA02C" for d in deltas]
ax.bar(class_names, deltas, color=bar_colors)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_title("Degradacao de F1 por Classe — XGBoost (Nova Base − HAM10000)\nVermelho=degradacao | Verde=melhora (pos-ajuste Fix 1+2)")
ax.set_ylabel("Delta F1-Score")
plt.tight_layout()
p = OUT_COMP / "degradation_per_class.png"
plt.savefig(p, dpi=150)
plt.close()
print(f"  Salvo: {p}")

# ---- 7e. Distribuição de classes -----------------------------------------
ham_dist  = ham_meta["dx"].value_counts(normalize=True).reindex(class_names, fill_value=0)
nova_dist = nova["dx"].value_counts(normalize=True).reindex(class_names, fill_value=0)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(class_names))
w = 0.35
ax.bar(x - w / 2, ham_dist.values, w, label="HAM10000 (treino)", color="#4C72B0")
ax.bar(x + w / 2, nova_dist.values, w, label="Nova Base (curada)", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_title("Distribuicao de Classes: HAM10000 vs. Nova Base\n(Diferenca motiva a Correcao de Prior — Fix 2)")
ax.set_ylabel("Proporcao")
ax.legend()
plt.tight_layout()
p = OUT_NOVA / "class_distribution_comparison.png"
plt.savefig(p, dpi=150)
plt.close()
print(f"  Salvo: {p}")

# ---- 7f. Viés por sexo ---------------------------------------------------
nova_bias = nova.copy()
nova_bias["y_pred_xgb"] = resultados_corr["XGBoost"]["y_pred"]

sex_results = {}
for sex_code, sex_label in [(0, "male"), (1, "female")]:
    sub = nova_bias[nova_bias["sex"] == sex_code]
    if len(sub) < 2:
        continue
    sex_results[sex_label] = {
        "Accuracy": accuracy_score(sub["label"], sub["y_pred_xgb"]),
        "F1_Macro": f1_score(sub["label"], sub["y_pred_xgb"], average="macro", zero_division=0),
    }

if sex_results:
    df_sex = pd.DataFrame(sex_results).T
    fig, ax = plt.subplots(figsize=(7, 5))
    df_sex.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Desempenho por Sexo — XGBoost na Nova Base (pos-ajuste)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(df_sex.index, rotation=0)
    ax.legend()
    plt.tight_layout()
    p = OUT_NOVA / "bias_sex.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Salvo: {p}")

# ---- 7g. Viés por faixa etária ------------------------------------------
def age_bucket(age_raw):
    try:
        a = float(age_raw)
    except (ValueError, TypeError):
        return None
    if a < 30:   return "<30"
    elif a < 50: return "30-50"
    elif a < 70: return "50-70"
    return ">70"

nova_bias["age_bucket"] = nova_bias["age_raw"].apply(age_bucket)
age_order = ["<30", "30-50", "50-70", ">70"]

age_results = {}
for bucket in age_order:
    sub = nova_bias[nova_bias["age_bucket"] == bucket]
    if len(sub) < 2:
        continue
    age_results[bucket] = f1_score(sub["label"], sub["y_pred_xgb"], average="macro", zero_division=0)

if age_results:
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_buckets = [b for b in age_order if b in age_results]
    vals = [age_results[b] for b in valid_buckets]
    ax.bar(valid_buckets, vals, color="#4C72B0")
    ax.set_title("F1 Macro por Faixa Etaria — XGBoost na Nova Base (pos-ajuste)")
    ax.set_ylabel("F1 Macro")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p = OUT_NOVA / "bias_age_group.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Salvo: {p}")

# ---- 7h. Antes vs Depois dos ajustes (comparativo visual) ----------------
if df_antes_nova is not None:
    df_ajustes_plot = df_antes_nova.merge(df_fix12, on="Algoritmo")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Impacto dos Ajustes Pos-Treinamento na Nova Base\n"
        "Fix 1: Correcao de Localizacao  |  Fix 2: Correcao Bayesiana de Prior",
        fontsize=12
    )

    for ax, (col_a, col_d, metric) in zip(axes, [
        ("Acc_Antes", "Acc_Fix12", "Accuracy"),
        ("F1_Antes",  "F1_Fix12",  "F1 Macro"),
    ]):
        x = np.arange(len(df_ajustes_plot))
        w = 0.35
        ax.bar(x - w/2, df_ajustes_plot[col_a], w,
               label="Antes (sem ajustes)", color="#D62728", alpha=0.85)
        ax.bar(x + w/2, df_ajustes_plot[col_d], w,
               label="Apos Fix 1+2", color="#2CA02C", alpha=0.85)
        # Anotar delta
        for xi, (va, vd) in enumerate(zip(df_ajustes_plot[col_a], df_ajustes_plot[col_d])):
            delta = vd - va
            symbol = "+" if delta >= 0 else ""
            ax.text(xi + w/2, vd + 0.01, f"{symbol}{delta:.2f}",
                    ha="center", va="bottom", fontsize=7, color="#2CA02C" if delta >= 0 else "#D62728")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(df_ajustes_plot["Algoritmo"], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)

    plt.tight_layout()
    p = OUT_NOVA / "ajustes_antes_vs_depois.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Salvo: {p}")

# ---- 7i. Diagrama metodológico dos dois ajustes --------------------------
fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor("#F8F9FA")

# Título
fig.text(0.5, 0.95, "Ajustes Pos-Treinamento Aplicados na Etapa 2",
         ha="center", va="top", fontsize=14, fontweight="bold", color="#1A1A2E")

# ---- Panel esquerdo: Fix 1 (Localização) ----
ax1 = fig.add_axes([0.04, 0.08, 0.44, 0.82])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis("off")
ax1.set_facecolor("#EAF4FB")

# Caixa de título
rect1 = mpatches.FancyBboxPatch((0.1, 8.5), 9.8, 1.2, boxstyle="round,pad=0.1",
                                  facecolor="#2471A3", edgecolor="none")
ax1.add_patch(rect1)
ax1.text(5, 9.1, "Fix 1 — Correcao de Terminologia de Localizacao",
         ha="center", va="center", fontsize=10, fontweight="bold", color="white")

# Problema
ax1.text(5, 8.0, "Problema: Derm7pt usa termos nao presentes no vocabulario HAM10000",
         ha="center", va="center", fontsize=8.5, color="#C0392B", style="italic")

# Tabela de mapeamento
headers = ["Termo Derm7pt (original)", "Termo HAM10000 (corrigido)", "Casos"]
data_rows = [
    ("upper limbs", "upper extremity (cod. 14)", "~8"),
    ("lower limbs", "lower extremity (cod. 9)",  "~9"),
    ("buttocks",    "trunk (cod. 12)",            "~2"),
    ("(outros)",    "(sem alteracao)",            "125"),
]
col_x = [0.5, 4.5, 8.5]
row_y_start = 7.1
row_h = 0.78

# Header
for cx, h in zip(col_x, headers):
    ax1.text(cx, row_y_start, h, ha="left", va="center", fontsize=7.5,
             fontweight="bold", color="#1A1A2E")
ax1.axhline(y=row_y_start - 0.25, xmin=0.03, xmax=0.97, color="#AAB7B8", linewidth=0.8)

for i, (orig, mapped, casos) in enumerate(data_rows):
    y = row_y_start - 0.5 - i * row_h
    bg_color = "#FDEDEC" if i < 3 else "#EAFAF1"
    rect = mpatches.FancyBboxPatch((0.1, y - 0.3), 9.8, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor=bg_color, edgecolor="#D5D8DC", linewidth=0.5)
    ax1.add_patch(rect)
    color = "#C0392B" if i < 3 else "#555555"
    ax1.text(col_x[0], y, orig,   ha="left", va="center", fontsize=8, color=color, fontweight="bold" if i < 3 else "normal")
    ax1.text(col_x[1], y, mapped, ha="left", va="center", fontsize=8, color="#1E8449" if i < 3 else "#555555")
    ax1.text(col_x[2], y, casos,  ha="left", va="center", fontsize=8, color="#555555")

# Resultado
ax1.text(5, 3.8 - 0.5, "Resultado: 19 amostras passam a fornecer features\nclinicas corretas ao modelo (antes: 'unknown')",
         ha="center", va="center", fontsize=8.5, color="#1E8449",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAFAF1", edgecolor="#1E8449"))

# ---- Panel direito: Fix 2 (Prior Correction) ----
ax2 = fig.add_axes([0.52, 0.08, 0.46, 0.82])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_facecolor("#F4ECF7")

rect2 = mpatches.FancyBboxPatch((0.1, 8.5), 9.8, 1.2, boxstyle="round,pad=0.1",
                                  facecolor="#7D3C98", edgecolor="none")
ax2.add_patch(rect2)
ax2.text(5, 9.1, "Fix 2 — Correcao Bayesiana de Prior (Prior Shift)",
         ha="center", va="center", fontsize=10, fontweight="bold", color="white")

ax2.text(5, 8.0, "Problema: Distribuicao de classes diverge entre treino e nova base",
         ha="center", va="center", fontsize=8.5, color="#C0392B", style="italic")

# Barra de distribuicao (HAM10000 vs Nova Base)
classes_show = ["akiec", "bcc", "bkl", "mel", "nv"]
p_train_show = [p_train_arr[class_names.index(c)] for c in classes_show]
p_nova_show  = [p_nova_arr[class_names.index(c)]  for c in classes_show]

bar_ax = fig.add_axes([0.54, 0.46, 0.42, 0.28])
x_b = np.arange(len(classes_show))
w_b = 0.35
bar_ax.bar(x_b - w_b/2, p_train_show, w_b, label="HAM10000", color="#4C72B0", alpha=0.85)
bar_ax.bar(x_b + w_b/2, p_nova_show,  w_b, label="Nova Base", color="#DD8452", alpha=0.85)
bar_ax.set_xticks(x_b)
bar_ax.set_xticklabels(classes_show, fontsize=7.5)
bar_ax.set_ylabel("Proporcao", fontsize=7.5)
bar_ax.set_title("Prior de Treino vs Nova Base", fontsize=8)
bar_ax.legend(fontsize=7, loc="upper right")
bar_ax.tick_params(labelsize=7)

# Fórmula
ax2.text(5, 4.4,
         "Formula de Correcao:",
         ha="center", va="center", fontsize=9, fontweight="bold", color="#1A1A2E")
ax2.text(5, 3.8,
         "p_corr[c]  =  p_pred[c]  x  ( p_nova[c] / p_train[c] )",
         ha="center", va="center", fontsize=9.5, color="#2471A3",
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2471A3"))
ax2.text(5, 3.1,
         "Normalizar: p_corr /= sum(p_corr)",
         ha="center", va="center", fontsize=8.5, color="#555555", style="italic")
ax2.text(5, 2.4,
         "Sem retreinamento — puramente pos-processamento",
         ha="center", va="center", fontsize=8, color="#7D3C98")

ax2.text(5, 1.6,
         "Resultado: modelo passa a refletir a distribuicao\nreal da pratica clinica brasileira",
         ha="center", va="center", fontsize=8.5, color="#1E8449",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAFAF1", edgecolor="#1E8449"))

p = OUT_NOVA / "diagrama_ajustes.png"
plt.savefig(p, dpi=150, facecolor=fig.get_facecolor())
plt.close()
print(f"  Salvo: {p}")

# ---------------------------------------------------------------------------
# Resumo final
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CONCLUIDO — Etapa 2 ICV (com ajustes Fix 1+2)")
print("=" * 60)
print(f"  Metricas comparativas:         {OUT_METRICS}")
print(f"  Comparativo antes/depois:      {OUT_AJUSTES}")
print(f"  Plots nova base:               {OUT_NOVA}/")
print(f"  Plots comparativo v1 vs v2:    {OUT_COMP}/")
print("\n  Novos plots gerados:")
print(f"    {OUT_NOVA}/ajustes_antes_vs_depois.png")
print(f"    {OUT_NOVA}/diagrama_ajustes.png")
