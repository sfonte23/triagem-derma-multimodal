"""
experimentos_revisao.py — Experimentos adicionais para o artigo revisado
(resposta aos reviewers #24749 MiNDS@SBCAS).

Gera os plots e CSVs novos solicitados nas Fases B/E do plano_revisao_reviews.md:

  Fase E:
    E1. Matriz de confusão correta da CNN Multimodal (corrigindo bug R1.6 do v1)

  Fase B:
    B1. 5-fold CV estratificada dos classificadores sobre embeddings  (R1.8)
    B2. Ablation study: image-only / MLP-only / fused                (R1.9)
    B3. SMOTE sobre embeddings + retreino classificadores            (R1.10)
    B4. t-SNE dos embeddings da fused_dense_1                        (R1.5)
    B5. Bootstrap IC 95% para recall_mel e recall_bcc                (R2.1)

Todos os plots vão para docs/entregas/1.1_artigo_validacao_externa_correcao/images/.
"""

import os
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "modelo_multimodal_final.keras"
OUT_IMG    = ROOT / "docs" / "entregas" / "1.1_artigo_validacao_externa_correcao" / "images"
OUT_CSV    = ROOT / "docs" / "entregas" / "1.1_artigo_validacao_externa_correcao"
OUT_IMG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constantes HAM10000
# ---------------------------------------------------------------------------
HAM10000_AGE_MEDIAN = 50.0
HAM10000_AGE_MEAN   = 51.853220169745384
HAM10000_AGE_STD    = 16.92083280896139
HAM10000_CLASSES    = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES_LONG    = {
    "akiec": "Ceratose Actinica",
    "bcc":   "Carcinoma Basocelular",
    "bkl":   "Lesao Benigna",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Nevus",
    "vasc":  "Lesao Vascular",
}
HAM10000_LOC_CATEGORIES = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp",
    "trunk", "unknown", "upper extremity"
]
HAM10000_LOC_MAPPING = {cat: code for code, cat in enumerate(HAM10000_LOC_CATEGORIES)}

# ---------------------------------------------------------------------------
# Função de perda customizada
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
# Utilitários
# ---------------------------------------------------------------------------
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
        labels = batch["label"].values
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

# ---------------------------------------------------------------------------
# Carregar modelo + dois extractors: image-only (GAP) e fused
# ---------------------------------------------------------------------------
print("=" * 60)
print("Carregando modelo e preparando extractors")
print("=" * 60)

model = load_model(
    MODEL_PATH,
    custom_objects={"categorical_focal_loss_fixed": categorical_focal_loss(gamma=2.0, alpha=0.75)}
)

layer_names = [l.name for l in model.layers]
print(f"  Layers disponiveis: {len(layer_names)}")

# Extractor da fusão (256-dim)
fused_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("fused_dense_1").output
)

# Extractor da branch de imagem pura (saída do GAP)
gap_candidates = [n for n in layer_names if "global_average_pooling" in n.lower()]
if not gap_candidates:
    gap_candidates = [n for n in layer_names if "avg" in n.lower() and "pool" in n.lower()]
print(f"  Candidatos GAP: {gap_candidates}")
gap_layer = gap_candidates[0] if gap_candidates else None

img_only_extractor = None
if gap_layer:
    img_only_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(gap_layer).output
    )
    print(f"  Image-only extractor: {gap_layer}")

# ---------------------------------------------------------------------------
# Carregar HAM10000 e preparar metadados
# ---------------------------------------------------------------------------
print("\nBaixando/carregando HAM10000...")
path_ham = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
data_dir = Path(path_ham)
img_dirs = [data_dir / "HAM10000_images_part_1", data_dir / "HAM10000_images_part_2"]
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

def map_localization(val):
    val_lower = str(val).lower().strip()
    if val_lower in HAM10000_LOC_MAPPING:
        return HAM10000_LOC_MAPPING[val_lower]
    return 13
ham_meta["localization"] = ham_meta["localization"].apply(map_localization)

le = LabelEncoder()
le.fit(HAM10000_CLASSES)
ham_meta["label"] = le.transform(ham_meta["dx"])

# Split 80/20 estratificado (random_state=42)
train_df, test_df = train_test_split(
    ham_meta, test_size=0.2, stratify=ham_meta["label"], random_state=42
)
train_sample = train_df.sample(n=min(2000, len(train_df)), random_state=42).reset_index(drop=True)
test_sample  = test_df.reset_index(drop=True)

print(f"  Treino: {len(train_sample)} | Teste: {len(test_sample)}")

# ---------------------------------------------------------------------------
# Extrair features (fused) — para classificadores clássicos e CNN
# ---------------------------------------------------------------------------
print("\nExtraindo features fused (treino)...")
X_train_fused, y_train, _, _ = get_vectors_batched(train_sample, fused_extractor)
print("Extraindo features fused (teste)...")
X_test_fused, y_test, X_test_imgs, X_test_clins = get_vectors_batched(test_sample, fused_extractor)

print("Extraindo features image-only (treino)...")
X_train_img, _, _, _ = get_vectors_batched(train_sample, img_only_extractor)
print("Extraindo features image-only (teste)...")
X_test_img, _, _, _ = get_vectors_batched(test_sample, img_only_extractor)

print(f"  fused: train={X_train_fused.shape}, test={X_test_fused.shape}")
print(f"  img-only: train={X_train_img.shape}, test={X_test_img.shape}")

# ---------------------------------------------------------------------------
# E1 — Matriz de confusão correta da CNN Multimodal (R1.6 bug fix)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("E1 — Matriz de confusao da CNN Multimodal (correta)")
print("=" * 60)

y_probs_cnn = model.predict([X_test_imgs, X_test_clins], verbose=0)
y_pred_cnn  = np.argmax(y_probs_cnn, axis=1)

cm_cnn = confusion_matrix(y_test, y_pred_cnn, labels=range(7))
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=HAM10000_CLASSES)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, values_format='d')
ax.set_title("Matriz de Confusao — CNN Multimodal Fim-a-Fim (HAM10000 test)")
plt.tight_layout()
plt.savefig(OUT_IMG / "matrix_cnn_multimodal.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: {OUT_IMG}/matrix_cnn_multimodal.png")

# ---------------------------------------------------------------------------
# B2 — Ablation study: image-only / MLP-only / fused
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("B2 — Ablation Study (image-only / MLP-only / fused)")
print("=" * 60)

# MLP-only features = features clinicas brutas
X_train_mlp = train_sample[["age", "sex", "localization"]].values.astype("float32")
X_test_mlp  = test_sample[["age", "sex", "localization"]].values.astype("float32")

# Normalizar features mlp (importante para SVM/NB)
scaler_mlp = StandardScaler()
X_train_mlp_n = scaler_mlp.fit_transform(X_train_mlp)
X_test_mlp_n  = scaler_mlp.transform(X_test_mlp)

# Normalizar features image-only (essas saem direto do GAP, valores grandes)
scaler_img = StandardScaler()
X_train_img_n = scaler_img.fit_transform(X_train_img)
X_test_img_n  = scaler_img.transform(X_test_img)

ablation_configs = [
    ("image-only", X_train_img_n, X_test_img_n),
    ("clinical-only", X_train_mlp_n, X_test_mlp_n),
    ("fused (proposto)", X_train_fused, X_test_fused),
]

ablation_rows = []
for cfg_name, Xtr, Xte in ablation_configs:
    print(f"\n  Configuracao: {cfg_name}")
    for clf_name, clf in [
        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]:
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_probs = clf.predict_proba(Xte)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_probs, multi_class="ovr")
        except Exception:
            auc = np.nan
        recall_mel = recall_score(y_test, y_pred, labels=[le.transform(["mel"])[0]], average="macro", zero_division=0)
        recall_bcc = recall_score(y_test, y_pred, labels=[le.transform(["bcc"])[0]], average="macro", zero_division=0)
        ablation_rows.append({
            "Configuracao": cfg_name, "Classificador": clf_name,
            "Accuracy": acc, "F1_Macro": f1, "AUC_OVR": auc,
            "Recall_mel": recall_mel, "Recall_bcc": recall_bcc,
        })
        print(f"    {clf_name}: Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f} R_mel={recall_mel:.3f} R_bcc={recall_bcc:.3f}")

df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv(OUT_CSV / "ablation_results.csv", index=False)
print(f"\n  Salvo: {OUT_CSV}/ablation_results.csv")

# Plot ablation
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics_plot = [("Accuracy", "Acuracia"), ("F1_Macro", "F1 Macro"), ("AUC_OVR", "AUC OVR")]
for ax, (col, label) in zip(axes, metrics_plot):
    pivot = df_ablation.pivot(index="Configuracao", columns="Classificador", values=col)
    pivot = pivot.reindex(["image-only", "clinical-only", "fused (proposto)"])
    pivot.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"], rot=15)
    ax.set_title(label, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(label)
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
plt.suptitle("Ablation Study — Contribuicao de cada Branch", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_IMG / "ablation_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: {OUT_IMG}/ablation_results.png")

# ---------------------------------------------------------------------------
# B3 — SMOTE sobre embeddings fused
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("B3 — SMOTE sobre embeddings fused")
print("=" * 60)

unique, counts = np.unique(y_train, return_counts=True)
print(f"  Antes do SMOTE: {dict(zip([HAM10000_CLASSES[i] for i in unique], counts))}")

k_neighbors = min(5, int(min(counts) - 1))
smote = SMOTE(random_state=42, k_neighbors=max(1, k_neighbors))
X_train_smote, y_train_smote = smote.fit_resample(X_train_fused, y_train)

unique, counts = np.unique(y_train_smote, return_counts=True)
print(f"  Apos SMOTE: {dict(zip([HAM10000_CLASSES[i] for i in unique], counts))}")

smote_rows = []
for clf_name, clf_fn in [
    ("XGBoost",       lambda: XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0)),
    ("Random Forest", lambda: RandomForestClassifier(n_estimators=100, random_state=42)),
    ("SVM",           lambda: SVC(probability=True, random_state=42)),
    ("Naive Bayes",   lambda: GaussianNB()),
]:
    print(f"  {clf_name}...")
    for variant, Xtr, ytr in [
        ("Baseline", X_train_fused, y_train),
        ("SMOTE",    X_train_smote, y_train_smote),
    ]:
        clf = clf_fn()
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(X_test_fused)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        recall_mel = recall_score(y_test, y_pred, labels=[le.transform(["mel"])[0]], average="macro", zero_division=0)
        recall_bcc = recall_score(y_test, y_pred, labels=[le.transform(["bcc"])[0]], average="macro", zero_division=0)
        smote_rows.append({
            "Classificador": clf_name, "Variante": variant,
            "Accuracy": acc, "F1_Macro": f1,
            "Recall_mel": recall_mel, "Recall_bcc": recall_bcc,
        })
        print(f"    {variant}: Acc={acc:.3f} F1={f1:.3f} R_mel={recall_mel:.3f} R_bcc={recall_bcc:.3f}")

df_smote = pd.DataFrame(smote_rows)
df_smote.to_csv(OUT_CSV / "smote_results.csv", index=False)
print(f"\n  Salvo: {OUT_CSV}/smote_results.csv")

# Plot SMOTE comparison (foco em F1 Macro e Recall mel)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (col, label) in zip(axes, [("F1_Macro", "F1 Macro"), ("Recall_mel", "Recall — Melanoma")]):
    pivot = df_smote.pivot(index="Classificador", columns="Variante", values=col)
    pivot = pivot[["Baseline", "SMOTE"]]
    pivot.plot(kind="bar", ax=ax, color=["#D62728", "#2CA02C"], rot=15)
    ax.set_title(label)
    ax.set_ylim(0, max(0.5, pivot.values.max() * 1.15))
    ax.set_ylabel(label)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
plt.suptitle("Impacto do SMOTE sobre Embeddings fused_dense_1", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_IMG / "smote_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: {OUT_IMG}/smote_comparison.png")

# ---------------------------------------------------------------------------
# B4 — t-SNE dos embeddings fused
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("B4 — t-SNE dos embeddings (fused_dense_1)")
print("=" * 60)

X_tsne_input = X_test_fused
y_tsne       = y_test

print("  Rodando t-SNE (pode levar 1-2 min)...")
try:
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", max_iter=1000)
except TypeError:
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", n_iter=1000)
X_tsne = tsne.fit_transform(X_tsne_input)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.tab10(np.linspace(0, 1, 7))
for i, cls in enumerate(HAM10000_CLASSES):
    mask = y_tsne == i
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=[colors[i]], label=f"{cls} ({mask.sum()})",
               s=20, alpha=0.7, edgecolors="none")
ax.set_title("t-SNE dos Embeddings da fused_dense_1 (HAM10000 test)", fontsize=12)
ax.set_xlabel("t-SNE dim 1")
ax.set_ylabel("t-SNE dim 2")
ax.legend(fontsize=9, loc="best", framealpha=0.9)
ax.grid(linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_IMG / "tsne_embeddings.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: {OUT_IMG}/tsne_embeddings.png")

# ---------------------------------------------------------------------------
# B1 — 5-fold CV estratificada sobre embeddings
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("B1 — 5-Fold CV Estratificada sobre Embeddings fused")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []

# Combinar treino + teste para ter mais dados de CV
X_all = np.vstack([X_train_fused, X_test_fused])
y_all = np.concatenate([y_train, y_test])

# Subsample para performance
if len(X_all) > 3000:
    idx_sub = np.random.RandomState(42).choice(len(X_all), 3000, replace=False)
    X_all = X_all[idx_sub]
    y_all = y_all[idx_sub]
print(f"  Total amostras: {len(X_all)}")

for clf_name, clf_fn in [
    ("XGBoost",       lambda: XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0)),
    ("Random Forest", lambda: RandomForestClassifier(n_estimators=100, random_state=42)),
    ("SVM",           lambda: SVC(probability=True, random_state=42)),
    ("Naive Bayes",   lambda: GaussianNB()),
]:
    print(f"  {clf_name}...")
    fold_accs, fold_f1s, fold_aucs = [], [], []
    for fold_idx, (tr, te) in enumerate(cv.split(X_all, y_all)):
        clf = clf_fn()
        clf.fit(X_all[tr], y_all[tr])
        y_pred = clf.predict(X_all[te])
        try:
            y_probs = clf.predict_proba(X_all[te])
            auc = roc_auc_score(y_all[te], y_probs, multi_class="ovr")
        except Exception:
            auc = np.nan
        fold_accs.append(accuracy_score(y_all[te], y_pred))
        fold_f1s.append(f1_score(y_all[te], y_pred, average="macro", zero_division=0))
        fold_aucs.append(auc)

    cv_rows.append({
        "Classificador": clf_name,
        "Acc_mean": np.mean(fold_accs), "Acc_std": np.std(fold_accs),
        "F1_mean":  np.mean(fold_f1s),  "F1_std":  np.std(fold_f1s),
        "AUC_mean": np.nanmean(fold_aucs), "AUC_std": np.nanstd(fold_aucs),
    })
    print(f"    Acc {np.mean(fold_accs):.3f}+-{np.std(fold_accs):.3f}  "
          f"F1 {np.mean(fold_f1s):.3f}+-{np.std(fold_f1s):.3f}  "
          f"AUC {np.nanmean(fold_aucs):.3f}+-{np.nanstd(fold_aucs):.3f}")

df_cv = pd.DataFrame(cv_rows)
df_cv.to_csv(OUT_CSV / "cv_5fold_results.csv", index=False)
print(f"\n  Salvo: {OUT_CSV}/cv_5fold_results.csv")

# Plot CV
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_cv))
w = 0.27
ax.bar(x - w, df_cv["Acc_mean"], w, yerr=df_cv["Acc_std"], label="Acuracia", color="#4C72B0", capsize=4)
ax.bar(x,     df_cv["F1_mean"],  w, yerr=df_cv["F1_std"],  label="F1 Macro", color="#DD8452", capsize=4)
ax.bar(x + w, df_cv["AUC_mean"], w, yerr=df_cv["AUC_std"], label="AUC OVR",  color="#55A868", capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(df_cv["Classificador"], rotation=15)
ax.set_title("5-Fold CV Estratificada — Embeddings fused_dense_1 (media +- desvio)")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_IMG / "cv_5fold.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: {OUT_IMG}/cv_5fold.png")

# ---------------------------------------------------------------------------
# B5 — Bootstrap IC 95% para recall por classe maligna
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("B5 — Bootstrap IC 95% para Recall (mel, bcc)")
print("=" * 60)

# Re-treinar classificadores no split original e usar test_sample
classificadores_final = {
    "XGBoost":       XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":           SVC(probability=True, random_state=42),
    "Naive Bayes":   GaussianNB(),
}
predicoes_finais = {}
for nome, clf in classificadores_final.items():
    clf.fit(X_train_fused, y_train)
    predicoes_finais[nome] = clf.predict(X_test_fused)
# CNN
predicoes_finais["CNN Multimodal"] = y_pred_cnn

mel_idx = le.transform(["mel"])[0]
bcc_idx = le.transform(["bcc"])[0]

n_boot = 1000
rng = np.random.RandomState(42)
boot_rows = []
for nome, y_pred in predicoes_finais.items():
    rec_mel_samples, rec_bcc_samples, f1_samples = [], [], []
    for _ in range(n_boot):
        idx_b = rng.choice(len(y_test), len(y_test), replace=True)
        yt_b, yp_b = y_test[idx_b], y_pred[idx_b]
        rec_mel_samples.append(recall_score(yt_b, yp_b, labels=[mel_idx], average="macro", zero_division=0))
        rec_bcc_samples.append(recall_score(yt_b, yp_b, labels=[bcc_idx], average="macro", zero_division=0))
        f1_samples.append(f1_score(yt_b, yp_b, average="macro", zero_division=0))

    def ci(samples):
        return np.percentile(samples, 2.5), np.mean(samples), np.percentile(samples, 97.5)

    rm_lo, rm_m, rm_hi = ci(rec_mel_samples)
    rb_lo, rb_m, rb_hi = ci(rec_bcc_samples)
    f1_lo, f1_m, f1_hi = ci(f1_samples)
    boot_rows.append({
        "Classificador": nome,
        "Recall_mel": rm_m, "Recall_mel_lo": rm_lo, "Recall_mel_hi": rm_hi,
        "Recall_bcc": rb_m, "Recall_bcc_lo": rb_lo, "Recall_bcc_hi": rb_hi,
        "F1_Macro":   f1_m, "F1_Macro_lo":   f1_lo, "F1_Macro_hi":   f1_hi,
    })
    print(f"  {nome:15s}  R_mel={rm_m:.3f} [{rm_lo:.3f}, {rm_hi:.3f}]  "
          f"R_bcc={rb_m:.3f} [{rb_lo:.3f}, {rb_hi:.3f}]")

df_boot = pd.DataFrame(boot_rows)
df_boot.to_csv(OUT_CSV / "bootstrap_ci_results.csv", index=False)
print(f"\n  Salvo: {OUT_CSV}/bootstrap_ci_results.csv")

# Plot bootstrap CI
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (col_m, col_lo, col_hi, label) in zip(axes, [
    ("Recall_mel", "Recall_mel_lo", "Recall_mel_hi", "Recall — Melanoma (IC 95%)"),
    ("Recall_bcc", "Recall_bcc_lo", "Recall_bcc_hi", "Recall — Carcinoma Basocelular (IC 95%)"),
]):
    x = np.arange(len(df_boot))
    means = df_boot[col_m].values
    err_lo = means - df_boot[col_lo].values
    err_hi = df_boot[col_hi].values - means
    colors = ["#D62728" if n == "CNN Multimodal" else "#4C72B0" for n in df_boot["Classificador"]]
    ax.bar(x, means, yerr=[err_lo, err_hi], color=colors, capsize=6, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_boot["Classificador"], rotation=15, ha="right")
    ax.set_title(label)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, max(0.5, (means + err_hi).max() * 1.2))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.suptitle("Bootstrap IC 95% — 1000 reamostragens (HAM10000 test)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_IMG / "bootstrap_ci.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: {OUT_IMG}/bootstrap_ci.png")

# ---------------------------------------------------------------------------
# Resumo final
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CONCLUIDO — Todos os experimentos da Fase B/E")
print("=" * 60)
print("Arquivos gerados:")
print(f"  Imagens: {OUT_IMG}/")
for f in sorted(OUT_IMG.glob("*.png")):
    print(f"    - {f.name}")
print(f"  CSVs: {OUT_CSV}/")
for f in sorted(OUT_CSV.glob("*.csv")):
    print(f"    - {f.name}")
