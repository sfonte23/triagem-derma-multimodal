"""
experimentos_revisao_continuacao.py — continua os experimentos B4/B1/B5
após crash do t-SNE na rodada anterior.

Salva cache de embeddings em .npy para acelerar runs futuros.
"""

import os
import sys
import warnings
from pathlib import Path

# Forçar unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "modelo_multimodal_final.keras"
OUT_IMG    = ROOT / "docs" / "entregas" / "1.1_artigo_validacao_externa_correcao" / "images"
OUT_CSV    = ROOT / "docs" / "entregas" / "1.1_artigo_validacao_externa_correcao"
CACHE_DIR  = ROOT / "data" / "embeddings_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HAM10000_AGE_MEDIAN = 50.0
HAM10000_AGE_MEAN   = 51.853220169745384
HAM10000_AGE_STD    = 16.92083280896139
HAM10000_CLASSES    = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM10000_LOC_CATEGORIES = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp",
    "trunk", "unknown", "upper extremity"
]
HAM10000_LOC_MAPPING = {cat: code for code, cat in enumerate(HAM10000_LOC_CATEGORIES)}

def categorical_focal_loss(gamma=2.0, alpha=0.75):
    def fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        mf = tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(alpha * mf * ce, axis=-1)
    return fn

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(cv2.resize(img, (320, 320)), cv2.COLOR_BGR2RGB)
    return img / 255.0

def get_vectors_batched(df, fe, batch_size=32):
    feats, labels, imgs, clins = [], [], [], []
    n_batches = (len(df) + batch_size - 1) // batch_size
    for i in range(0, len(df), batch_size):
        b = df.iloc[i:i + batch_size]
        x = np.array([load_image(p) for p in b["image_path"]])
        c = b[["age", "sex", "localization"]].values.astype("float32")
        f = fe.predict([x, c], verbose=0)
        feats.append(f); labels.append(b["label"].values); imgs.append(x); clins.append(c)
        if (i // batch_size) % 10 == 0:
            print(f"  batch {i // batch_size + 1}/{n_batches}", flush=True)
    return np.vstack(feats), np.concatenate(labels), np.vstack(imgs), np.vstack(clins)

def map_localization(val):
    v = str(val).lower().strip()
    if v in HAM10000_LOC_MAPPING:
        return HAM10000_LOC_MAPPING[v]
    return 13

# ---------------------------------------------------------------------------
# Cache check — se já existir, pula extração
# ---------------------------------------------------------------------------
cache_files = {
    "Xtr": CACHE_DIR / "X_train_fused.npy",
    "ytr": CACHE_DIR / "y_train.npy",
    "Xte": CACHE_DIR / "X_test_fused.npy",
    "yte": CACHE_DIR / "y_test.npy",
    "Xte_imgs":  CACHE_DIR / "X_test_imgs.npy",
    "Xte_clins": CACHE_DIR / "X_test_clins.npy",
}

if all(p.exists() for p in cache_files.values()):
    print("Cache de embeddings encontrado — carregando...", flush=True)
    X_train_fused = np.load(cache_files["Xtr"])
    y_train       = np.load(cache_files["ytr"])
    X_test_fused  = np.load(cache_files["Xte"])
    y_test        = np.load(cache_files["yte"])
    X_test_imgs   = np.load(cache_files["Xte_imgs"])
    X_test_clins  = np.load(cache_files["Xte_clins"])
    print(f"  train={X_train_fused.shape}, test={X_test_fused.shape}", flush=True)
else:
    print("Cache nao encontrado — extraindo features...", flush=True)
    print("Carregando modelo...", flush=True)
    model = load_model(
        MODEL_PATH,
        custom_objects={"categorical_focal_loss_fixed": categorical_focal_loss(2.0, 0.75)}
    )
    fused_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer("fused_dense_1").output)

    print("Carregando HAM10000...", flush=True)
    path_ham = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    data_dir = Path(path_ham)
    img_dirs = [data_dir / "HAM10000_images_part_1", data_dir / "HAM10000_images_part_2"]
    ham = pd.read_csv(data_dir / "HAM10000_metadata.csv")
    img_dict = {}
    for d in img_dirs:
        if d.exists():
            for f in os.listdir(d):
                if f.endswith(".jpg"):
                    img_dict[f[:-4]] = str(d / f)
    ham["image_path"] = ham["image_id"].map(img_dict)
    ham = ham.dropna(subset=["image_path"]).reset_index(drop=True)
    ham["age"] = ham["age"].replace("unknown", np.nan).astype(float).fillna(HAM10000_AGE_MEDIAN)
    ham["age"] = (ham["age"] - HAM10000_AGE_MEAN) / HAM10000_AGE_STD
    ham["sex"] = ham["sex"].map({"male": 0, "female": 1, "unknown": 2}).fillna(2)
    ham["localization"] = ham["localization"].apply(map_localization)
    le = LabelEncoder(); le.fit(HAM10000_CLASSES)
    ham["label"] = le.transform(ham["dx"])

    train_df, test_df = train_test_split(ham, test_size=0.2, stratify=ham["label"], random_state=42)
    train_sample = train_df.sample(n=min(2000, len(train_df)), random_state=42).reset_index(drop=True)
    test_sample  = test_df.reset_index(drop=True)
    print(f"  treino: {len(train_sample)} | teste: {len(test_sample)}", flush=True)

    print("Extraindo features train...", flush=True)
    X_train_fused, y_train, _, _ = get_vectors_batched(train_sample, fused_extractor)
    print("Extraindo features test...", flush=True)
    X_test_fused, y_test, X_test_imgs, X_test_clins = get_vectors_batched(test_sample, fused_extractor)

    np.save(cache_files["Xtr"], X_train_fused)
    np.save(cache_files["ytr"], y_train)
    np.save(cache_files["Xte"], X_test_fused)
    np.save(cache_files["yte"], y_test)
    np.save(cache_files["Xte_imgs"],  X_test_imgs)
    np.save(cache_files["Xte_clins"], X_test_clins)
    print(f"  Cache salvo: {CACHE_DIR}/", flush=True)

# LabelEncoder reconstruido
le = LabelEncoder(); le.fit(HAM10000_CLASSES)
class_names = list(le.classes_)
mel_idx = le.transform(["mel"])[0]
bcc_idx = le.transform(["bcc"])[0]

# ---------------------------------------------------------------------------
# B4 — t-SNE dos embeddings
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("B4 — t-SNE dos embeddings (fused_dense_1)", flush=True)
print("=" * 60, flush=True)

print("Rodando t-SNE (~2 min)...", flush=True)
try:
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", max_iter=1000)
except TypeError:
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", n_iter=1000)
X_tsne = tsne.fit_transform(X_test_fused)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.tab10(np.linspace(0, 1, 7))
for i, cls in enumerate(HAM10000_CLASSES):
    mask = y_test == i
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=[colors[i]], label=f"{cls} ({mask.sum()})",
               s=20, alpha=0.7, edgecolors="none")
ax.set_title("t-SNE dos Embeddings da fused_dense_1 (HAM10000 test)", fontsize=12)
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
ax.legend(fontsize=9, loc="best", framealpha=0.9)
ax.grid(linestyle="--", alpha=0.3)
plt.tight_layout()
p = OUT_IMG / "tsne_embeddings.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Salvo: {p}", flush=True)

# ---------------------------------------------------------------------------
# B1 — 5-fold CV estratificada sobre embeddings
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("B1 — 5-Fold CV Estratificada", flush=True)
print("=" * 60, flush=True)

X_all = np.vstack([X_train_fused, X_test_fused])
y_all = np.concatenate([y_train, y_test])
if len(X_all) > 3000:
    idx_sub = np.random.RandomState(42).choice(len(X_all), 3000, replace=False)
    X_all = X_all[idx_sub]; y_all = y_all[idx_sub]
print(f"  Total amostras: {len(X_all)}", flush=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for clf_name, clf_fn in [
    ("XGBoost",       lambda: XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0)),
    ("Random Forest", lambda: RandomForestClassifier(n_estimators=100, random_state=42)),
    ("SVM",           lambda: SVC(probability=True, random_state=42)),
    ("Naive Bayes",   lambda: GaussianNB()),
]:
    print(f"  {clf_name}...", flush=True)
    accs, f1s, aucs = [], [], []
    for fold_idx, (tr, te) in enumerate(cv.split(X_all, y_all)):
        clf = clf_fn()
        clf.fit(X_all[tr], y_all[tr])
        yp = clf.predict(X_all[te])
        try:
            yprob = clf.predict_proba(X_all[te])
            auc = roc_auc_score(y_all[te], yprob, multi_class="ovr")
        except Exception:
            auc = np.nan
        accs.append(accuracy_score(y_all[te], yp))
        f1s.append(f1_score(y_all[te], yp, average="macro", zero_division=0))
        aucs.append(auc)
    cv_rows.append({
        "Classificador": clf_name,
        "Acc_mean": np.mean(accs), "Acc_std": np.std(accs),
        "F1_mean":  np.mean(f1s),  "F1_std":  np.std(f1s),
        "AUC_mean": np.nanmean(aucs), "AUC_std": np.nanstd(aucs),
    })
    print(f"    Acc {np.mean(accs):.3f}+-{np.std(accs):.3f}  "
          f"F1 {np.mean(f1s):.3f}+-{np.std(f1s):.3f}  "
          f"AUC {np.nanmean(aucs):.3f}+-{np.nanstd(aucs):.3f}", flush=True)

df_cv = pd.DataFrame(cv_rows)
df_cv.to_csv(OUT_CSV / "cv_5fold_results.csv", index=False)
print(f"  Salvo: {OUT_CSV}/cv_5fold_results.csv", flush=True)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_cv)); w = 0.27
ax.bar(x - w, df_cv["Acc_mean"], w, yerr=df_cv["Acc_std"], label="Acuracia", color="#4C72B0", capsize=4)
ax.bar(x,     df_cv["F1_mean"],  w, yerr=df_cv["F1_std"],  label="F1 Macro", color="#DD8452", capsize=4)
ax.bar(x + w, df_cv["AUC_mean"], w, yerr=df_cv["AUC_std"], label="AUC OVR",  color="#55A868", capsize=4)
ax.set_xticks(x); ax.set_xticklabels(df_cv["Classificador"], rotation=15)
ax.set_title("5-Fold CV Estratificada — Embeddings fused_dense_1 (media +- desvio)")
ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
p = OUT_IMG / "cv_5fold.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Salvo: {p}", flush=True)

# ---------------------------------------------------------------------------
# B5 — Bootstrap IC 95% para recall por classe maligna
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("B5 — Bootstrap IC 95% (recall mel, bcc)", flush=True)
print("=" * 60, flush=True)

# Carregar modelo de novo para gerar predições CNN
if "model" not in dir():
    print("  Recarregando modelo CNN para predicoes...", flush=True)
    model = load_model(
        MODEL_PATH,
        custom_objects={"categorical_focal_loss_fixed": categorical_focal_loss(2.0, 0.75)}
    )
y_probs_cnn = model.predict([X_test_imgs, X_test_clins], verbose=0)
y_pred_cnn  = np.argmax(y_probs_cnn, axis=1)

classificadores_final = {
    "XGBoost":       XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":           SVC(probability=True, random_state=42),
    "Naive Bayes":   GaussianNB(),
}
predicoes = {}
for nome, clf in classificadores_final.items():
    clf.fit(X_train_fused, y_train)
    predicoes[nome] = clf.predict(X_test_fused)
predicoes["CNN Multimodal"] = y_pred_cnn

n_boot = 1000
rng = np.random.RandomState(42)
boot_rows = []
for nome, y_pred in predicoes.items():
    rm, rb, fm = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(len(y_test), len(y_test), replace=True)
        yt, yp = y_test[idx], y_pred[idx]
        rm.append(recall_score(yt, yp, labels=[mel_idx], average="macro", zero_division=0))
        rb.append(recall_score(yt, yp, labels=[bcc_idx], average="macro", zero_division=0))
        fm.append(f1_score(yt, yp, average="macro", zero_division=0))
    rm_lo, rm_m, rm_hi = np.percentile(rm, 2.5), np.mean(rm), np.percentile(rm, 97.5)
    rb_lo, rb_m, rb_hi = np.percentile(rb, 2.5), np.mean(rb), np.percentile(rb, 97.5)
    fm_lo, fm_m, fm_hi = np.percentile(fm, 2.5), np.mean(fm), np.percentile(fm, 97.5)
    boot_rows.append({
        "Classificador": nome,
        "Recall_mel": rm_m, "Recall_mel_lo": rm_lo, "Recall_mel_hi": rm_hi,
        "Recall_bcc": rb_m, "Recall_bcc_lo": rb_lo, "Recall_bcc_hi": rb_hi,
        "F1_Macro":   fm_m, "F1_Macro_lo":   fm_lo, "F1_Macro_hi":   fm_hi,
    })
    print(f"  {nome:15s}  R_mel={rm_m:.3f} [{rm_lo:.3f}, {rm_hi:.3f}]  "
          f"R_bcc={rb_m:.3f} [{rb_lo:.3f}, {rb_hi:.3f}]", flush=True)

df_boot = pd.DataFrame(boot_rows)
df_boot.to_csv(OUT_CSV / "bootstrap_ci_results.csv", index=False)
print(f"  Salvo: {OUT_CSV}/bootstrap_ci_results.csv", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (cm, cl, ch, lbl) in zip(axes, [
    ("Recall_mel", "Recall_mel_lo", "Recall_mel_hi", "Recall — Melanoma (IC 95%)"),
    ("Recall_bcc", "Recall_bcc_lo", "Recall_bcc_hi", "Recall — Carcinoma Basocelular (IC 95%)"),
]):
    x = np.arange(len(df_boot))
    means  = df_boot[cm].values
    err_lo = means - df_boot[cl].values
    err_hi = df_boot[ch].values - means
    colors = ["#D62728" if n == "CNN Multimodal" else "#4C72B0" for n in df_boot["Classificador"]]
    ax.bar(x, means, yerr=[err_lo, err_hi], color=colors, capsize=6, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(df_boot["Classificador"], rotation=15, ha="right")
    ax.set_title(lbl); ax.set_ylabel("Recall")
    ax.set_ylim(0, max(0.5, (means + err_hi).max() * 1.2))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.suptitle("Bootstrap IC 95% — 1000 reamostragens (HAM10000 test)", fontsize=12)
plt.tight_layout()
p = OUT_IMG / "bootstrap_ci.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Salvo: {p}", flush=True)

print("\n" + "=" * 60, flush=True)
print("CONCLUIDO — B4 + B1 + B5", flush=True)
print("=" * 60, flush=True)
