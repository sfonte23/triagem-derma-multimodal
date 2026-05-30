"""Testa como o desempenho do modelo muda ao reescalar pesos da branch clinica.
   Sem retreinar — modifica W da Dense(32) clinica e roda inferencia."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True)

ROOT  = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "modelo_multimodal_final.keras"
CACHE = ROOT / "data" / "embeddings_cache"
OUT_DIR = ROOT / "docs" / "entregas" / "2.0_artigo_multimodal_base_nacional" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HAM_CLASSES = ["akiec","bcc","bkl","df","mel","nv","vasc"]

def focal_loss(gamma=2.0, alpha=0.75):
    def fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        mf = tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(alpha * mf * ce, axis=-1)
    return fn

print("Carregando modelo...", flush=True)
model = load_model(MODEL, custom_objects={"categorical_focal_loss_fixed": focal_loss(2.0, 0.75)},
                   compile=False)

# Achar a Dense(32) clinica
target = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        try:
            if len(layer.input.shape) == 2 and layer.input.shape[-1] == 3 and layer.units == 32:
                target = layer
                break
        except Exception:
            pass
print(f"Camada alvo: {target.name}, units={target.units}", flush=True)

# Pesos originais
W_orig, b_orig = target.get_weights()
print(f"W_orig shape: {W_orig.shape}", flush=True)

# Carregar test set cacheado
X_imgs  = np.load(CACHE / "X_test_imgs.npy")
X_clins = np.load(CACHE / "X_test_clins.npy")
y_test  = np.load(CACHE / "y_test.npy")
print(f"Test: imgs {X_imgs.shape}, clins {X_clins.shape}, labels {y_test.shape}", flush=True)

mel_idx = HAM_CLASSES.index("mel")
bcc_idx = HAM_CLASSES.index("bcc")
nv_idx  = HAM_CLASSES.index("nv")

# Configuracoes a testar: (age_w, sex_w, loc_w)
CONFIGS = [
    ("Baseline (1,1,1)",        1.0, 1.0, 1.0),
    ("Pedido (0.5,0.5,2.0)",    0.5, 0.5, 2.0),
    ("Agressivo loc (0.1,0.1,5)", 0.1, 0.1, 5.0),
    ("Sem age (0,1,1)",         0.0, 1.0, 1.0),
    ("Sem clinica (0,0,0)",     0.0, 0.0, 0.0),
    ("So age (1,0,0)",          1.0, 0.0, 0.0),
]

results = []
for label, wa, ws, wl in CONFIGS:
    # Reset para pesos originais
    target.set_weights([W_orig.copy(), b_orig.copy()])
    # Aplicar fatores na matriz (cada LINHA = uma feature de input)
    W_mod = W_orig.copy()
    W_mod[0, :] *= wa   # age
    W_mod[1, :] *= ws   # sex
    W_mod[2, :] *= wl   # localization
    target.set_weights([W_mod, b_orig.copy()])

    print(f"\n>>> {label}", flush=True)
    print(f"    Pesos: age*{wa}, sex*{ws}, loc*{wl}", flush=True)

    # Inferencia
    y_probs = model.predict([X_imgs, X_clins], batch_size=64, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_probs, multi_class="ovr")
    except Exception:
        auc = np.nan
    rec_mel = recall_score(y_test, y_pred, labels=[mel_idx], average="macro", zero_division=0)
    rec_bcc = recall_score(y_test, y_pred, labels=[bcc_idx], average="macro", zero_division=0)
    rec_nv  = recall_score(y_test, y_pred, labels=[nv_idx],  average="macro", zero_division=0)

    print(f"    Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}", flush=True)
    print(f"    Recall: mel={rec_mel:.3f}  bcc={rec_bcc:.3f}  nv={rec_nv:.3f}", flush=True)

    results.append({
        "config": label, "age": wa, "sex": ws, "loc": wl,
        "acc": acc, "f1": f1, "auc": auc,
        "rec_mel": rec_mel, "rec_bcc": rec_bcc, "rec_nv": rec_nv,
    })

# Restaurar pesos originais (boa pratica)
target.set_weights([W_orig.copy(), b_orig.copy()])

# Salvar CSV
import csv
with (OUT_DIR / "teste_pesos_clinicos.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print(f"\nCSV salvo: {OUT_DIR}/teste_pesos_clinicos.csv", flush=True)

# Plot comparativo
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: Acc / F1 / AUC
ax = axes[0]
labels = [r["config"] for r in results]
x = np.arange(len(labels))
w_bar = 0.27
ax.bar(x - w_bar, [r["acc"] for r in results], w_bar, label="Acuracia", color="#4C72B0")
ax.bar(x,         [r["f1"]  for r in results], w_bar, label="F1 Macro", color="#DD8452")
ax.bar(x + w_bar, [r["auc"] for r in results], w_bar, label="AUC OVR",  color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("Metricas globais por configuracao de pesos da branch clinica",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
# Valores
for i, r in enumerate(results):
    ax.text(i - w_bar, r["acc"] + 0.01, f"{r['acc']:.3f}", ha="center", fontsize=7)
    ax.text(i,         r["f1"]  + 0.01, f"{r['f1']:.3f}",  ha="center", fontsize=7)
    ax.text(i + w_bar, r["auc"] + 0.01, f"{r['auc']:.3f}", ha="center", fontsize=7)

# Plot 2: Recall por classe maligna
ax = axes[1]
ax.bar(x - w_bar, [r["rec_mel"] for r in results], w_bar, label="Recall Melanoma", color="#D62728")
ax.bar(x,         [r["rec_bcc"] for r in results], w_bar, label="Recall BCC",      color="#9467BD")
ax.bar(x + w_bar, [r["rec_nv"]  for r in results], w_bar, label="Recall Nevus",    color="#8C564B")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Recall por classe")
ax.set_title("Recall por classe critica",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
for i, r in enumerate(results):
    ax.text(i - w_bar, r["rec_mel"] + 0.02, f"{r['rec_mel']:.2f}", ha="center", fontsize=7)
    ax.text(i,         r["rec_bcc"] + 0.02, f"{r['rec_bcc']:.2f}", ha="center", fontsize=7)
    ax.text(i + w_bar, r["rec_nv"]  + 0.02, f"{r['rec_nv']:.2f}",  ha="center", fontsize=7)

plt.tight_layout()
fig.savefig(OUT_DIR / "teste_pesos_clinicos.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"Plot salvo: {OUT_DIR}/teste_pesos_clinicos.png", flush=True)

# Sumario
print("\n=== SUMARIO ===", flush=True)
print(f"{'Config':25s} {'Acc':>6s} {'F1':>6s} {'AUC':>6s} {'R_mel':>6s} {'R_bcc':>6s}", flush=True)
for r in results:
    print(f"{r['config']:25s} {r['acc']:>6.3f} {r['f1']:>6.3f} {r['auc']:>6.3f} "
          f"{r['rec_mel']:>6.3f} {r['rec_bcc']:>6.3f}", flush=True)
