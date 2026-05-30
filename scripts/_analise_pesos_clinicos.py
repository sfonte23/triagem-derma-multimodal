"""Carrega o modelo HAM10000 e analisa os pesos aprendidos pela branch clinica.
   Gera heatmap (3 features x 32 neuronios) + barplot de importancia agregada."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True)

ROOT  = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "modelo_multimodal_final.keras"
OUT_DIR = ROOT / "docs" / "entregas" / "2.0_artigo_multimodal_base_nacional" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
print("OK.", flush=True)

# ---- Achar a primeira camada Dense da branch clinica (input shape (None, 3)) ----
target_layer = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        try:
            in_shape = layer.input.shape
        except Exception:
            try:
                in_shape = layer.input_shape
            except Exception:
                continue
        # input shape (None, 3) significa 3 features clinicas
        if len(in_shape) == 2 and in_shape[-1] == 3:
            target_layer = layer
            print(f"Encontrado: {layer.name}, units={layer.units}, in_shape={in_shape}", flush=True)
            break

if target_layer is None:
    print("Nao achei layer Dense com input (None,3). Listando candidatos...", flush=True)
    for l in model.layers:
        if isinstance(l, tf.keras.layers.Dense):
            try:
                shp = l.input.shape
            except Exception:
                shp = "?"
            print(f"  {l.name}: units={l.units}, input={shp}")
    sys.exit(1)

# ---- Extrair pesos ----
W, b = target_layer.get_weights()  # W shape: (3, 32), b shape: (32,)
print(f"W shape: {W.shape}, b shape: {b.shape}", flush=True)
print(f"W range: [{W.min():.4f}, {W.max():.4f}]", flush=True)

FEATURES = ["age (Z-score)", "sex (0/1/2)", "localization (0-14)"]

# Magnitude agregada por feature (L1 norm = soma dos abs)
importancia_l1 = np.abs(W).sum(axis=1)        # shape (3,)
importancia_l2 = np.sqrt((W ** 2).sum(axis=1)) # shape (3,)
importancia_max = np.abs(W).max(axis=1)        # shape (3,)
importancia_mean = np.abs(W).mean(axis=1)      # shape (3,)

print("\n=== Importancia agregada (magnitude absoluta) ===")
for i, name in enumerate(FEATURES):
    print(f"  {name:25s}  L1={importancia_l1[i]:6.3f}  L2={importancia_l2[i]:6.3f}  "
          f"mean={importancia_mean[i]:.3f}  max={importancia_max[i]:.3f}")

# ---- Plot 1: heatmap dos pesos (3 features x 32 neuronios) ----
fig, ax = plt.subplots(figsize=(14, 3.2))
vmax = np.abs(W).max()
im = ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
ax.set_yticks(range(3))
ax.set_yticklabels(FEATURES, fontsize=10)
ax.set_xticks(range(0, 32, 2))
ax.set_xticklabels([str(i) for i in range(0, 32, 2)], fontsize=8)
ax.set_xlabel("Neuronio da camada Dense(32) da branch clinica", fontsize=10)
ax.set_title("Pesos aprendidos da primeira camada da branch clinica\n"
             f"(modelo treinado HAM10000 — layer '{target_layer.name}')",
             fontsize=11, fontweight="bold")

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Peso", fontsize=9)

# Anotar a magnitude L2 no lado direito de cada linha
for i in range(3):
    ax.text(33, i, f" L2={importancia_l2[i]:.2f}", va="center", ha="left",
            fontsize=9, color="#333", fontweight="bold")

plt.tight_layout()
fig.savefig(OUT_DIR / "heatmap_pesos_branch_clinica.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"\nSalvo: {OUT_DIR}/heatmap_pesos_branch_clinica.png", flush=True)

# ---- Plot 2: barplot da importancia agregada ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Subplot esquerdo: barras horizontais comparando 4 metricas
metrics = {
    "L1 (sum abs)": importancia_l1,
    "L2 (norm)":    importancia_l2,
    "Max abs":      importancia_max,
    "Mean abs":     importancia_mean,
}
ax = axes[0]
x = np.arange(len(FEATURES))
w_bar = 0.20
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for i, (name, vals) in enumerate(metrics.items()):
    ax.bar(x + (i - 1.5) * w_bar, vals, w_bar, label=name, color=colors[i],
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(["age", "sex", "localiz."], fontsize=10)
ax.set_ylabel("Magnitude do peso", fontsize=10)
ax.set_title("Importancia agregada por feature\n(varias metricas)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Subplot direito: distribuicao (violin/box) dos pesos por feature
ax = axes[1]
data = [W[i, :] for i in range(3)]  # pesos por feature
bp = ax.boxplot(data, labels=["age", "sex", "localiz."], patch_artist=True, widths=0.5,
                showmeans=True, meanline=True,
                medianprops={"color": "black", "linewidth": 2},
                meanprops={"color": "red", "linewidth": 2, "linestyle": "--"})
box_colors = ["#4C72B0", "#DD8452", "#55A868"]
for patch, c in zip(bp["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_ylabel("Valor do peso", fontsize=10)
ax.set_title("Distribuicao dos 32 pesos por feature\n(box = quartis, traço vermelho = média)",
             fontsize=10, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
fig.savefig(OUT_DIR / "importancia_features_clinicas.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"Salvo: {OUT_DIR}/importancia_features_clinicas.png", flush=True)

# ---- Sumario textual ----
print("\n=== INTERPRETACAO ===")
ranking = sorted(enumerate(importancia_l2), key=lambda x: -x[1])
print("Ranking de importancia (por L2 norm):")
for rank, (idx, val) in enumerate(ranking, 1):
    print(f"  {rank}. {FEATURES[idx]:25s}  L2={val:.3f}")
top_feat = FEATURES[ranking[0][0]]
ratio = ranking[0][1] / ranking[-1][1]
print(f"\nA feature mais 'importante' segundo a magnitude dos pesos: {top_feat}")
print(f"Razao L2(top)/L2(menor): {ratio:.2f}x")
print("\nNota: magnitude alta de pesos NAO implica necessariamente que a feature")
print("contribui para a predicao final — pode estar sendo cancelada por outras camadas.")
print("Esta e uma analise LOCAL (primeira camada), nao causal.")
