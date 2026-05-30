"""Analise dos pesos FINAIS da branch clinica:
   1) Jacobiano numerico: peso efetivo de cada feature original (age/sex/loc)
      sobre os 8 outputs finais da branch clinica (apos todas as camadas).
   2) Camada de fusao: quanto peso o modelo da pra branch visual vs clinica.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True)

ROOT  = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "modelo_multimodal_final.keras"
CACHE = ROOT / "data" / "embeddings_cache"
OUT_DIR = ROOT / "docs" / "entregas" / "2.0_artigo_multimodal_base_nacional" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def focal_loss(gamma=2.0, alpha=0.75):
    def fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        mf = tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(alpha * mf * ce, axis=-1)
    return fn

print("Carregando modelo (compile=False)...", flush=True)
model = load_model(MODEL, custom_objects={"categorical_focal_loss_fixed": focal_loss(2.0, 0.75)},
                   compile=False)
print("OK.", flush=True)

# ============================================================================
# PARTE 1 — JACOBIANO NUMERICO da branch clinica
# ============================================================================
# Estrategia: criar sub-modelo (clin_input -> branch_clin_output 8-dim)
# Calcular Jacobiano em varias amostras reais e tirar a media absoluta.

print("\n=== PARTE 1: Jacobiano numerico da branch clinica ===", flush=True)

# Achar o input da branch clinica (shape (None, 3))
clin_input = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.InputLayer):
        shp = layer.output.shape
        if len(shp) == 2 and shp[-1] == 3:
            clin_input = layer.output
            print(f"  Input clinico: {layer.name} shape={shp}", flush=True)
            break

# Achar a saida da branch clinica (Dense(8) antes do Concatenate)
clin_output = None
concat_layer = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Concatenate):
        concat_layer = layer
        print(f"  Concatenate: {layer.name}", flush=True)
        for inp in layer.input:
            if inp.shape[-1] == 8:
                clin_output = inp
                print(f"    Branch clinica output: shape={inp.shape}", flush=True)
            else:
                print(f"    Branch visual output: shape={inp.shape}", flush=True)
        break

# Construir sub-modelo
sub_model = Model(inputs=clin_input, outputs=clin_output)
print(f"  Sub-modelo: input shape {sub_model.input_shape} -> output shape {sub_model.output_shape}",
      flush=True)

# Carregar amostras reais de teste para avaliar Jacobiano
X_test_clins = np.load(CACHE / "X_test_clins.npy")
print(f"  Usando {len(X_test_clins)} amostras de teste para calcular Jacobiano medio.", flush=True)

# Calcular Jacobiano via GradientTape para cada amostra, depois media absoluta
# Jacobiano: dY/dX onde Y in R^8, X in R^3 -> matriz 8x3 por amostra
batch_size = 64
n = len(X_test_clins)
jac_abs_sum = np.zeros((8, 3))
n_used = 0
for i in range(0, n, batch_size):
    batch = X_test_clins[i:i+batch_size].astype(np.float32)
    batch_t = tf.constant(batch)
    with tf.GradientTape() as tape:
        tape.watch(batch_t)
        out = sub_model(batch_t)   # shape (B, 8)
    # batch_jac shape: (B, 8, 3)
    batch_jac = tape.batch_jacobian(out, batch_t)
    jac_abs = tf.reduce_sum(tf.abs(batch_jac), axis=0).numpy()  # (8, 3) soma absoluta no batch
    jac_abs_sum += jac_abs
    n_used += len(batch)
    if (i // batch_size) % 5 == 0:
        print(f"  batch {i//batch_size + 1}/{(n+batch_size-1)//batch_size}", flush=True)

jac_mean_abs = jac_abs_sum / n_used   # shape (8, 3): media de |dY_j/dX_k|
print(f"  Jacobiano medio absoluto (shape (8,3)):", flush=True)
print(f"    range [{jac_mean_abs.min():.4f}, {jac_mean_abs.max():.4f}]", flush=True)

# Importancia total por feature de input (soma sobre os 8 outputs)
imp_input = jac_mean_abs.sum(axis=0)   # shape (3,)
FEATURES = ["age", "sex", "localization"]
print(f"\n  Importancia total de cada feature de INPUT (soma sobre 8 outputs):", flush=True)
for i, name in enumerate(FEATURES):
    print(f"    {name:18s} {imp_input[i]:.4f}", flush=True)

# ============================================================================
# PARTE 2 — peso da camada DE FUSAO (apos concatenate)
# ============================================================================
print("\n=== PARTE 2: Peso da camada de fusao (apos concatenate) ===", flush=True)

# Achar a primeira Dense apos o concat
fusion_layer = None
concat_output_dim = concat_layer.output.shape[-1] if concat_layer else None
print(f"  Concat output dim: {concat_output_dim}", flush=True)

# Detectar primeira Dense que recebe input com a dim do concat
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        try:
            in_shp = layer.input.shape
            if len(in_shp) == 2 and in_shp[-1] == concat_output_dim:
                fusion_layer = layer
                print(f"  Camada de fusao: {layer.name}, units={layer.units}, in_shape={in_shp}",
                      flush=True)
                break
        except Exception:
            continue

# Extrair pesos
W_fusion, b_fusion = fusion_layer.get_weights()  # shape (1544, 256) por exemplo
print(f"  W_fusion shape: {W_fusion.shape}", flush=True)

# Identificar quantas dims sao visual e quantas sao clinicas
# branch visual = primeiras N_visual dims, clinicas = ultimas 8 dims
n_clin = 8
n_visual = W_fusion.shape[0] - n_clin
print(f"  Visual: primeiras {n_visual} dims | Clinica: ultimas {n_clin} dims", flush=True)

W_visual = W_fusion[:n_visual, :]   # (n_visual, units)
W_clin   = W_fusion[n_visual:, :]    # (8, units)

# Magnitude L2 e L1 por linha (cada linha = cada dim de input)
l2_per_dim_visual = np.sqrt((W_visual ** 2).sum(axis=1))
l2_per_dim_clin   = np.sqrt((W_clin ** 2).sum(axis=1))

l2_total_visual = np.sqrt((W_visual ** 2).sum())
l2_total_clin   = np.sqrt((W_clin ** 2).sum())

mean_per_dim_visual = l2_per_dim_visual.mean()
mean_per_dim_clin   = l2_per_dim_clin.mean()

print(f"\n  L2 total dos pesos:", flush=True)
print(f"    Visual ({n_visual} dims):  {l2_total_visual:.4f}", flush=True)
print(f"    Clinica ({n_clin} dims):    {l2_total_clin:.4f}", flush=True)
print(f"    Razao visual/clinica:       {l2_total_visual/l2_total_clin:.2f}x", flush=True)

print(f"\n  L2 medio POR DIMENSAO de input (mais justo: normaliza pelo n de dims):", flush=True)
print(f"    Visual:  {mean_per_dim_visual:.4f}", flush=True)
print(f"    Clinica: {mean_per_dim_clin:.4f}", flush=True)
print(f"    Razao visual/clinica por-dim: {mean_per_dim_visual/mean_per_dim_clin:.2f}x", flush=True)

# ============================================================================
# PLOTS
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

# Plot 1: Jacobiano (3 features x 8 outputs) — peso efetivo APOS toda a branch
ax = axes[0]
vmax = jac_mean_abs.max()
im = ax.imshow(jac_mean_abs.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax,
               interpolation="nearest")
ax.set_yticks(range(3))
ax.set_yticklabels(FEATURES, fontsize=10)
ax.set_xticks(range(8))
ax.set_xticklabels([f"o{i}" for i in range(8)], fontsize=8)
ax.set_xlabel("Output da branch clinica (8 dims)", fontsize=10)
ax.set_title("Jacobiano efetivo |dY/dX|\n(propagado pela MLP completa)",
             fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
# Anotar valores
for i in range(3):
    for j in range(8):
        v = jac_mean_abs[j, i]
        c = "white" if v > vmax * 0.6 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=c)

# Plot 2: Importancia agregada das 3 features (soma sobre 8 outputs)
ax = axes[1]
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars = ax.bar(FEATURES, imp_input, color=colors, edgecolor="black", linewidth=0.6)
ax.set_ylabel("Sum |dY/dX| sobre 8 outputs", fontsize=10)
ax.set_title("Importancia efetiva por feature de input\n(branch clinica completa)",
             fontsize=10, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4)
for bar, val in zip(bars, imp_input):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
# Razao
ratio_top = max(imp_input) / min(imp_input)
ax.text(0.5, 0.95, f"Razao max/min = {ratio_top:.2f}x",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, fontstyle="italic",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))

# Plot 3: Fusao Visual vs Clinica
ax = axes[2]
labels_fus = ["Visual\n(1.536 dims\nda CNN)", "Clinica\n(8 dims da\nMLP)"]
vals_total = [l2_total_visual, l2_total_clin]
vals_per_dim = [mean_per_dim_visual, mean_per_dim_clin]
x = np.arange(2)
w_bar = 0.35
ax.bar(x - w_bar/2, vals_total, w_bar, label="L2 total", color="#1f77b4",
       edgecolor="black", linewidth=0.6)
ax2 = ax.twinx()
ax2.bar(x + w_bar/2, vals_per_dim, w_bar, label="L2 medio por dim", color="#ff7f0e",
        edgecolor="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels_fus, fontsize=9)
ax.set_ylabel("L2 total dos pesos", color="#1f77b4", fontsize=10)
ax2.set_ylabel("L2 medio por dim de input", color="#ff7f0e", fontsize=10)
ax.set_title("Peso da camada de FUSAO\n(Visual vs Clinica)",
             fontsize=10, fontweight="bold")
ax.tick_params(axis="y", labelcolor="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#ff7f0e")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.text(0.5, 1.10,
        f"Razao L2 total V/C: {l2_total_visual/l2_total_clin:.1f}x\n"
        f"Razao por-dim V/C: {mean_per_dim_visual/mean_per_dim_clin:.2f}x",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8.5, fontstyle="italic",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))

plt.tight_layout()
fig.savefig(OUT_DIR / "pesos_finais_branch_clinica.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"\nSalvo: {OUT_DIR}/pesos_finais_branch_clinica.png", flush=True)

# Sumario final
print("\n=== SUMARIO ===", flush=True)
print(f"1. Branch clinica completa: cada feature de input afeta os 8 outputs com magnitudes:", flush=True)
for i, name in enumerate(FEATURES):
    print(f"   {name:18s} sum|dY/dX| = {imp_input[i]:.3f}", flush=True)
print(f"   Razao max/min: {ratio_top:.2f}x", flush=True)
print(f"\n2. Camada de fusao: peso total atribuido a cada source", flush=True)
print(f"   Visual: L2={l2_total_visual:.2f}  |  Clinica: L2={l2_total_clin:.2f}", flush=True)
print(f"   Razao visual/clinica (TOTAL): {l2_total_visual/l2_total_clin:.1f}x", flush=True)
print(f"   Razao visual/clinica (POR DIM): {mean_per_dim_visual/mean_per_dim_clin:.2f}x", flush=True)
print(f"\n   ⚠️  L2 total visual e maior simplesmente porque tem MAIS dims (1.536 vs 8).", flush=True)
print(f"   O numero relevante e POR DIM — visual e {mean_per_dim_visual/mean_per_dim_clin:.2f}x", flush=True)
print(f"   maior por dim, mostrando que cada feature visual individual tem peso", flush=True)
print(f"   {'maior' if mean_per_dim_visual > mean_per_dim_clin else 'menor'} que cada feature clinica.",
      flush=True)
