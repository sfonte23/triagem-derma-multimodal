# =============================================================================
# _threshold_tuning.py  —  Calibracao de limiar de decisao para triagem
# -----------------------------------------------------------------------------
# Responde ao apontamento da banca: recall de Melanoma de 32,7% (limiar argmax
# padrao) e inseguro para triagem. Demonstra que rebaixando o limiar de decisao
# sobre P(melanoma) o sistema atinge sensibilidade clinica (>=85-90%) ao custo
# de especificidade — comportamento desejavel para "segunda opiniao".
#
# IMPORTANTE: usa os embeddings em cache do pipeline ATUAL (split por imagem,
# com data leakage por lesion_id). Os numeros sao um PREVIEW e devem ser
# regerados apos o retreino com split agrupado por lesion_id.
#
# Reproduz fielmente a config do artigo: XGBoost (random_state=42),
# SMOTE(k_neighbors=5) balanceando todas as classes ate a majoritaria.
# =============================================================================
import numpy as np
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL = CLASSES.index("mel")   # 4
CACHE = Path("data/embeddings_cache")

# --- Carregar embeddings fundidos em cache ---
X_train = np.load(CACHE / "X_train_fused.npy")
X_test  = np.load(CACHE / "X_test_fused.npy")
y_train = np.load(CACHE / "y_train.npy")
y_test  = np.load(CACHE / "y_test.npy")
print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")
print(f"Melanomas no teste: {(y_test == MEL).sum()} de {len(y_test)}")

# --- SMOTE (k=5, balanceia todas as classes a majoritaria) ---
unique, counts = np.unique(y_train, return_counts=True)
k = min(5, int(counts.min() - 1))
sm = SMOTE(random_state=42, k_neighbors=max(1, k))
X_bal, y_bal = sm.fit_resample(X_train, y_train)
print(f"Apos SMOTE: {X_bal.shape}")

# --- XGBoost (config identica ao artigo) ---
clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                    random_state=42, verbosity=0)
clf.fit(X_bal, y_bal)
proba = clf.predict_proba(X_test)
p_mel = proba[:, MEL]
y_pred_argmax = proba.argmax(axis=1)

# --- Sanity check: recall argmax deve bater com os 32,7% do artigo ---
recall_argmax = recall_score(y_test, y_pred_argmax, labels=[MEL], average="macro")
acc_argmax = accuracy_score(y_test, y_pred_argmax)
print(f"\n[Sanity] Recall mel (argmax multiclasse): {recall_argmax:.3f} "
      f"(artigo: 0,327) | Acuracia global: {acc_argmax:.3f}")

# --- Triagem binaria melanoma-vs-resto: varredura de limiar sobre P(mel) ---
y_true_mel = (y_test == MEL).astype(int)
n_mel = y_true_mel.sum()
n_neg = len(y_true_mel) - n_mel

def metrics_at(t):
    flag = (p_mel >= t).astype(int)
    tp = int(((flag == 1) & (y_true_mel == 1)).sum())
    fp = int(((flag == 1) & (y_true_mel == 0)).sum())
    sens = tp / n_mel                      # recall melanoma (sensibilidade)
    spec = (n_neg - fp) / n_neg            # especificidade
    ppv  = tp / (tp + fp) if (tp + fp) else 0.0
    return sens, spec, ppv, flag.sum()

print("\n" + "=" * 72)
print("VARREDURA DE LIMIAR — Triagem melanoma-vs-resto (XGBoost + SMOTE)")
print("=" * 72)
print(f"{'Limiar':>8} | {'Sensib.(recall mel)':>20} | {'Especif.':>9} | "
      f"{'PPV':>6} | {'Sinalizados':>11}")
print("-" * 72)
for t in [0.50, 0.30, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03]:
    sens, spec, ppv, n_flag = metrics_at(t)
    print(f"{t:>8.2f} | {sens*100:>18.1f}% | {spec*100:>7.1f}% | "
          f"{ppv*100:>4.1f}% | {n_flag:>11}")

# --- Limiares que atingem alvos clinicos de sensibilidade ---
print("\n" + "=" * 72)
print("LIMIAR PARA ALVOS DE SENSIBILIDADE CLINICA")
print("=" * 72)
order = np.argsort(-p_mel)  # do maior p_mel para o menor
for target in [0.85, 0.90, 0.95]:
    # menor limiar (maior) que ainda garante sensibilidade >= target
    best_t = None
    for t in np.round(np.arange(0.005, 0.50, 0.005), 3):
        sens, spec, ppv, n_flag = metrics_at(t)
        if sens >= target:
            best_t = (t, sens, spec, ppv, n_flag)
    if best_t:
        t, sens, spec, ppv, n_flag = best_t
        print(f"  Sensib. >= {target*100:.0f}%  ->  limiar={t:.3f} | "
              f"recall={sens*100:.1f}% | especif.={spec*100:.1f}% | "
              f"PPV={ppv*100:.1f}% | sinalizados={n_flag}/{len(y_test)}")
    else:
        print(f"  Sensib. >= {target*100:.0f}%  ->  inatingivel nesta faixa")
