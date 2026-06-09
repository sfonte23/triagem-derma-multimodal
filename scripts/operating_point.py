# =============================================================================
# operating_point.py — Ponto de operacao clinico do pipeline CAMPEAO
# -----------------------------------------------------------------------------
# Pipeline: image-only + balanceamento na imagem (nb04) -> embeddings (256)
#           + metadados clinicos ONE-HOT -> XGBoost + SMOTE
#
# Tarefa de triagem: "QUALQUER MALIGNA" (mel + bcc + akiec) vs benignas.
# Sinaliza para encaminhamento se P(maligna) = P(mel)+P(bcc)+P(akiec) >= limiar.
#
# Gera:
#   - operating_point_malignant.csv : tabela de pontos de operacao (80/85/90%)
#   - operating_point_curve.png     : ROC (maligna vs benigna) + sens/espec x limiar
#   - testa calibracao de probabilidade (isotonica) e reporta honestamente
#
# Uso: venv/Scripts/python.exe scripts/operating_point.py
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier

# classes HAM (LabelEncoder alfabetico): akiec0 bcc1 bkl2 df3 mel4 nv5 vasc6
MALIG = [0, 1, 4]            # akiec, bcc, mel
NAMES = {4: "mel", 1: "bcc", 0: "akiec"}
B = Path("data/embeddings_cache/grouped")
WIN, MM = B / "imageonlybalanced", B / "multimodal"
OUT = Path("resultados"); OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

# --- clinico one-hot (alinhado; vem do multimodal, split byte-identico) ---
def clin_onehot():
    def c(t): return np.load(MM / f"X_{t}_clin.npy")
    ctr = np.vstack([c("train"), c("val")]); cte = c("test")
    locs = np.unique(np.concatenate([ctr[:, 2], cte[:, 2]])).astype(int)
    sexs = np.unique(np.concatenate([ctr[:, 1], cte[:, 1]])).astype(int)
    oh = lambda x: np.hstack([x[:, 0:1],
        np.stack([(x[:, 1].astype(int) == s) * 1.0 for s in sexs], 1),
        np.stack([(x[:, 2].astype(int) == l) * 1.0 for l in locs], 1)])
    return oh(ctr), oh(cte)

ohtr, ohte = clin_onehot()
Xtr = np.hstack([np.vstack([np.load(WIN / "X_train_imgfused.npy"),
                            np.load(WIN / "X_val_imgfused.npy")]), ohtr])
Xte = np.hstack([np.load(WIN / "X_test_imgfused.npy"), ohte])
ytr = np.concatenate([np.load(MM / "y_train.npy"), np.load(MM / "y_val.npy")])
yte = np.load(MM / "y_test.npy")

ymal = np.isin(yte, MALIG).astype(int)
npos, nneg = int(ymal.sum()), int((ymal == 0).sum())
print(f"Teste: {len(yte)} | malignas {npos} (mel {(yte==4).sum()}, bcc {(yte==1).sum()}, "
      f"akiec {(yte==0).sum()}) | benignas {nneg}")

# --- treina XGBoost + SMOTE (config oficial) ---
k = min(5, int(np.bincount(ytr).min() - 1))
Xb, yb = SMOTE(random_state=SEED, k_neighbors=max(1, k)).fit_resample(Xtr, ytr)
clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                    random_state=SEED, verbosity=0).fit(Xb, yb)
pb = clf.predict_proba(Xte)
pmal = pb[:, MALIG].sum(1)
auc_mal = roc_auc_score(ymal, pmal)
print(f"Acuracia global multiclasse (argmax): {accuracy_score(yte, pb.argmax(1)):.3f}")
print(f"AUC (maligna vs benigna): {auc_mal:.3f}\n")

# --- tabela de pontos de operacao ---
def metrics_at(t, scores=pmal):
    fl = scores >= t
    tp = int((fl & (ymal == 1)).sum()); fp = int((fl & (ymal == 0)).sum())
    sens = tp / npos; spec = (nneg - fp) / nneg
    acc = (tp + nneg - fp) / len(ymal); ppv = tp / (tp + fp) if (tp + fp) else 0.0
    return sens, spec, acc, ppv, int(fl.sum())

def find_thr(target, scores=pmal):
    best = None
    for t in np.round(np.arange(0.005, 0.95, 0.005), 3):
        if metrics_at(t, scores)[0] >= target:
            best = t
    return best

rows = []
for label, t in [("0,50 (padrao)", 0.50)] + \
                 [(f"sens>={int(p*100)}%", find_thr(p)) for p in (0.80, 0.85, 0.90)]:
    s, sp, a, pv, n = metrics_at(t)
    row = {"ponto": label, "limiar": round(t, 3), "sensib": round(s, 3),
           "especif": round(sp, 3), "acuracia": round(a, 3), "ppv": round(pv, 3),
           "sinalizados": n, "pct_sinalizado": round(n / len(yte), 3)}
    # quebra por classe maligna
    fl = pmal >= t
    for ci, cn in NAMES.items():
        m = (yte == ci); row[f"recall_{cn}"] = round(int((fl & m).sum()) / int(m.sum()), 3)
    rows.append(row)

tab = pd.DataFrame(rows)
tab.to_csv(OUT / "operating_point_malignant.csv", index=False)
print("=== Pontos de operacao (QUALQUER MALIGNA) ===")
print(tab.to_string(index=False))

# --- CALIBRACAO de probabilidade (isotonica) — comparacao honesta ---
# Fit em SMOTE(train70), calibra em val10 (distribuicao real), avalia em test.
Xtr70 = np.hstack([np.load(WIN / "X_train_imgfused.npy"), ohtr[:len(np.load(WIN/"X_train_imgfused.npy"))]])
ytr70 = np.load(MM / "y_train.npy")
Xval = np.hstack([np.load(WIN / "X_val_imgfused.npy"),
                  ohtr[len(np.load(WIN/"X_train_imgfused.npy")):]])
yval = np.load(MM / "y_val.npy")
kb = min(5, int(np.bincount(ytr70).min() - 1))
Xb70, yb70 = SMOTE(random_state=SEED, k_neighbors=max(1, kb)).fit_resample(Xtr70, ytr70)
base = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                     random_state=SEED, verbosity=0).fit(Xb70, yb70)
pmal_unc = base.predict_proba(Xte)[:, MALIG].sum(1)
print("\n=== Calibracao de probabilidade (isotonica) — efeito ===")
try:
    from sklearn.frozen import FrozenEstimator   # sklearn >= 1.6 (substitui cv='prefit')
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(Xval, yval)
    pmal_cal = cal.predict_proba(Xte)[:, MALIG].sum(1)
    for nome, sc in [("sem calibrar", pmal_unc), ("calibrado", pmal_cal)]:
        t = find_thr(0.85, sc); s, sp, a, pv, n = metrics_at(t, sc)
        print(f"  {nome:14} AUC {roc_auc_score(ymal, sc):.3f} | @sens85%: limiar {t:.3f} "
              f"espec {sp*100:.0f}% PPV {pv*100:.0f}%")
    print("  (calibracao e ~monotonica: muda o VALOR do limiar e a leitura da prob.,")
    print("   nao o trade-off sensib/espec — a discriminacao vem do modelo/AUC.)")
except Exception as e:
    print(f"  [calibracao pulada: {e}]")

# --- figura para o artigo ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
fpr, tpr, _ = roc_curve(ymal, pmal)
ax[0].plot(fpr, tpr, color="#1B4480", lw=2, label=f"AUC = {auc_mal:.3f}")
ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
for p in (0.80, 0.85, 0.90):
    s, sp, *_ = metrics_at(find_thr(p))
    ax[0].plot(1 - sp, s, "o", color="#6FA539", ms=7)
    ax[0].annotate(f"{int(p*100)}%", (1 - sp, s), textcoords="offset points",
                   xytext=(6, -4), fontsize=8)
ax[0].set_xlabel("1 - Especificidade (FPR)"); ax[0].set_ylabel("Sensibilidade (TPR)")
ax[0].set_title("ROC — maligna vs benigna"); ax[0].legend(loc="lower right"); ax[0].grid(alpha=0.3)

ts = np.linspace(0.005, 0.6, 120)
sens = [metrics_at(t)[0] for t in ts]; spec = [metrics_at(t)[1] for t in ts]
ax[1].plot(ts, sens, color="#1B4480", lw=2, label="Sensibilidade")
ax[1].plot(ts, spec, color="#6FA539", lw=2, label="Especificidade")
ax[1].axvline(find_thr(0.85), color="red", ls="--", lw=1, label="Ponto 85%")
ax[1].set_xlabel("Limiar de decisao P(maligna)"); ax[1].set_ylabel("Taxa")
ax[1].set_title("Sensibilidade x Especificidade x Limiar"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "operating_point_curve.png", dpi=150)
print(f"\nSalvos: {OUT}/operating_point_malignant.csv + operating_point_curve.png")
