# -*- coding: utf-8 -*-
"""Matrizes de confusao da CAMPEA (XGBoost+SMOTE sobre image-only+balance + clinico one-hot).
Gera 1 figura com 2 paineis:
  (a) 7 classes  (predicao argmax)
  (b) maligna x benigna no limiar de triagem 0,05 (ponto de operacao)
Saida: resultados/matriz_confusao_campea.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

CLASSES = ["akiec","bcc","bkl","df","mel","nv","vasc"]
MALIG = [0,1,4]; TAU = 0.05
B = Path("data/embeddings_cache/grouped"); MM, WIN = B/"multimodal", B/"imageonlybalanced"
OUT = Path("resultados"); OUT.mkdir(parents=True, exist_ok=True)

# clinico one-hot (alinhado; do multimodal)
ctr = np.vstack([np.load(MM/"X_train_clin.npy"), np.load(MM/"X_val_clin.npy")]); cte = np.load(MM/"X_test_clin.npy")
locs = np.unique(np.concatenate([ctr[:,2],cte[:,2]])).astype(int); sexs = np.unique(np.concatenate([ctr[:,1],cte[:,1]])).astype(int)
oh = lambda c: np.hstack([c[:,0:1],
    np.stack([(c[:,1].astype(int)==s)*1.0 for s in sexs],1),
    np.stack([(c[:,2].astype(int)==l)*1.0 for l in locs],1)])
Xtr = np.hstack([np.vstack([np.load(WIN/"X_train_imgfused.npy"), np.load(WIN/"X_val_imgfused.npy")]), oh(ctr)])
Xte = np.hstack([np.load(WIN/"X_test_imgfused.npy"), oh(cte)])
ytr = np.concatenate([np.load(MM/"y_train.npy"), np.load(MM/"y_val.npy")]); yte = np.load(MM/"y_test.npy")

k = min(5, int(np.bincount(ytr).min()-1))
Xb, yb = SMOTE(random_state=42, k_neighbors=max(1,k)).fit_resample(Xtr, ytr)
clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0).fit(Xb, yb)
proba = clf.predict_proba(Xte)
pred = proba.argmax(1)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# (a) 7 classes
cm7 = confusion_matrix(yte, pred, labels=range(7))
ConfusionMatrixDisplay(cm7, display_labels=CLASSES).plot(ax=ax[0], cmap="Blues", values_format="d", colorbar=False)
ax[0].set_title("(a) Campea XGBoost - 7 classes (argmax)")

# (b) maligna x benigna no limiar 0,05
ymal = np.isin(yte, MALIG).astype(int)
flag = (proba[:, MALIG].sum(1) >= TAU).astype(int)
cmb = confusion_matrix(ymal, flag, labels=[1,0])  # linhas: maligna, benigna ; col: sinalizada, liberada
ConfusionMatrixDisplay(cmb, display_labels=["sinalizada","liberada"]).plot(ax=ax[1], cmap="Greens", values_format="d", colorbar=False)
ax[1].set_yticklabels(["maligna","benigna"])
ax[1].set_title(f"(b) Maligna x benigna no limiar {TAU} (triagem)")

plt.tight_layout()
plt.savefig(OUT/"matriz_confusao_campea.png", dpi=150)
# numeros da binaria (para conferencia)
tp=cmb[0,0]; fn=cmb[0,1]; fp=cmb[1,0]; tn=cmb[1,1]
print("7 classes salvas. Binaria @0,05:")
print(f"  TP(maligna sinalizada)={tp}  FN(maligna liberada)={fn}  FP(benigna sinalizada)={fp}  TN(benigna liberada)={tn}")
print(f"  sensib={tp/(tp+fn):.3f}  especif={tn/(tn+fp):.3f}  PPV={tp/(tp+fp):.3f}")
print("salvo:", OUT/"matriz_confusao_campea.png")
