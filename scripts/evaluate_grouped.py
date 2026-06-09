# =============================================================================
# evaluate_grouped.py  —  Pipeline classico sobre embeddings SEM data leakage
# -----------------------------------------------------------------------------
# Consome os embeddings exportados pelos 3 notebooks Colab (split agrupado por
# lesion_id) e gera tudo que a revisao da banca pediu:
#   - Metricas globais (XGB/RF/SVM/NB) base e +SMOTE  (tab globais + smote)
#   - Recall por classe maligna (mel, bcc)            (ponto 4)
#   - Validacao cruzada StratifiedGroupKFold por lesao (ponto 3.2)
#   - Bootstrap IC95% para recall mel/bcc
#   - Varredura de limiar (PR) p/ triagem melanoma     (ponto 4)
#   - Ablacao: fused(256) vs gap(1536) vs clin(3) + multimodal vs image-only (ponto 1)
# Saida: resultados/
#
# LAYOUT ESPERADO (baixe do Drive para ca):
#   data/embeddings_cache/grouped/
#       multimodal/   <- nb01 (X_*_fused.npy, X_*_gap.npy, X_*_clin.npy, y_*.npy, lesion_*.npy)
#       imageonly/    <- nb02 (X_*_imgfused.npy, X_*_gap.npy, y_*.npy, lesion_*.npy)
#       imgbalanced/  <- nb03 (X_*_fused.npy, X_*_gap.npy, X_*_clin.npy, y_*.npy, lesion_*.npy)
#
# Uso:  venv/Scripts/python.exe scripts/evaluate_grouped.py
# =============================================================================
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL, BCC = CLASSES.index("mel"), CLASSES.index("bcc")
ROOT = Path("data/embeddings_cache/grouped")
OUT  = Path("resultados")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

# subpastas e o nome do arquivo de feature "fundida" (256-d) de cada modelo
MODELS = {
    "multimodal":        {"folder": "multimodal",        "fused": "fused"},
    "imageonly":         {"folder": "imageonly",         "fused": "imgfused"},
    "imgbalanced":       {"folder": "imgbalanced",       "fused": "fused"},
    "imageonlybalanced": {"folder": "imageonlybalanced", "fused": "imgfused"},
}

def _clf(name):
    if name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                             random_state=SEED, verbosity=0)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=SEED)
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=SEED)
    if name == "Naive Bayes":
        return GaussianNB()
    raise ValueError(name)

# -----------------------------------------------------------------------------
def load_feature(folder: Path, feat: str):
    """Retorna (X_fit, y_fit, g_fit, X_te, y_te, g_te, X_all, y_all, g_all).
    X_fit = train+val (80%); X_te = test (20%, held-out por lesao)."""
    def L(tag, kind):
        return np.load(folder / f"X_{tag}_{kind}.npy")
    def Ly(tag): return np.load(folder / f"y_{tag}.npy")
    def Lg(tag): return np.load(folder / f"lesion_{tag}.npy", allow_pickle=True)

    Xtr = np.vstack([L("train", feat), L("val", feat)])
    ytr = np.concatenate([Ly("train"), Ly("val")])
    gtr = np.concatenate([Lg("train"), Lg("val")])
    Xte, yte, gte = L("test", feat), Ly("test"), Lg("test")
    X_all = np.vstack([Xtr, Xte]); y_all = np.concatenate([ytr, yte])
    g_all = np.concatenate([gtr, gte])
    return Xtr, ytr, gtr, Xte, yte, gte, X_all, y_all, g_all

def metrics(yte, pred, proba=None):
    rec = recall_score(yte, pred, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    out = {"acc": accuracy_score(yte, pred),
           "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
           "recall_mel": rec[MEL], "recall_bcc": rec[BCC]}
    if proba is not None and proba.shape[1] == len(CLASSES):
        try:
            out["auc_ovr"] = roc_auc_score(yte, proba, multi_class="ovr", average="macro")
        except Exception:
            out["auc_ovr"] = np.nan
    else:
        out["auc_ovr"] = np.nan
    return out

def fit_eval(Xtr, ytr, Xte, yte, smote=False):
    if smote:
        k = min(5, int(np.bincount(ytr).min() - 1))
        Xtr, ytr = SMOTE(random_state=SEED, k_neighbors=max(1, k)).fit_resample(Xtr, ytr)
    rows, xgb_proba = {}, None
    for name in ["XGBoost", "Random Forest", "SVM", "Naive Bayes"]:
        clf = _clf(name).fit(Xtr, ytr)
        proba = clf.predict_proba(Xte) if hasattr(clf, "predict_proba") else None
        pred = clf.predict(Xte)
        rows[name] = metrics(yte, pred, proba)
        if name == "XGBoost":
            xgb_proba = proba
    return pd.DataFrame(rows).T, xgb_proba

# -----------------------------------------------------------------------------
def threshold_sweep(proba, yte):
    p_mel = proba[:, MEL]
    y_mel = (yte == MEL).astype(int)
    n_pos, n_neg = y_mel.sum(), (y_mel == 0).sum()
    def at(t):
        flag = (p_mel >= t).astype(int)
        tp = int(((flag == 1) & (y_mel == 1)).sum())
        fp = int(((flag == 1) & (y_mel == 0)).sum())
        return tp / n_pos, (n_neg - fp) / n_neg, int(flag.sum())
    rows = []
    for t in [0.50, 0.30, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03]:
        s, e, n = at(t); rows.append({"limiar": t, "recall_mel": s, "especif": e, "sinalizados": n})
    targets = {}
    for tgt in [0.85, 0.90]:
        best = None
        for t in np.round(np.arange(0.005, 0.50, 0.005), 3):
            s, e, n = at(t)
            if s >= tgt: best = {"limiar": t, "recall_mel": s, "especif": e, "sinalizados": n}
        targets[f">={int(tgt*100)}%"] = best
    return pd.DataFrame(rows), targets

def cv_grouped(X_all, y_all, g_all):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name in ["XGBoost", "Random Forest", "SVM", "Naive Bayes"]:
        accs, f1s, aucs = [], [], []
        for tr, va in sgkf.split(X_all, y_all, groups=g_all):
            clf = _clf(name).fit(X_all[tr], y_all[tr])
            pred = clf.predict(X_all[va])
            accs.append(accuracy_score(y_all[va], pred))
            f1s.append(f1_score(y_all[va], pred, average="macro", zero_division=0))
            if hasattr(clf, "predict_proba"):
                try:
                    aucs.append(roc_auc_score(y_all[va], clf.predict_proba(X_all[va]),
                                              multi_class="ovr", average="macro"))
                except Exception:
                    pass
        rows.append({"modelo": name,
                     "acc": f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
                     "f1_macro": f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
                     "auc_ovr": (f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}" if aucs else "—")})
    return pd.DataFrame(rows)

def bootstrap_ci(Xtr, ytr, Xte, yte, smote=True, n=1000):
    if smote:
        k = min(5, int(np.bincount(ytr).min() - 1))
        Xtr, ytr = SMOTE(random_state=SEED, k_neighbors=max(1, k)).fit_resample(Xtr, ytr)
    clf = _clf("XGBoost").fit(Xtr, ytr)
    pred = clf.predict(Xte)
    rng = np.random.default_rng(SEED)
    rec_mel, rec_bcc = [], []
    idx_all = np.arange(len(yte))
    for _ in range(n):
        bs = rng.choice(idx_all, size=len(idx_all), replace=True)
        yt, pr = yte[bs], pred[bs]
        if (yt == MEL).any():
            rec_mel.append(recall_score(yt, pr, labels=[MEL], average="macro", zero_division=0))
        if (yt == BCC).any():
            rec_bcc.append(recall_score(yt, pr, labels=[BCC], average="macro", zero_division=0))
    def ci(v): return (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5))
    return {"mel": ci(rec_mel), "bcc": ci(rec_bcc)}

# -----------------------------------------------------------------------------
def main():
    summary = {}
    for mname, cfg in MODELS.items():
        folder = ROOT / cfg["folder"]
        if not (folder / f"X_test_{cfg['fused']}.npy").exists():
            print(f"[skip] {mname}: nao encontrado em {folder}")
            continue
        print("\n" + "#" * 72 + f"\n# MODELO: {mname}\n" + "#" * 72)
        Xtr, ytr, gtr, Xte, yte, gte, X_all, y_all, g_all = load_feature(folder, cfg["fused"])
        print(f"Fit(train+val): {Xtr.shape} | Teste: {Xte.shape} | mel no teste: {(yte==MEL).sum()}")

        base, _        = fit_eval(Xtr, ytr, Xte, yte, smote=False)
        smote_df, prob = fit_eval(Xtr, ytr, Xte, yte, smote=True)
        print("\n-- Base (sem SMOTE) --\n", base.round(3))
        print("\n-- +SMOTE --\n", smote_df.round(3))
        base.to_csv(OUT / f"{mname}_globais_base.csv")
        smote_df.to_csv(OUT / f"{mname}_globais_smote.csv")

        cv = cv_grouped(X_all, y_all, g_all)
        print("\n-- CV StratifiedGroupKFold (por lesao) --\n", cv.to_string(index=False))
        cv.to_csv(OUT / f"{mname}_cv_grouped.csv", index=False)

        boot = bootstrap_ci(Xtr, ytr, Xte, yte, smote=True)
        print(f"\n-- Bootstrap IC95% (XGB+SMOTE) -- mel={boot['mel']} | bcc={boot['bcc']}")

        sweep, tgts = threshold_sweep(prob, yte)
        print("\n-- Varredura de limiar (XGB+SMOTE) --\n", sweep.round(3).to_string(index=False))
        print("   alvos:", {k: (None if v is None else {kk: round(vv,3) for kk,vv in v.items()}) for k,v in tgts.items()})
        sweep.to_csv(OUT / f"{mname}_threshold.csv", index=False)

        summary[mname] = {
            "xgb_smote": smote_df.loc["XGBoost"].to_dict(),
            "bootstrap_mel": boot["mel"], "threshold_targets": tgts,
        }

    # ---- Ablacao: fused vs gap vs clin (multimodal) + multimodal vs image-only ----
    print("\n" + "=" * 72 + "\n ABLACAO — contribuicao de cada modalidade (XGBoost+SMOTE)\n" + "=" * 72)
    abl = []
    mm = ROOT / "multimodal"
    if (mm / "X_test_fused.npy").exists():
        for feat, label in [("fused", "multimodal (img+clin)"), ("gap", "image-only (GAP 1536)"),
                            ("clin", "clinical-only (3)")]:
            if not (mm / f"X_test_{feat}.npy").exists():
                continue
            Xtr, ytr, _, Xte, yte, *_ = load_feature(mm, feat)
            df, _ = fit_eval(Xtr, ytr, Xte, yte, smote=True)
            r = df.loc["XGBoost"].to_dict(); r["config"] = label; abl.append(r)
    io = ROOT / "imageonly"
    if (io / "X_test_imgfused.npy").exists():
        Xtr, ytr, _, Xte, yte, *_ = load_feature(io, "imgfused")
        df, _ = fit_eval(Xtr, ytr, Xte, yte, smote=True)
        r = df.loc["XGBoost"].to_dict(); r["config"] = "image-only (modelo treinado sem clinico)"; abl.append(r)
    if abl:
        abl_df = pd.DataFrame(abl).set_index("config")
        print(abl_df.round(3))
        abl_df.to_csv(OUT / "ablacao.csv")

    # ---- Fusao do clinico NO CLASSIFICADOR (one-hot) — proposta corrigida ----
    # O clinico (age_z, sex, loc_code) vem do multimodal (split byte-identico a
    # todos os modelos), one-hot de sex+loc. Compara imagem pura vs imagem+clinico.
    print("\n" + "=" * 72 + "\n FUSAO CLINICA @ CLASSIFICADOR (one-hot) — XGBoost+SMOTE\n" + "=" * 72)
    mmf = ROOT / "multimodal"
    if (mmf / "X_test_clin.npy").exists():
        def _clin(tag): return np.load(mmf / f"X_{tag}_clin.npy")
        c_tr = np.vstack([_clin("train"), _clin("val")]); c_te = _clin("test")
        locs = np.unique(np.concatenate([c_tr[:, 2], c_te[:, 2]])).astype(int)
        sexs = np.unique(np.concatenate([c_tr[:, 1], c_te[:, 1]])).astype(int)
        def onehot(c):
            age = c[:, 0:1]
            sx = np.stack([(c[:, 1].astype(int) == s).astype(float) for s in sexs], 1)
            lc = np.stack([(c[:, 2].astype(int) == l).astype(float) for l in locs], 1)
            return np.hstack([age, sx, lc])
        clin_tr, clin_te = onehot(c_tr), onehot(c_te)
        fus = []
        for mname, cfg in MODELS.items():
            folder = ROOT / cfg["folder"]
            if not (folder / f"X_test_{cfg['fused']}.npy").exists():
                continue
            Xtr, ytr, _, Xte, yte, *_ = load_feature(folder, cfg["fused"])
            d0, _ = fit_eval(Xtr, ytr, Xte, yte, smote=True)
            d1, _ = fit_eval(np.hstack([Xtr, clin_tr]), ytr,
                             np.hstack([Xte, clin_te]), yte, smote=True)
            r0 = d0.loc["XGBoost"].to_dict(); r0["config"] = f"{mname}: imagem"
            r1 = d1.loc["XGBoost"].to_dict(); r1["config"] = f"{mname}: imagem+clinico(one-hot)"
            fus += [r0, r1]
        fus_df = pd.DataFrame(fus).set_index("config")
        print(fus_df.round(3))
        fus_df.to_csv(OUT / "fusao_clinica.csv")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
    print("\nResultados salvos em:", OUT)

if __name__ == "__main__":
    main()
