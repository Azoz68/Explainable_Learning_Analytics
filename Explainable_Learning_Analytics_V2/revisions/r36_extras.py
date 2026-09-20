import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
import sys, os, json, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
BASE = str(DATA_DIR)
PREP = str(PREPROCESSED_CSV)
OUT = os.path.join(str(RESULTS_DIR), "R3-6"); RS = 42; CN = ['Distinction', 'Fail', 'Pass', 'Withdrawn']; D, F, P, W = 0, 1, 2, 3; T0 = time.time()
df = pd.read_csv(PREP); target = 'target_result'
CAT = ["gender", "region", "highest_education", "imd_band", "age_band", "disability", "code_module", "code_presentation"]
fc = [c for c in df.columns if c not in {"id_student", "final_result", target}]; X_raw, y = df[fc].copy(), df[target].astype(int)
Xt, Xte, yt_, yte = train_test_split(X_raw, y, test_size=0.20, stratify=y, random_state=RS); Xtr, Xva, ytr, yva = train_test_split(Xt, yt_, test_size=0.20, stratify=yt_, random_state=RS)
Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
for c in CAT:
    le = LabelEncoder(); le.fit(Xtr[c].astype(str)); seen = set(le.classes_)
    for s in (Xtr, Xva, Xte): s[c] = le.transform(s[c].astype(str).apply(lambda v: v if v in seen else le.classes_[0]))
Xtr, Xva, Xte = [s.fillna(0).astype(float) for s in (Xtr, Xva, Xte)]
sc = StandardScaler(); X_train = pd.DataFrame(sc.fit_transform(Xtr), columns=fc, index=Xtr.index); X_val = pd.DataFrame(sc.transform(Xva), columns=fc, index=Xva.index); X_test = pd.DataFrame(sc.transform(Xte), columns=fc, index=Xte.index)
yt, yv, ytr_ = yte.values, yva.values, ytr.values
tp = pd.read_csv(os.path.join(OUT, "test_probabilities.csv")); assert (tp["df_index"].values == X_test.index.values).all()
P4 = tp[[f"xgb4_{c}" for c in CN]].values; Pw = tp[[f"xgbw_{c}" for c in CN]].values; Ph = tp[[f"htbt_{c}" for c in CN]].values; sb = tp["bin_xgb_atrisk"].values
yb = np.isin(yt, [F, W]).astype(int); s4 = P4[:, F] + P4[:, W]

shares = {"dataset": {CN[c]: int((y.values == c).sum()) for c in range(4)}, "test": {CN[c]: int((yt == c).sum()) for c in range(4)},
          "dataset_share": {CN[c]: float((y.values == c).mean()) for c in range(4)}, "at_risk_share_dataset": float(np.isin(y.values, [F, W]).mean()), "at_risk_share_test": float(yb.mean())}
json.dump(shares, open(os.path.join(OUT, "class_shares.json"), "w"), indent=1)

rng = np.random.RandomState(RS); B = 2000; n = len(yt); contrasts = {k: [] for k in ["weighted_minus_ref_balanced_accuracy", "weighted_minus_ref_macro_f1", "weighted_minus_ref_accuracy", "derived_minus_binary_roc_auc", "derived_minus_binary_pr_auc", "xgb_minus_htbt_macro_f1", "xgb_minus_htbt_accuracy", "weighted_minus_ref_recall_Fail", "weighted_minus_ref_recall_Distinction", "weighted_minus_ref_recall_Pass"]}
p4, pw, ph = P4.argmax(1), Pw.argmax(1), Ph.argmax(1)
for _ in range(B):
    i = rng.randint(0, n, n); yi = yt[i]
    contrasts["weighted_minus_ref_balanced_accuracy"].append(balanced_accuracy_score(yi, pw[i]) - balanced_accuracy_score(yi, p4[i]))
    contrasts["weighted_minus_ref_macro_f1"].append(f1_score(yi, pw[i], average='macro') - f1_score(yi, p4[i], average='macro'))
    contrasts["weighted_minus_ref_accuracy"].append(accuracy_score(yi, pw[i]) - accuracy_score(yi, p4[i]))
    ybi = yb[i]; contrasts["derived_minus_binary_roc_auc"].append(roc_auc_score(ybi, s4[i]) - roc_auc_score(ybi, sb[i])); contrasts["derived_minus_binary_pr_auc"].append(average_precision_score(ybi, s4[i]) - average_precision_score(ybi, sb[i]))
    contrasts["xgb_minus_htbt_macro_f1"].append(f1_score(yi, p4[i], average='macro') - f1_score(yi, ph[i], average='macro')); contrasts["xgb_minus_htbt_accuracy"].append(accuracy_score(yi, p4[i]) - accuracy_score(yi, ph[i]))
    for c, nm in [(F, "Fail"), (D, "Distinction"), (P, "Pass")]: contrasts[f"weighted_minus_ref_recall_{nm}"].append(recall_score(yi == c, pw[i] == c) - recall_score(yi == c, p4[i] == c))
ci = {k: {"point": float(v_pt), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v, v_pt in [
    ("weighted_minus_ref_balanced_accuracy", contrasts["weighted_minus_ref_balanced_accuracy"], balanced_accuracy_score(yt, pw) - balanced_accuracy_score(yt, p4)),
    ("weighted_minus_ref_macro_f1", contrasts["weighted_minus_ref_macro_f1"], f1_score(yt, pw, average='macro') - f1_score(yt, p4, average='macro')),
    ("weighted_minus_ref_accuracy", contrasts["weighted_minus_ref_accuracy"], accuracy_score(yt, pw) - accuracy_score(yt, p4)),
    ("derived_minus_binary_roc_auc", contrasts["derived_minus_binary_roc_auc"], roc_auc_score(yb, s4) - roc_auc_score(yb, sb)),
    ("derived_minus_binary_pr_auc", contrasts["derived_minus_binary_pr_auc"], average_precision_score(yb, s4) - average_precision_score(yb, sb)),
    ("xgb_minus_htbt_macro_f1", contrasts["xgb_minus_htbt_macro_f1"], f1_score(yt, p4, average='macro') - f1_score(yt, ph, average='macro')),
    ("xgb_minus_htbt_accuracy", contrasts["xgb_minus_htbt_accuracy"], accuracy_score(yt, p4) - accuracy_score(yt, ph)),
    ("weighted_minus_ref_recall_Fail", contrasts["weighted_minus_ref_recall_Fail"], recall_score(yt == F, pw == F) - recall_score(yt == F, p4 == F)),
    ("weighted_minus_ref_recall_Distinction", contrasts["weighted_minus_ref_recall_Distinction"], recall_score(yt == D, pw == D) - recall_score(yt == D, p4 == D)),
    ("weighted_minus_ref_recall_Pass", contrasts["weighted_minus_ref_recall_Pass"], recall_score(yt == P, pw == P) - recall_score(yt == P, p4 == P))]}
ci["note"] = "percentile bootstrap over test enrolments (B=2000, seed 42), single training run per model; training variability not included"
json.dump(ci, open(os.path.join(OUT, "paired_bootstrap_contrasts.json"), "w"), indent=1); print("paired CIs:", {k: (round(v["point"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in ci.items() if isinstance(v, dict)})

def brier_sum(yy, Pm): return float(np.mean(np.sum((Pm - np.eye(4)[yy]) ** 2, 1)))
def ece(yy, Pm, bins=15, equal_mass=False):
    conf = Pm.max(1); hit = (Pm.argmax(1) == yy).astype(float)
    edges = np.quantile(conf, np.linspace(0, 1, bins + 1)) if equal_mass else np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.sum(): e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)
prev = np.bincount(ytr_, minlength=4) / len(ytr_); Pclim = np.tile(prev, (n, 1))
cal = []
for name, Pm in [("XGBoost (reference)", P4), ("HTBT", Ph), ("XGBoost + balanced class weights", Pw), ("training-prevalence predictor (baseline)", Pclim)]:
    cal.append({"model": name, "brier_sum_over_classes": brier_sum(yt, Pm), "brier_mean_over_classes": brier_sum(yt, Pm) / 4, "ece_top_label_15_equal_width": ece(yt, Pm), "ece_top_label_15_equal_mass": ece(yt, Pm, equal_mass=True), "mean_confidence": float(Pm.max(1).mean()), "accuracy": accuracy_score(yt, Pm.argmax(1))})
pd.DataFrame(cal).to_csv(os.path.join(OUT, "table_calibration_v2.csv"), index=False); print(pd.DataFrame(cal).round(4).to_string(index=False))

m_es = xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", n_estimators=1000, learning_rate=0.1, max_depth=6, random_state=RS, n_jobs=-1, verbosity=0, early_stopping_rounds=30)
m_es.fit(X_train, ytr_, eval_set=[(X_val, yv)], verbose=False); Pes = m_es.predict_proba(X_test); pes = Pes.argmax(1)
es = {"best_iteration": int(m_es.best_iteration), "accuracy": accuracy_score(yt, pes), "balanced_accuracy": balanced_accuracy_score(yt, pes), "macro_f1": f1_score(yt, pes, average='macro'),
      "macro_ovr_auc": roc_auc_score(pd.get_dummies(yt), Pes, average='macro', multi_class='ovr'), "brier_sum": brier_sum(yt, Pes), "ece_15_equal_width": ece(yt, Pes), "at_risk_auc": roc_auc_score(yb, Pes[:, F] + Pes[:, W]),
      "reference_200_rounds": {"accuracy": accuracy_score(yt, p4), "macro_f1": f1_score(yt, p4, average='macro'), "macro_ovr_auc": roc_auc_score(pd.get_dummies(yt), P4, average='macro', multi_class='ovr'), "brier_sum": brier_sum(yt, P4)}}
json.dump(es, open(os.path.join(OUT, "xgb_early_stopped_variant.json"), "w"), indent=1); print("early-stopped variant:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in es.items() if k != "reference_200_rounds"})

model = joblib.load(os.path.join(BASE, "models", "xgb_final.joblib")); Pv4 = model.predict_proba(X_val)
def bfac(): return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=200, learning_rate=0.1, max_depth=6, random_state=RS, n_jobs=-1, verbosity=0)
ybtr = np.isin(ytr_, [F, W]).astype(int); mb = bfac().fit(X_train, ybtr); mbw = bfac().fit(X_train, ybtr, sample_weight=compute_sample_weight('balanced', ybtr))
pd.DataFrame({"df_index": X_val.index.values, "y_true": yv, **{f"xgb4_{CN[c]}": Pv4[:, c] for c in range(4)}, "bin_xgb_atrisk": mb.predict_proba(X_val)[:, 1], "bin_xgbw_atrisk": mbw.predict_proba(X_val)[:, 1]}).to_csv(os.path.join(OUT, "val_probabilities.csv"), index=False)
json.dump({"xgboost": xgb.__version__, "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}, open(os.path.join(OUT, "versions.json"), "w"), indent=1)
print(f"EXTRAS DONE in {time.time()-T0:.0f}s")
