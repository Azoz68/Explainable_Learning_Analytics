import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
import sys, os, json, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
from scipy.stats import wilcoxon, spearmanr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = str(DATA_DIR)
PREP = str(PREPROCESSED_CSV)
OUT = os.path.join(str(RESULTS_DIR), "R3-3_R3-2_R1-2"); os.makedirs(OUT, exist_ok=True)
CK   = os.path.join(OUT, "cache"); os.makedirs(CK, exist_ok=True)
RS = 42; SEEDS = [0, 1, 2, 3, 4]; KS = [3, 5, 7, 10, 15]; CN = ['Distinction', 'Fail', 'Pass', 'Withdrawn']
T0 = time.time()

df = pd.read_csv(PREP); target = 'target_result'
CAT = ["gender", "region", "highest_education", "imd_band", "age_band", "disability", "code_module", "code_presentation"]
fc = [c for c in df.columns if c not in {"id_student", "final_result", target}]; nF = len(fc)
X_raw, y = df[fc].copy(), df[target].astype(int)
Xt, Xte, yt_, yte = train_test_split(X_raw, y, test_size=0.20, stratify=y, random_state=RS)
Xtr, Xva, ytr, yva = train_test_split(Xt, yt_, test_size=0.20, stratify=yt_, random_state=RS)
Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
for c in CAT:
    le = LabelEncoder(); le.fit(Xtr[c].astype(str)); seen = set(le.classes_)
    for s in (Xtr, Xva, Xte): s[c] = le.transform(s[c].astype(str).apply(lambda v: v if v in seen else le.classes_[0]))
Xtr, Xva, Xte = [s.fillna(0).astype(float) for s in (Xtr, Xva, Xte)]
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(Xtr), columns=fc, index=Xtr.index)
X_val = pd.DataFrame(sc.transform(Xva), columns=fc, index=Xva.index); X_test = pd.DataFrame(sc.transform(Xte), columns=fc, index=Xte.index)
yt = yte.values; model = joblib.load(os.path.join(BASE, "models", "xgb_final.joblib")); assert list(model.feature_names_in_) == fc
def pf(A): return model.predict_proba(pd.DataFrame(A, columns=fc))
Xte_np = X_test.values; N = len(Xte_np); P0 = pf(Xte_np); pred = P0.argmax(1); p0 = P0[np.arange(N), pred]
assert abs((pred == yt).mean() - 0.5762) < 0.002
median_b = X_train.median().values; mean_b = X_train.mean().values
paper50 = X_test.sample(n=50, random_state=RS).index; pos50 = np.array([X_test.index.get_loc(i) for i in paper50])
print(f"test n={N}; paper subset n=50 located; setup {time.time()-T0:.0f}s")

f_shap = os.path.join(CK, "shap_signed.npy")
if os.path.exists(f_shap): S = np.load(f_shap)
else:
    sv = np.array(shap.TreeExplainer(model).shap_values(X_test)); S = sv[np.arange(N), :, pred] if sv.ndim == 3 else sv
    np.save(f_shap, S)
print(f"SHAP {S.shape} ready {time.time()-T0:.0f}s")

L = {}
for sd in SEEDS:
    f = os.path.join(CK, f"lime_seed{sd}.npy")
    if os.path.exists(f): L[sd] = np.load(f); print(f"  LIME seed {sd} cached"); continue
    expl = LimeTabularExplainer(training_data=X_train.values, feature_names=fc, class_names=[str(c) for c in range(4)], mode="classification", random_state=sd)
    M = np.zeros((N, nF)); t1 = time.time()
    for i in range(N):
        e = expl.explain_instance(Xte_np[i], pf, num_features=nF, labels=(int(pred[i]),))
        for fi, w in e.local_exp[int(pred[i])]: M[i, fi] = w
        if i % 500 == 0 and i: print(f"  LIME seed {sd}: {i}/{N} ({(time.time()-t1)/i:.2f}s/inst)", flush=True)
    np.save(f, M); L[sd] = M; print(f"  LIME seed {sd} done in {time.time()-t1:.0f}s", flush=True)

def l1(v): a = np.abs(v); return a / (a.sum(1, keepdims=True) + 1e-9)
def topk_idx(scores, k): return np.argsort(np.abs(scores), axis=1)[:, ::-1][:, :k]
def ablate_drop(scores, k, baseline):
    idx = topk_idx(scores, k); Xa = Xte_np.copy(); Xa[np.arange(N)[:, None], idx] = baseline[idx]
    return p0 - pf(Xa)[np.arange(N), pred]
def adaptive(S_, L_, k, baseline=median_b):
    fs = np.maximum(0, ablate_drop(S_, k, baseline)); fl = np.maximum(0, ablate_drop(L_, k, baseline)); tot = fs + fl
    w = np.where(tot > 0, fs / np.where(tot > 0, tot, 1), 0.5); H = w[:, None] * l1(S_) + (1 - w)[:, None] * l1(L_)
    return w, H, fs, fl

Lm = np.mean([L[sd] for sd in SEEDS], axis=0)
_, H5, fs5, fl5 = adaptive(S, Lm, 5)
rng = np.random.RandomState(RS); B = 10000
def bci(v, stat=np.mean):
    idx = rng.randint(0, len(v), (B, len(v))); s_ = stat(v[idx], axis=1); return float(np.percentile(s_, 2.5)), float(np.percentile(s_, 97.5))
def rank_biserial(d): pos = (d > 0).sum(); neg = (d < 0).sum(); return (pos - neg) / max(1, pos + neg)

comp = {"SHAP": S, "LIME": Lm, "Adaptive hybrid": H5}
for a in [0.25, 0.5, 0.75]: comp[f"Fixed-alpha hybrid (alpha={a})"] = a * l1(S) + (1 - a) * l1(Lm)

f_val = os.path.join(CK, "lime_val500_seed0.npy"); vidx = np.random.RandomState(RS).choice(len(X_val), 500, replace=False); Xv = X_val.values[vidx]
Pv = pf(Xv); pv_pred = Pv.argmax(1)
if os.path.exists(f_val): Lv = np.load(f_val)
else:
    expl = LimeTabularExplainer(training_data=X_train.values, feature_names=fc, class_names=[str(c) for c in range(4)], mode="classification", random_state=0); Lv = np.zeros((500, nF))
    for i in range(500):
        e = expl.explain_instance(Xv[i], pf, num_features=nF, labels=(int(pv_pred[i]),))
        for fi, w in e.local_exp[int(pv_pred[i])]: Lv[i, fi] = w
    np.save(f_val, Lv)
svv = np.array(shap.TreeExplainer(model).shap_values(pd.DataFrame(Xv, columns=fc))); Sv = svv[np.arange(500), :, pv_pred]
pv0 = Pv[np.arange(500), pv_pred]
def drop_on(Xb, pb, predb, scores, k, baseline):
    idx = topk_idx(scores, k); Xa = Xb.copy(); Xa[np.arange(len(Xb))[:, None], idx] = baseline[idx]; return pb - pf(Xa)[np.arange(len(Xb)), predb]
fsv = np.maximum(0, drop_on(Xv, pv0, pv_pred, Sv, 5, median_b)); flv = np.maximum(0, drop_on(Xv, pv0, pv_pred, Lv, 5, median_b)); totv = fsv + flv
alpha_val = float(np.where(totv > 0, fsv / np.where(totv > 0, totv, 1), 0.5).mean())
comp[f"Validation-estimated global alpha ({alpha_val:.3f})"] = alpha_val * l1(S) + (1 - alpha_val) * l1(Lm)
pi = permutation_importance(model, X_val, yva.values, n_repeats=5, random_state=RS, scoring="neg_log_loss", n_jobs=-1).importances_mean
comp["Permutation importance (global ranking)"] = np.tile(pi, (N, 1))
rnd = np.random.RandomState(RS).rand(N, nF); comp["Random ranking (control)"] = rnd
def del_ins_auc(scores, baseline=median_b, steps=nF):
    order = np.argsort(np.abs(scores), axis=1)[:, ::-1]; dele = np.zeros((N, steps + 1)); ins = np.zeros((N, steps + 1))
    Xd = Xte_np.copy(); Xi = np.tile(baseline, (N, 1)); dele[:, 0] = p0; ins[:, 0] = pf(Xi)[np.arange(N), pred]
    for s in range(steps):
        j = order[:, s]; Xd[np.arange(N), j] = baseline[j]; Xi[np.arange(N), j] = Xte_np[np.arange(N), j]
        dele[:, s + 1] = pf(Xd)[np.arange(N), pred]; ins[:, s + 1] = pf(Xi)[np.arange(N), pred]
    return (dele / p0[:, None]).mean(1), (ins / p0[:, None]).mean(1), dele.mean(0), ins.mean(0)

ex = {}
def suff5(scores):
    idx5 = topk_idx(scores, 5); Xk = np.tile(median_b, (N, 1)); Xk[np.arange(N)[:, None], idx5] = Xte_np[np.arange(N)[:, None], idx5]; return pf(Xk)[np.arange(N), pred] / p0
crit = {}
for name in ["Adaptive hybrid", "LIME", "Fixed-alpha hybrid (alpha=0.25)", "SHAP"]:
    dA, iA, _, _ = del_ins_auc(comp[name]); crit[name] = {"del": dA, "ins": iA, "suf": suff5(comp[name])}
def contrast(a, b, key, better_low=False):
    d = crit[a][key] - crit[b][key]; d = -d if better_low else d
    return {"mean": float(d.mean()), "ci": bci(d), "wilcoxon_p": float(wilcoxon(crit[a][key], crit[b][key], zero_method="pratt").pvalue), "sign_effect": rank_biserial(d)}
for b in ["LIME", "Fixed-alpha hybrid (alpha=0.25)", "SHAP"]:
    ex[f"hybrid_minus_{b}"] = {"deletion_AUC_advantage(lower_better,sign_flipped)": contrast("Adaptive hybrid", b, "del", True), "insertion_AUC_advantage": contrast("Adaptive hybrid", b, "ins"), "sufficiency_advantage": contrast("Adaptive hybrid", b, "suf")}

reg = {"only_LIME_zero_w1": [], "only_SHAP_zero_w0": [], "both_zero_w05": []}
for sd in SEEDS:
    w, H, fs, fl = adaptive(S, L[sd], 5); reg["only_LIME_zero_w1"].append(((fl == 0) & (fs > 0)).mean()); reg["only_SHAP_zero_w0"].append(((fs == 0) & (fl > 0)).mean()); reg["both_zero_w05"].append(((fs == 0) & (fl == 0)).mean())
ex["weight_regimes_k5_seed_avg"] = {k: float(np.mean(v)) for k, v in reg.items()}

def jacc(A_, B_, k):
    ta, tb = topk_idx(A_, k), topk_idx(B_, k); return float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(ta, tb)]))
ex["jaccard_topk"] = {f"k{k}": {"SHAP_vs_LIME": jacc(S, Lm, k), "hybrid_vs_LIME": jacc(H5 if k == 5 else adaptive(S, Lm, k)[1], Lm, k), "hybrid_vs_SHAP": jacc(H5 if k == 5 else adaptive(S, Lm, k)[1], S, k)} for k in [5, 15]}

rs = np.array([spearmanr(np.abs(S[i]), np.abs(Lm[i])).correlation for i in range(N)]); rs = rs[~np.isnan(rs)]
ex["shap_lime_rank_consistency"] = {"n": int(len(rs)), "mean": float(rs.mean()), "median": float(np.median(rs)), "ci_mean": bci(rs), "share_positive": float((rs > 0).mean())}

Xn = lambda scores: (lambda idx5: (lambda X_: (X_.__setitem__((np.arange(N)[:, None], idx5), X_train.values[np.random.RandomState(RS).randint(0, len(X_train), (N, 5)), idx5]) or X_))(Xte_np.copy()))(topk_idx(scores, 5))
dn_s = p0 - pf(Xn(S))[np.arange(N), pred]; dn_l = p0 - pf(Xn(Lm))[np.arange(N), pred]; dn_h = p0 - pf(Xn(H5))[np.arange(N), pred]
ex["noise_baseline_means"] = {"SHAP": float(dn_s.mean()), "LIME": float(dn_l.mean()), "hybrid": float(dn_h.mean())}
ex["noise_baseline_SHAP_minus_LIME"] = {"mean": float((dn_s - dn_l).mean()), "ci": bci(dn_s - dn_l), "wilcoxon_p": float(wilcoxon(dn_s, dn_l, zero_method="pratt").pvalue)}
json.dump(ex, open(os.path.join(OUT, "R3-2_provenance_extras.json"), "w"), indent=1); print("EXTRAS2 DONE", json.dumps(ex)[:3000])
