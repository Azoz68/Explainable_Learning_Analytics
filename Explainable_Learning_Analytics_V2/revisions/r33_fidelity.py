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

ks_rows = []; w_by_k = {}
for k in KS:
    w, H, fs, fl = adaptive(S, L[0], k); w_by_k[k] = w
    dS, dL, dH = ablate_drop(S, k, median_b), ablate_drop(L[0], k, median_b), ablate_drop(H, k, median_b)
    for subset, idx in [("full test (n=6519)", np.arange(N)), ("paper subset (n=50)", pos50)]:
        ks_rows.append({"k": k, "subset": subset, "SHAP_mean_drop": dS[idx].mean(), "LIME_mean_drop": dL[idx].mean(), "Hybrid_mean_drop": dH[idx].mean(),
                        "w_bar": w[idx].mean(), "share_SHAP_dominant(w>0.6)": (w[idx] > 0.6).mean(), "share_LIME_dominant(w<0.4)": (w[idx] < 0.4).mean(),
                        "share_w_equal_0.5(both_zero)": (np.abs(w[idx] - 0.5) < 1e-12).mean(), "ordering_H>=S>L": bool(dH[idx].mean() >= dS[idx].mean() > dL[idx].mean())})
for r in ks_rows:
    r["spearman_w_vs_k5"] = spearmanr(w_by_k[r["k"]], w_by_k[5]).correlation if r["subset"].startswith("full") else np.nan
pd.DataFrame(ks_rows).to_csv(os.path.join(OUT, "table_R1-2_k_sensitivity.csv"), index=False)
print("R1-2 done", time.time() - T0)

K = 5; per_seed = {}
for sd in SEEDS:
    w, H, fs, fl = adaptive(S, L[sd], K)
    per_seed[sd] = {"SHAP": ablate_drop(S, K, median_b), "LIME": ablate_drop(L[sd], K, median_b), "Hybrid": ablate_drop(H, K, median_b), "w": w, "fl": fl}
D = {m: np.mean([per_seed[sd][m] for sd in SEEDS], axis=0) for m in ["SHAP", "LIME", "Hybrid"]}
Wm = np.mean([per_seed[sd]["w"] for sd in SEEDS], axis=0); fl0 = np.mean([(per_seed[sd]["fl"] == 0) for sd in SEEDS], axis=0)
rng = np.random.RandomState(RS); B = 10000
def bci(v, stat=np.mean):
    idx = rng.randint(0, len(v), (B, len(v))); s = stat(v[idx], axis=1); return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
def q25(a, axis=None): return np.percentile(a, 25, axis=axis)
rows = []
for m in ["SHAP", "LIME", "Hybrid"]:
    v = D[m]; lo, hi = bci(v); qlo, qhi = bci(v, q25)
    rows.append({"method": m, "n": N, "seeds": len(SEEDS), "mean_drop": v.mean(), "ci_low": lo, "ci_high": hi, "sd": v.std(ddof=1), "p25": np.percentile(v, 25), "p25_ci_low": qlo, "p25_ci_high": qhi,
                 "median": np.median(v), "max": v.max(), "min": v.min(), "cross_seed_sd_of_mean": np.std([per_seed[sd][m].mean() for sd in SEEDS], ddof=1)})

    rows.append({"method": m + " (paper n=50 subset)", "n": 50, "seeds": len(SEEDS), "mean_drop": v[pos50].mean(), "sd": v[pos50].std(ddof=1), "median": np.median(v[pos50]), "p25": np.percentile(v[pos50], 25), "max": v[pos50].max(), "min": v[pos50].min()})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_R3-3_fidelity_by_method.csv"), index=False)
def rank_biserial(d): pos = (d > 0).sum(); neg = (d < 0).sum(); return (pos - neg) / max(1, pos + neg)
prs = []
for a, b in [("Hybrid", "SHAP"), ("Hybrid", "LIME"), ("SHAP", "LIME")]:
    d = D[a] - D[b]; lo, hi = bci(d); nz = d[d != 0]
    w_std = wilcoxon(D[a], D[b], zero_method='wilcox').pvalue if len(nz) else 1.0; w_pratt = wilcoxon(D[a], D[b], zero_method='pratt').pvalue
    prs.append({"comparison": f"{a} - {b}", "mean_diff": d.mean(), "ci_low": lo, "ci_high": hi, "share_ties": float((d == 0).mean()), "share_a_greater": float((d > 0).mean()), "share_a_smaller": float((d < 0).mean()),
                "wilcoxon_p_pratt": w_pratt, "wilcoxon_p_wilcox": w_std, "rank_biserial": rank_biserial(d), "cohen_dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0})
ps = [r["wilcoxon_p_pratt"] for r in prs]; order = np.argsort(ps)
for rank, i in enumerate(order): prs[i]["wilcoxon_p_pratt_holm"] = min(1.0, ps[i] * (3 - rank))
pd.DataFrame(prs).to_csv(os.path.join(OUT, "table_R3-3_paired_tests.csv"), index=False)
wstats = {"w_bar": float(Wm.mean()), "w_bar_ci": bci(Wm), "share_SHAP_dominant": float((Wm > 0.6).mean()), "share_SHAP_dominant_ci": bci((Wm > 0.6).astype(float)),
          "share_LIME_dominant": float((Wm < 0.4).mean()), "share_LIME_dominant_ci": bci((Wm < 0.4).astype(float)), "share_fidL_zero_regime": float(fl0.mean()),
          "paper50_w_bar": float(Wm[pos50].mean()), "paper50_share_SHAP_dominant": float((Wm[pos50] > 0.6).mean()), "paper50_share_LIME_dominant": float((Wm[pos50] < 0.4).mean())}
json.dump(wstats, open(os.path.join(OUT, "R3-3_adaptive_weight_stats.json"), "w"), indent=1)
print("R3-3 done", time.time() - T0, {m: round(D[m].mean(), 4) for m in D})

Lm = np.mean([L[sd] for sd in SEEDS], axis=0)
_, H5, _, _ = adaptive(S, Lm, 5)
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
noise_b_rng = np.random.RandomState(RS)
crows = []; curves = {}
for name, scores in comp.items():
    dAUC, iAUC, dcurve, icurve = del_ins_auc(scores); curves[name] = (dcurve, icurve)
    idx5 = topk_idx(scores, 5); Xk = np.tile(median_b, (N, 1)); Xk[np.arange(N)[:, None], idx5] = Xte_np[np.arange(N)[:, None], idx5]
    suff = pf(Xk)[np.arange(N), pred] / p0
    Xn = Xte_np.copy(); Xn[np.arange(N)[:, None], idx5] = X_train.values[noise_b_rng.randint(0, len(X_train), (N, 5)), idx5]
    crows.append({"method": name, "deletion_AUC_lower_better": float(dAUC.mean()), "deletion_ci": bci(dAUC), "insertion_AUC_higher_better": float(iAUC.mean()), "insertion_ci": bci(iAUC),
                  "sufficiency_top5_higher_better": float(suff.mean()), "sufficiency_ci": bci(suff),
                  "top5_drop_median_baseline(selection criterion)": float(ablate_drop(scores, 5, median_b).mean()),
                  "top5_drop_mean_baseline": float(ablate_drop(scores, 5, mean_b).mean()),
                  "top5_drop_noise_baseline": float((p0 - pf(Xn)[np.arange(N), pred]).mean())})
    print(f"  R3-2 {name}: del={dAUC.mean():.4f} ins={iAUC.mean():.4f} suff={suff.mean():.4f}", flush=True)
pd.DataFrame(crows).to_csv(os.path.join(OUT, "table_R3-2_independent_criteria.csv"), index=False)

dA_h, _, _, _ = del_ins_auc(H5); dA_s, _, _, _ = del_ins_auc(S); _, iA_h, _, _ = del_ins_auc(H5); _, iA_s, _, _ = del_ins_auc(S)
json.dump({"deletion_AUC_hybrid_minus_SHAP": {"mean": float((dA_h - dA_s).mean()), "ci": bci(dA_h - dA_s), "wilcoxon_p": float(wilcoxon(dA_h, dA_s, zero_method='pratt').pvalue), "rank_biserial": rank_biserial(dA_s - dA_h)},
           "insertion_AUC_hybrid_minus_SHAP": {"mean": float((iA_h - iA_s).mean()), "ci": bci(iA_h - iA_s), "wilcoxon_p": float(wilcoxon(iA_h, iA_s, zero_method='pratt').pvalue), "rank_biserial": rank_biserial(iA_h - iA_s)},
           "alpha_from_validation": alpha_val}, open(os.path.join(OUT, "R3-2_hybrid_vs_shap_independent.json"), "w"), indent=1)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]; GRID = "#e1e0d9"
ks = pd.DataFrame(ks_rows); full = ks[ks.subset.str.startswith("full")]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
for m, col in [("SHAP", C[0]), ("LIME", C[1]), ("Hybrid", C[2])]: ax[0].plot(full.k, full[f"{m}_mean_drop"], "-o", color=col, lw=2, ms=4, label=m)
ax[0].set(xlabel="k (features ablated)", ylabel="mean probability drop", title="(a) fidelity vs k, full test set"); ax[0].legend(frameon=False)
ax[1].plot(full.k, full.w_bar, "-o", color=C[0], lw=2, ms=4, label="mean SHAP weight w̄"); ax[1].plot(full.k, full["share_SHAP_dominant(w>0.6)"], "-s", color=C[1], lw=2, ms=4, label="share SHAP-dominant")
ax[1].plot(full.k, full["share_LIME_dominant(w<0.4)"], "-^", color=C[2], lw=2, ms=4, label="share LIME-dominant"); ax[1].axhline(0.5, ls="--", lw=1, color="#898781")
ax[1].set(xlabel="k", ylabel="value", title="(b) adaptive weights vs k", ylim=(0, 1)); ax[1].legend(frameon=False, fontsize=8)
for a in ax: a.set_xticks(KS); a.grid(color=GRID, lw=0.6); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_R1-2_k_sensitivity.png"), dpi=200); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9)); xs = np.arange(nF + 1)
for (name, (dc, ic)), col in zip(curves.items(), C * 2):
    if name.startswith("Fixed-alpha") and "0.5" not in name: continue
    ax[0].plot(xs, dc, color=col, lw=1.8, label=name); ax[1].plot(xs, ic, color=col, lw=1.8, label=name)
ax[0].set(xlabel="features removed (attribution order)", ylabel="mean predicted-class probability", title="(a) deletion curve (lower = more faithful)")
ax[1].set(xlabel="features inserted (attribution order)", ylabel="mean predicted-class probability", title="(b) insertion curve (higher = more faithful)"); ax[1].legend(frameon=False, fontsize=7)
for a in ax: a.grid(color=GRID, lw=0.6); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_R3-2_deletion_insertion.png"), dpi=200); plt.close(fig)
print(f"ALL DONE in {time.time()-T0:.0f}s -> {OUT}")
