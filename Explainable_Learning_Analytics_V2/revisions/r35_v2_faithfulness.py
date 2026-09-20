import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
import sys, os, json, math, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon, rankdata
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score, f1_score, roc_auc_score, recall_score
import xgboost as xgb, torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r35_attention_faithfulness.py"), encoding="utf-8").read()
exec(src[src.find("class PositionalEncoding"):src.find("# ---------------- data exactly")])
BASE = str(DATA_DIR)
PREP = str(PREPROCESSED_CSV)
OUT = os.path.join(str(RESULTS_DIR), "R3-5"); RS = 42; SEQ = 30; CN = ['Distinction', 'Fail', 'Pass', 'Withdrawn']; T0 = time.time()
df = pd.read_csv(PREP); target = 'target_result'
sf = [c for c in df.columns if c not in ['id_student', 'code_module', 'code_presentation', target, 'final_result']]; y = df[target].astype(int).values
tv, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=RS); tr, va = train_test_split(tv, test_size=0.2, stratify=y[tv], random_state=RS)
Xs = df[sf].fillna(0).reset_index(drop=True).copy()
for col in Xs.select_dtypes(include='object').columns:
    le = LabelEncoder(); le.fit(Xs.iloc[tr][col].astype(str)); seen = set(le.classes_); Xs[col] = Xs[col].astype(str).apply(lambda v: int(le.transform([v])[0]) if v in seen else -1)
arr = Xs.values.astype(np.float32); sc = StandardScaler().fit(arr[tr]); Xall = sc.transform(arr).astype(np.float32)
vle = pd.read_csv(os.path.join(BASE, "studentVle.csv"), usecols=['id_student', 'code_module', 'code_presentation', 'date', 'sum_click'])
vle['date'] = pd.to_numeric(vle['date'], errors='coerce').fillna(0).astype(int); vle = vle[(vle['date'] >= 0) & (vle['date'] < SEQ)]
key = lambda fr: fr['id_student'].astype(str) + '_' + fr['code_module'].astype(str) + '_' + fr['code_presentation'].astype(str)
k2i = {k: i for i, k in enumerate(key(df))}; vle['k'] = key(vle); g = vle.groupby(['k', 'date'])['sum_click'].sum().reset_index()
seqs = np.zeros((len(df), SEQ), dtype=np.float32); ki = g['k'].map(k2i); ok = ki.notna(); seqs[ki[ok].astype(int).values, g.loc[ok, 'date'].values] = g.loc[ok, 'sum_click'].values; seqs = np.log1p(seqs); del vle, g
model = HTBT(32, SEQ); model.load_state_dict(torch.load(os.path.join(BASE, "models_htbt", "htbt_best.pt"), map_location='cpu', weights_only=False)); model.eval()
te = np.sort(te); Xte, Ste, yte = Xall[te], seqs[te], y[te]; N = len(te)
def fwd(Xb, Sb):
    out = []
    with torch.no_grad():
        for s in range(0, len(Xb), 512): lg, _, _ = model(torch.from_numpy(Xb[s:s + 512]), torch.from_numpy(Sb[s:s + 512])); out.append(torch.softmax(lg, 1).numpy())
    return np.vstack(out)
P0 = fwd(Xte, Ste); pred = P0.argmax(1); p0 = P0[np.arange(N), pred]
A = np.load(os.path.join(OUT, "attention_test.npy")); imp = np.load(os.path.join(OUT, "occlusion_impact_test.npy"))
active = Ste > 0; n_active = active.sum(1)
print(f"sanity acc={accuracy_score(yte, pred):.4f}; share of inactive cells {1 - active.mean():.3f}; students with all-empty sequence {(n_active == 0).mean():.3f}")

def kerby_rb(a, b):
    d = a - b; d = d[d != 0]
    if len(d) == 0: return 0.0
    r = rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())
def occlude(S, idx):
    S2 = S.copy(); S2[np.arange(len(S))[:, None], idx] = 0.0; return S2
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); m = len(ps)
    for rank, i in enumerate(o): adj[i] = min(1.0, ps[i] * (m - rank))
    return np.maximum.accumulate(adj[o])[np.argsort(o)]

sel = np.where(n_active >= 6)[0]; Ns = len(sel); rng = np.random.RandomState(RS)
A_act = np.where(active, A, -np.inf)
top3 = np.argsort(A_act, axis=1)[:, ::-1][:, :3]; A_act_low = np.where(active, A, np.inf); bot3 = np.argsort(A_act_low, axis=1)[:, :3]
rand3 = np.stack([rng.choice(np.where(active[i])[0], 3, replace=False) if n_active[i] >= 3 else np.array([0, 1, 2]) for i in range(N)])
d_top = (p0 - fwd(Xte, occlude(Ste, top3))[np.arange(N), pred])[sel]
d_bot = (p0 - fwd(Xte, occlude(Ste, bot3))[np.arange(N), pred])[sel]
d_rnd = (p0 - fwd(Xte, occlude(Ste, rand3))[np.arange(N), pred])[sel]
d_all = (p0 - fwd(Xte, np.zeros_like(Ste))[np.arange(N), pred])[sel]
tests = {}
for name, a, b in [("top3_vs_random3", d_top, d_rnd), ("top3_vs_bottom3", d_top, d_bot), ("bottom3_vs_random3", d_bot, d_rnd)]:
    tests[name] = {"mean_diff": float((a - b).mean()), "wilcoxon_p_pratt": float(wilcoxon(a, b, zero_method='pratt').pvalue), "rank_biserial_kerby": kerby_rb(a, b)}
ps = holm([tests[k]["wilcoxon_p_pratt"] for k in tests])
for k, p_ in zip(tests, ps): tests[k]["wilcoxon_p_holm"] = float(p_)
occ = pd.DataFrame([{"condition": n_, "n": Ns, "mean_drop": v.mean(), "mean_abs_drop": np.abs(v).mean(), "median": np.median(v), "p90_abs": np.percentile(np.abs(v), 90),
                     "share_abs_gt_0.01": (np.abs(v) > 0.01).mean(), "flip_rate": None} for n_, v in [("top-3 attended active days", d_top), ("bottom-3 attended active days", d_bot), ("random 3 active days", d_rnd), ("all 30 days", d_all)]])
for cond, idx in [("top-3 attended active days", top3), ("bottom-3 attended active days", bot3), ("random 3 active days", rand3)]:
    occ.loc[occ.condition == cond, "flip_rate"] = float((fwd(Xte, occlude(Ste, idx)).argmax(1) != pred)[sel].mean())
occ.loc[occ.condition == "all 30 days", "flip_rate"] = float((fwd(Xte, np.zeros_like(Ste)).argmax(1) != pred)[sel].mean())
occ.to_csv(os.path.join(OUT, "v2_table_occlusion_active.csv"), index=False); json.dump({"n_students_ge6_active": int(Ns), "tests": tests}, open(os.path.join(OUT, "v2_occlusion_tests.json"), "w"), indent=1)
print("occlusion (active-day restricted):", occ[["condition", "mean_drop", "mean_abs_drop"]].round(4).values.tolist(), tests)

def gxi_compute():
    G = np.zeros((N, SEQ))
    for s in range(0, N, 512):
        xb = torch.from_numpy(Xte[s:s + 512]); sb = torch.from_numpy(Ste[s:s + 512]).requires_grad_(True)
        lg, _, _ = model(xb, sb); selv = lg[torch.arange(len(sb)), torch.from_numpy(pred[s:s + 512])]
        G[s:s + 512] = (torch.autograd.grad(selv.sum(), sb)[0] * sb).detach().numpy()
    return G
gxi = gxi_compute(); np.save(os.path.join(OUT, "v2_gxi_test.npy"), gxi)
def rho_active(X1, X2, min_active=4):
    out = np.full(N, np.nan)
    for i in range(N):
        m = active[i]
        if m.sum() >= min_active and X1[i][m].std() > 0 and X2[i][m].std() > 0: out[i] = spearmanr(X1[i][m], X2[i][m]).correlation
    return out
r_att_occ = rho_active(A, imp); r_gxi_occ = rho_active(np.abs(gxi), imp); r_gxi_att = rho_active(np.abs(gxi), A)
def summ(r):
    v = r[~np.isnan(r)]; return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(v.mean()), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))], "share_positive": float((v > 0).mean()), "wilcoxon_p_vs_zero": float(wilcoxon(v, zero_method='pratt').pvalue) if len(v) else None}
rho_v2 = {"attention_vs_occlusion_active_days": summ(r_att_occ), "gxi_vs_occlusion_active_days": summ(r_gxi_occ), "gxi_vs_attention_active_days": summ(r_gxi_att),
          "full_vector_attention_vs_occlusion_median_for_reference": float(np.nanmedian([spearmanr(A[i], imp[i]).correlation if imp[i].std() > 0 else np.nan for i in range(N)])),
          "day_level_spearman_mean_attention_vs_share_active": float(spearmanr(A.mean(0), active.mean(0)).correlation),
          "per_instance_max_attention": {"median": float(np.median(A.max(1))), "p90": float(np.percentile(A.max(1), 90)), "share_ge_0.10": float((A.max(1) >= 0.10).mean()), "uniform": 1 / SEQ}}
json.dump(rho_v2, open(os.path.join(OUT, "v2_rho_stats.json"), "w"), indent=1); print("rho v2:", {k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in rho_v2.items()})

def summary(Pm):
    pr = Pm.argmax(1); return {"accuracy": accuracy_score(yte, pr), "macro_f1": f1_score(yte, pr, average='macro'), "macro_ovr_auc": roc_auc_score(pd.get_dummies(yte), Pm, average='macro', multi_class='ovr'),
                               "pred_changed": float((pr != pred).mean()), **{f"recall_{CN[c]}": float(recall_score(yte == c, pr == c)) for c in range(4)}}
rows = [{"condition": "original", "seeds": 1, **summary(P0)}, {"condition": "all 30 days zeroed", "seeds": 1, **summary(fwd(Xte, np.zeros_like(Ste)))}]
for cond in ["day order permuted within student", "sequences swapped between students"]:
    per = []
    for sd in range(5):
        r_ = np.random.RandomState(sd); S2 = np.stack([s[r_.permutation(SEQ)] for s in Ste]) if cond.startswith("day") else Ste[r_.permutation(N)]; per.append(summary(fwd(Xte, S2)))
    rows.append({"condition": cond, "seeds": 5, **{k: float(np.mean([p[k] for p in per])) for k in per[0]}, **{k + "_sd": float(np.std([p[k] for p in per], ddof=1)) for k in ["accuracy", "macro_f1", "pred_changed"]}})
rows.append({"condition": "static features at training mean (degenerate: constant Fail prediction)", "seeds": 1, **summary(fwd(np.zeros_like(Xte), Ste))})

def factory(): return xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", n_estimators=200, learning_rate=0.1, max_depth=6, random_state=RS, n_jobs=-1, verbosity=0)
m_seq = factory().fit(seqs[tr], y[tr]); rows.append({"condition": "XGBoost on the 30-day sequence only (retrained)", "seeds": 1, **summary(m_seq.predict_proba(seqs[te]))})
Xcomb = np.hstack([Xall, seqs]); m_comb = factory().fit(Xcomb[tr], y[tr]); rows.append({"condition": "XGBoost on 32 static features + 30-day sequence (retrained)", "seeds": 1, **summary(m_comb.predict_proba(Xcomb[te]))})
m_stat = factory().fit(Xall[tr], y[tr]); rows.append({"condition": "XGBoost on the 32 static features only (retrained, same encoding as HTBT)", "seeds": 1, **summary(m_stat.predict_proba(Xall[te]))})
abl = pd.DataFrame(rows); abl.to_csv(os.path.join(OUT, "v2_table_branch_ablation.csv"), index=False); print(abl[["condition", "accuracy", "macro_f1", "recall_Distinction"]].round(4).to_string(index=False))

Z = StandardScaler().fit_transform(A); base = KMeans(n_clusters=3, random_state=RS, n_init=10).fit(Z).labels_
km_rows = [{"setting": "n_init=10, seed 42 (this run)", "sizes": sorted(np.bincount(base).tolist(), reverse=True), "ari_vs_base": 1.0}]
for sd in range(5):
    lab = KMeans(n_clusters=3, random_state=sd, n_init=10).fit(Z).labels_; km_rows.append({"setting": f"n_init=10, seed {sd}", "sizes": sorted(np.bincount(lab).tolist(), reverse=True), "ari_vs_base": float(adjusted_rand_score(base, lab))})
lab = KMeans(n_clusters=3, random_state=RS).fit(Z).labels_; km_rows.append({"setting": "sklearn default n_init, seed 42 (notebook convention)", "sizes": sorted(np.bincount(lab).tolist(), reverse=True), "ari_vs_base": float(adjusted_rand_score(base, lab))})
clus = []
for c in range(3):
    idx = base == c; clus.append({"cluster": c, "n": int(idx.sum()), "mean_active_days": float(n_active[idx].mean()), "share_all_empty": float((n_active[idx] == 0).mean()), "peak_attention_day": int(A[idx].mean(0).argmax()),
                                  "share_active_after_day15": float(active[idx][:, 15:].mean()), "share_at_risk": float(np.isin(yte[idx], [1, 3]).mean()), "dominant_true_outcome": CN[int(np.bincount(yte[idx], minlength=4).argmax())]})
mA = A.mean(0); t7 = {"day_indexing": "position d = OULAD presentation day d, window days 0-29; weights averaged over the 4 heads", "most_attended_positions_mean_profile": np.argsort(mA)[::-1][:3].tolist(), "highest_mean_weights": [float(mA[i]) for i in np.argsort(mA)[::-1][:3]],
      "least_attended_positions": np.argsort(mA)[:3].tolist(), "lowest_mean_weights": [float(mA[i]) for i in np.argsort(mA)[:3]], "uniform_reference": 1 / SEQ, "per_instance_peakedness": rho_v2["per_instance_max_attention"],
      "kmeans": km_rows, "clusters_base": clus, "share_top3_attended_days_inactive": float((~np.take_along_axis(active, np.argsort(A, 1)[:, ::-1][:, :3], 1)).mean())}
json.dump(t7, open(os.path.join(OUT, "v2_table7_attention.json"), "w"), indent=1); print("Table 7 v2:", t7["most_attended_positions_mean_profile"], [r["sizes"] for r in km_rows[:3]], "ARI", [round(r["ari_vs_base"], 3) for r in km_rows])

ex = {}
ex["kmeans_single_init_seeds"] = []
for sd in [42, 0, 1, 2, 3, 4]:
    lab = KMeans(n_clusters=3, random_state=sd, n_init=1).fit(Z).labels_; ex["kmeans_single_init_seeds"].append({"seed": sd, "sizes": sorted(np.bincount(lab).tolist(), reverse=True), "ari_vs_base": float(adjusted_rand_score(base, lab))})
ex["rho_attention_occlusion_by_active_days"] = []
for a_, b_ in [(4, 5), (6, 9), (10, 19), (20, 30)]:
    m_ = (n_active >= a_) & (n_active <= b_) & ~np.isnan(r_att_occ); ex["rho_attention_occlusion_by_active_days"].append({"active_days": f"{a_}-{b_}", "n": int(m_.sum()), "median": float(np.median(r_att_occ[m_])), "share_positive": float((r_att_occ[m_] > 0).mean())})
rs_ = spearmanr(A.mean(0), active.mean(0)); ex["day_level_spearman"] = {"rho": float(rs_.correlation), "p": float(rs_.pvalue), "n_days": SEQ}
ex["all30_mean_drop_all_students"] = float((p0 - fwd(Xte, np.zeros_like(Ste))[np.arange(N), pred]).mean()); ex["all30_mean_abs_drop_ge6"] = float(np.abs(d_all).mean())
r_att_clk = rho_active(A, Ste); ex["rho_attention_vs_logclicks_active_days"] = {"n": int((~np.isnan(r_att_clk)).sum()), "median": float(np.nanmedian(r_att_clk))}
topc3 = np.argsort(np.where(active, Ste, -np.inf), axis=1)[:, ::-1][:, :3]; d_clk = (p0 - fwd(Xte, occlude(Ste, topc3))[np.arange(N), pred])[sel]
ex["occlude_top3_highest_click_active_days"] = {"mean_drop": float(d_clk.mean()), "top3_attended_mean_drop": float(d_top.mean()), "wilcoxon_p_top_att_vs_top_click": float(wilcoxon(d_top, d_clk, zero_method='pratt').pvalue),
    "rank_biserial_top_att_minus_top_click": kerby_rb(d_top, d_clk), "mean_logclicks_top3_attended": float(np.take_along_axis(Ste, top3, 1)[sel].mean()), "mean_logclicks_random3": float(np.take_along_axis(Ste, rand3, 1)[sel].mean())}
json.dump(ex, open(os.path.join(OUT, "v2_review_extras.json"), "w"), indent=1); print("review extras:", json.dumps(ex)[:1500])

np.save(os.path.join(OUT, "v2_rho_attention_occlusion_active.npy"), r_att_occ); np.save(os.path.join(OUT, "v2_kmeans_labels_base.npy"), base); np.save(os.path.join(OUT, "v2_mean_attention_profile.npy"), A.mean(0))
_cols = ["#2a78d6", "#eb6834", "#e34948"]; _order = np.argsort(-np.bincount(base))
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3.8))
for k, c in enumerate(_order):
    idx = base == c; m_ = A[idx].mean(0); s_ = A[idx].std(0)
    ax2[0].plot(range(SEQ), m_, color=_cols[k], lw=2, label=f"cluster {k + 1} (n = {int(idx.sum()):,})"); ax2[0].fill_between(range(SEQ), m_ - s_, m_ + s_, color=_cols[k], alpha=0.15)
    ax2[1].plot(range(SEQ), active[idx].mean(0), color=_cols[k], lw=2, label=f"cluster {k + 1}")
ax2[0].axhline(1 / SEQ, ls="--", lw=1, color="#898781"); ax2[0].set(xlabel="day in window", ylabel="attention (mean, band = 1 SD)", title="(a) mean attention profile by cluster"); ax2[0].legend(frameon=False, fontsize=8)
ax2[1].set(xlabel="day in window", ylabel="share of students active", title="(b) activity by cluster"); ax2[1].legend(frameon=False, fontsize=8)
for a in ax2: a.grid(color="#e1e0d9", lw=0.6); a.spines[["top", "right"]].set_visible(False)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "v2_fig14_attention_clusters.png"), dpi=200); plt.close(fig2)

C1, C2, C3, GRID = "#2a78d6", "#eb6834", "#e34948", "#e1e0d9"
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
ax[0].plot(range(SEQ), mA, "-o", color=C1, ms=3, lw=2, label="mean attention"); ax[0].axhline(1 / SEQ, ls="--", lw=1, color="#898781"); ax[0].set(xlabel="day in window", ylabel="attention", title="(a) mean attention per day")
ax[1].plot(range(SEQ), active.mean(0), "-o", color=C2, ms=3, lw=2); ax[1].set(xlabel="day in window", ylabel="share of students active", title="(b) activity per day (test set)")
v = r_att_occ[~np.isnan(r_att_occ)]; ax[2].hist(v, bins=30, color=C1, alpha=0.85); ax[2].axvline(0, ls="--", lw=1, color="#898781"); ax[2].axvline(np.median(v), color=C3, lw=2)
ax[2].set(xlabel="per-student Spearman ρ on active days", ylabel="students", title="(c) attention vs occlusion (active days)")
for a in ax: a.grid(color=GRID, lw=0.6); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "v2_fig_attention_faithfulness.png"), dpi=200); plt.close(fig)
json.dump({"xgboost": xgb.__version__, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}, open(os.path.join(OUT, "versions.json"), "w"), indent=1)
print(f"V2 DONE in {time.time()-T0:.0f}s")
