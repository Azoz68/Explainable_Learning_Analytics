import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
import sys, os, json, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import entropy, norm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = str(DATA_DIR)
PREP = str(PREPROCESSED_CSV)
OUT = os.path.join(str(RESULTS_DIR), "R3-7"); os.makedirs(OUT, exist_ok=True)
RS = 42; CUTOFFS = [7, 14, 21, 30, 45, 60, 90, 120, 180, 270]; F, W = 1, 3
T0 = time.time()

def rd(n): return pd.read_csv(os.path.join(BASE, n))
assessments, courses, studentAssessment = rd("assessments.csv"), rd("courses.csv"), rd("studentAssessment.csv")
studentInfo, studentRegistration, studentVle, vle = rd("studentInfo.csv"), rd("studentRegistration.csv"), rd("studentVle.csv"), rd("vle.csv")
for d_ in [assessments, studentAssessment, studentInfo, studentRegistration, studentVle, vle, courses]:
    d_.replace("?", np.nan, inplace=True)
studentVle = studentVle.assign(date=pd.to_numeric(studentVle["date"], errors="coerce").fillna(0).astype(int))
studentAssessment = studentAssessment.assign(date_submitted=pd.to_numeric(studentAssessment["date_submitted"], errors="coerce").fillna(0).astype(int))
assessments = assessments.assign(date=pd.to_numeric(assessments["date"], errors="coerce").fillna(0).astype(int))
studentRegistration = studentRegistration.assign(
    date_registration=pd.to_numeric(studentRegistration["date_registration"], errors="coerce").fillna(0).astype(int),
    date_unregistration=pd.to_numeric(studentRegistration["date_unregistration"], errors="coerce").fillna(-1).astype(int))
studentInfo = studentInfo.copy(); studentInfo["imd_band"] = studentInfo["imd_band"].fillna("Unknown")
K = ["id_student", "code_module", "code_presentation"]
print(f"raw loaded in {time.time()-T0:.0f}s")

reg = pd.merge(studentRegistration, courses.copy(), on=["code_module", "code_presentation"], how="left")
reg["module_presentation_length"] = pd.to_numeric(reg["module_presentation_length"], errors="coerce")
reg["study_duration"] = (reg["module_presentation_length"] - reg["date_registration"]).clip(0).fillna(0)
reg_agg = reg.groupby(K, as_index=False).agg(study_duration=("study_duration", "mean"), first_registration=("date_registration", "min"))
_ur = rd("studentRegistration.csv"); _ur["u"] = pd.to_numeric(_ur["date_unregistration"], errors="coerce")
unreg_all = studentInfo[K].merge(_ur.groupby(K, as_index=False)["u"].max(), on=K, how="left")["u"].values

def build_master(cutoff):
    sv = studentVle[studentVle["date"] <= cutoff]
    va = pd.merge(sv, vle, on=["id_site", "code_module", "code_presentation"], how="left", validate="m:1")
    behav = va.groupby(K, as_index=False).agg(total_clicks=("sum_click", "sum"), avg_clicks_per_visit=("sum_click", "mean"),
        max_clicks_per_visit=("sum_click", "max"), std_clicks=("sum_click", "std"), active_days=("date", pd.Series.nunique),
        last_activity_day=("date", "max"), unique_vle_sites=("id_site", pd.Series.nunique))
    behav["std_clicks"] = behav["std_clicks"].fillna(0)
    behav["clicks_per_active_day"] = behav["total_clicks"] / (behav["active_days"] + 1)
    behav["revisit_ratio"] = behav["total_clicks"] / (behav["unique_vle_sites"] + 1)
    sa = studentAssessment[studentAssessment["date_submitted"] <= cutoff]
    am = pd.merge(sa, assessments, on="id_assessment", how="left", validate="m:1")
    am = am[am["assessment_type"] != "Exam"]
    perf = am.groupby(K, as_index=False).agg(avg_score=("score", "mean"), num_assessments=("id_assessment", "nunique"),
                                             first_submission=("date_submitted", "min"), last_submission=("date_submitted", "max"))
    def wavg(g):
        s = g["score"].to_numpy(dtype=float); w = g["weight"].to_numpy(dtype=float); ws = np.nansum(w)
        v = (np.nanmean(s) if len(s) > 0 else 0.0) if ws == 0 else np.nansum(s * w) / ws
        return 0.0 if np.isnan(v) else v
    def trend(g):
        g = g.dropna(subset=["score", "date"]).sort_values("date")
        if len(g) < 2: return 0.0
        x = np.arange(len(g), dtype=float); yy = g["score"].to_numpy(dtype=float)
        xm, ym = x.mean(), yy.mean(); den = ((x - xm) ** 2).sum()
        return float(((x - xm) * (yy - ym)).sum() / den) if den > 0 else 0.0
    grp = am.groupby(K)
    wdf = grp.apply(wavg, include_groups=False).reset_index(name="weighted_score")
    tdf = grp.apply(trend, include_groups=False).reset_index(name="performance_trend")
    perf = perf.merge(wdf, on=K, how="left").merge(tdf, on=K, how="left")
    perf["avg_score"] = perf["avg_score"].fillna(0); perf["weighted_score"] = perf["weighted_score"].fillna(perf["avg_score"]); perf["performance_trend"] = perf["performance_trend"].fillna(0)
    daily = sv.groupby(K + ["date"], as_index=False)["sum_click"].sum()
    def ent_gap_burst(g):
        c = g["sum_click"].to_numpy(dtype=float); p = c / (c.sum() + 1e-9); e = entropy(p + 1e-12) if len(p) > 0 else 0.0
        d = np.sort(g["date"].to_numpy(dtype=int))
        if len(d) < 2: return pd.Series({"activity_entropy": e, "avg_gap_between_logins": 0.0, "burstiness_index": 0.0})
        df_ = np.diff(d); return pd.Series({"activity_entropy": e, "avg_gap_between_logins": float(np.mean(df_)), "burstiness_index": float(np.std(df_) / (np.mean(df_) + 1e-9))})
    egb = daily.groupby(K).apply(ent_gap_burst, include_groups=False).reset_index()
    m = studentInfo.copy()
    for part in (perf, behav, reg_agg, egb): m = pd.merge(m, part, on=K, how="left")
    num = m.select_dtypes(include=[np.number]).columns.tolist(); m[num] = m[num].fillna(0)
    m["engagement_efficiency"] = (m["avg_score"] + 1) / (m["total_clicks"] + 1)
    m["cbii"] = (0.55 * (m["weighted_score"] / (m["weighted_score"].max() + 1e-9)) + 0.30 * (1 / (1 + m["activity_entropy"]))
                 + 0.15 * (m["performance_trend"] / (abs(m["performance_trend"]).max() + 1e-9)))
    m["tpi"] = m["study_duration"] / (m["num_assessments"] + 1)
    m["dropout_risk_proxy"] = 0.4 * (1 / (1 + m["study_duration"])) + 0.3 * (1 / (1 + m["clicks_per_active_day"])) + 0.3 * (1 / (1 + m["weighted_score"]))
    m["final_result"] = m["final_result"].astype(str).fillna("Unknown"); m["target_result"] = LabelEncoder().fit_transform(m["final_result"])
    return m.replace([np.inf, -np.inf], 0).fillna(0)

CAT = ["gender", "region", "highest_education", "imd_band", "age_band", "disability", "code_module", "code_presentation"]
def prepare(master):
    fc = [c for c in master.columns if c not in {"id_student", "final_result", "target_result"}]
    X_raw, y = master[fc].copy(), master["target_result"].astype(int)
    Xt, Xte, yt_, yte = train_test_split(X_raw, y, test_size=0.20, stratify=y, random_state=RS)
    Xtr, Xva, ytr, yva = train_test_split(Xt, yt_, test_size=0.20, stratify=yt_, random_state=RS)
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in CAT:
        le = LabelEncoder(); le.fit(Xtr[c].astype(str)); seen = set(le.classes_)
        for s in (Xtr, Xva, Xte): s[c] = le.transform(s[c].astype(str).apply(lambda v: v if v in seen else le.classes_[0]))
    Xtr, Xva, Xte = [s.fillna(0).astype(float) for s in (Xtr, Xva, Xte)]
    sc = StandardScaler(); Xtr = pd.DataFrame(sc.fit_transform(Xtr), columns=fc, index=Xtr.index)
    Xva = pd.DataFrame(sc.transform(Xva), columns=fc, index=Xva.index); Xte = pd.DataFrame(sc.transform(Xte), columns=fc, index=Xte.index)
    return fc, Xtr, Xva, Xte, ytr.values, yva.values, yte.values
def factory(): return xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", n_estimators=200, learning_rate=0.1, max_depth=6, random_state=RS, n_jobs=-1, verbosity=0)
def metrics(y, Pm):
    pr = Pm.argmax(1); yb = np.isin(y, [F, W]).astype(int)
    return {"accuracy": accuracy_score(y, pr), "macro_f1": f1_score(y, pr, average="macro"),
            "macro_ovr_auc": roc_auc_score(pd.get_dummies(y), Pm, average="macro", multi_class="ovr"),
            "at_risk_auc": roc_auc_score(yb, Pm[:, F] + Pm[:, W])}

rows, val_probs, val_y = [], {}, None
CK = os.path.join(OUT, 'cache'); os.makedirs(CK, exist_ok=True)
for T in CUTOFFS:
    t1 = time.time(); fp = os.path.join(CK, f'val_probs_{T}.npy'); fr = os.path.join(CK, f'row_{T}.json')
    if os.path.exists(fp) and os.path.exists(fr) and T != 30:
        val_probs[T] = np.load(fp); rows.append(json.load(open(fr))); val_y = np.load(os.path.join(CK, 'val_y.npy')); print(f'T*={T:>3}: cached'); continue
    master = build_master(T)
    if T == 30:
        ref = pd.read_csv(PREP); fc0 = [c for c in ref.columns if c not in {"id_student", "final_result", "target_result"}]
        worst = 0.0
        for c in fc0:
            if not pd.api.types.is_numeric_dtype(ref[c]) or not pd.api.types.is_numeric_dtype(master[c]): assert (ref[c].astype(str).values == master[c].astype(str).values).all(), c
            else: worst = max(worst, float(np.nanmax(np.abs(ref[c].values.astype(float) - master[c].values.astype(float)))))
        print(f"  cutoff 30 re-implementation vs saved CSV: max |diff| over numeric features = {worst:.2e}")
        assert worst < 1e-6, "feature re-implementation does not reproduce the notebook"
    fc, Xtr, Xva, Xte, ytr, yva, yte = prepare(master)
    m = factory().fit(Xtr, ytr)
    Pv = m.predict_proba(Xva); val_probs[T] = Pv; val_y = yva
    r = {"cutoff_day": T, "partition": "validation", "n": int(len(yva)), **{k: float(v) for k, v in metrics(yva, Pv).items()}}
    np.save(fp, Pv); np.save(os.path.join(CK, 'val_y.npy'), yva); json.dump(r, open(fr, 'w'))
    if T == 30:
        Pt = m.predict_proba(Xte); tm = metrics(yte, Pt)
        json.dump({"cutoff_day": 30, "partition": "test (single evaluation)", "n": int(len(yte)), **tm}, open(os.path.join(OUT, "test_day30_single_evaluation.json"), "w"), indent=1)
        print(f"  single test evaluation at T*=30: acc={tm['accuracy']:.4f} macroF1={tm['macro_f1']:.4f} AUC={tm['macro_ovr_auc']:.4f} (paper 0.5762/0.4809/0.7950)")
    rows.append(r); print(f"T*={T:>3}: val acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f} auc={r['macro_ovr_auc']:.4f} atrisk-auc={r['at_risk_auc']:.4f}  ({time.time()-t1:.0f}s)")

_, _, Xva_, _, _, _, _ = prepare(build_master(30)); va_idx = Xva_.index.values; u = unreg_all[va_idx]
exp_rows = []
for T in CUTOFFS:
    left = (~np.isnan(u)) & (u <= T); still = ~left; Pv = val_probs[T]; yv_ = val_y
    r = {"cutoff_day": T, "withdrawn_share_already_unregistered_by_T": float(left[yv_ == W].mean()), "share_all_enrolments_already_unregistered": float(left.mean()),
         "n_still_registered": int(still.sum()), "at_risk_share_still_registered": float(np.isin(yv_[still], [F, W]).mean())}
    r.update({f"{k}_still_registered": v for k, v in metrics(yv_[still], Pv[still]).items()}); exp_rows.append(r)
pd.DataFrame(exp_rows).to_csv(os.path.join(OUT, "table_withdrawal_exposure_validation.csv"), index=False)
print("exposure:", [(e["cutoff_day"], round(e["withdrawn_share_already_unregistered_by_T"], 3), round(e["at_risk_auc_still_registered"], 3)) for e in exp_rows])

rng = np.random.RandomState(RS); B = 1000; n = len(val_y)
strata = [np.where(val_y == c)[0] for c in range(4)]
def resample(): return np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
boot = {T: {"accuracy": [], "macro_f1": [], "macro_ovr_auc": [], "at_risk_auc": []} for T in CUTOFFS}
pair = {(a, b): [] for a, b in zip(CUTOFFS[:-1], CUTOFFS[1:])}
for _ in range(B):
    i = resample(); mets = {}
    for T in CUTOFFS:
        mm = metrics(val_y[i], val_probs[T][i]); mets[T] = mm["macro_ovr_auc"]
        for k in boot[T]: boot[T][k].append(mm[k])
    for a, b in pair: pair[(a, b)].append(mets[b] - mets[a])
for r in rows:
    T = r["cutoff_day"]
    for k in ["accuracy", "macro_f1", "macro_ovr_auc", "at_risk_auc"]:
        r[f"{k}_ci_low"], r[f"{k}_ci_high"] = np.percentile(boot[T][k], 2.5), np.percentile(boot[T][k], 97.5)
tab = pd.DataFrame(rows); tab.to_csv(os.path.join(OUT, "table_cutoff_validation.csv"), index=False)

def midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T_ = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T_[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T_; return out
def delong_var(pos_scores, neg_scores):
    m_, n_ = len(pos_scores), len(neg_scores); allx = np.concatenate([pos_scores, neg_scores])
    tx, ty, tz = midrank(pos_scores), midrank(neg_scores), midrank(allx)
    auc = (tz[:m_].sum() - m_ * (m_ + 1) / 2) / (m_ * n_)
    v10 = (tz[:m_] - tx) / n_; v01 = 1.0 - (tz[m_:] - ty) / m_
    return auc, v10, v01
def delong_test(yb, s1, s2):
    pos, neg = yb == 1, yb == 0
    a1, v10a, v01a = delong_var(s1[pos], s1[neg]); a2, v10b, v01b = delong_var(s2[pos], s2[neg])
    S10 = np.cov(np.vstack([v10a, v10b])); S01 = np.cov(np.vstack([v01a, v01b]))
    S = S10 / len(v10a) + S01 / len(v01a); var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a2 - a1) / np.sqrt(var) if var > 0 else 0.0
    return a1, a2, z, 2 * norm.sf(abs(z))
CN = ['Distinction', 'Fail', 'Pass', 'Withdrawn']; drows = []
for (a, b), diffs in pair.items():
    lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    row = {"from_cutoff": a, "to_cutoff": b, "delta_macro_auc": float(metrics(val_y, val_probs[b])["macro_ovr_auc"] - metrics(val_y, val_probs[a])["macro_ovr_auc"]), "delta_bootstrap_mean": float(np.mean(diffs)), "delta_ci_low": lo, "delta_ci_high": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    ps = []
    for c in range(4):
        yb = (val_y == c).astype(int); _, _, z, p = delong_test(yb, val_probs[a][:, c], val_probs[b][:, c]); ps.append(p); row[f"delong_p_{CN[c]}"] = p
    order = np.argsort(ps); adj = np.empty(4)
    for rank, idx in enumerate(order): adj[idx] = min(1.0, ps[idx] * (4 - rank))
    adj = np.maximum.accumulate(adj[order])[np.argsort(order)]
    for c in range(4): row[f"delong_p_holm_{CN[c]}"] = adj[c]
    drows.append(row)
allp = [(i, c, drows[i][f"delong_p_{CN[c]}"]) for i in range(len(drows)) for c in range(4)]; ps36 = np.array([x[2] for x in allp]); o = np.argsort(ps36); adj = np.empty(len(ps36))
for rank, j in enumerate(o): adj[j] = min(1.0, ps36[j] * (len(ps36) - rank))
adj = np.maximum.accumulate(adj[o])[np.argsort(o)]
for (i, c, _), a_ in zip(allp, adj): drows[i][f"delong_p_holm36_{CN[c]}"] = float(a_)
pd.DataFrame(drows).to_csv(os.path.join(OUT, "table_adjacent_cutoff_tests.csv"), index=False)

C1, C2, C3, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#e1e0d9"
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
x = tab["cutoff_day"].values
for a_, k, col, lab in [(ax[0], "accuracy", C1, "validation accuracy"), (ax[1], "macro_f1", C2, "validation macro-F1"), (ax[2], "macro_ovr_auc", C3, "validation macro OvR AUC")]:
    a_.fill_between(x, tab[f"{k}_ci_low"], tab[f"{k}_ci_high"], color=col, alpha=0.18, lw=0)
    a_.plot(x, tab[k], "-o", color=col, lw=2, ms=4); a_.axvline(30, ls="--", lw=1, color="#898781"); a_.set_xscale("log")
    a_.set_xticks(x); a_.set_xticklabels([str(v) for v in x], fontsize=8); a_.set(xlabel="prediction cut-off day (log scale)", title=lab)
    a_.grid(color=GRID, lw=0.6); a_.spines[["top", "right"]].set_visible(False)
ax[0].text(31, tab["accuracy"].max(), "operational T* = 30", fontsize=8, color="#52514e", rotation=90, va="top")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cutoff_validation_curves.png"), dpi=200); plt.close(fig)
print(f"done in {time.time()-T0:.0f}s -> {OUT}")
