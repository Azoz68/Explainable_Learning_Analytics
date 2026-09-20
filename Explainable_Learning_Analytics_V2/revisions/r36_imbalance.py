import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
import sys, os, json, math, warnings, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             f1_score, recall_score, precision_score, confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = str(DATA_DIR)
PREP = str(PREPROCESSED_CSV)
OUT = os.path.join(str(RESULTS_DIR), "R3-6"); os.makedirs(OUT, exist_ok=True)
RS = 42; CN = ['Distinction', 'Fail', 'Pass', 'Withdrawn']; D, F, P, W = 0, 1, 2, 3
np.random.seed(RS)
T0 = time.time()

df = pd.read_csv(PREP); target = 'target_result'
CAT = ["gender", "region", "highest_education", "imd_band", "age_band", "disability", "code_module", "code_presentation"]
feature_cols = [c for c in df.columns if c not in {"id_student", "final_result", target}]
X_raw, y = df[feature_cols].copy(), df[target].astype(int)
X_temp_raw, X_test_raw, y_temp, y_test = train_test_split(X_raw, y, test_size=0.20, stratify=y, random_state=RS)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_temp_raw, y_temp, test_size=0.20, stratify=y_temp, random_state=RS)
X_train_enc, X_val_enc, X_test_enc = X_train_raw.copy(), X_val_raw.copy(), X_test_raw.copy()
for col in CAT:
    le = LabelEncoder(); le.fit(X_train_enc[col].astype(str)); seen = set(le.classes_)
    for s in (X_train_enc, X_val_enc, X_test_enc):
        s[col] = le.transform(s[col].astype(str).apply(lambda v: v if v in seen else le.classes_[0]))
X_train_enc, X_val_enc, X_test_enc = [s.fillna(0).astype(float) for s in (X_train_enc, X_val_enc, X_test_enc)]
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train_enc), columns=feature_cols, index=X_train_enc.index)
X_val   = pd.DataFrame(sc.transform(X_val_enc),   columns=feature_cols, index=X_val_enc.index)
X_test  = pd.DataFrame(sc.transform(X_test_enc),  columns=feature_cols, index=X_test_enc.index)
yt, yv, ytr = y_test.values, y_val.values, y_train.values
print(f"splits {X_train.shape} {X_val.shape} {X_test.shape}")

model = joblib.load(os.path.join(BASE, "models", "xgb_final.joblib"))
assert list(model.feature_names_in_) == feature_cols
Pte = model.predict_proba(X_test); Pva = model.predict_proba(X_val)
pred = Pte.argmax(1)
acc = accuracy_score(yt, pred); assert abs(acc - 0.5762) < 0.002, acc
print(f"SANITY saved-model test accuracy = {acc:.4f}")

def multiclass_report(y, Pm, name):
    pr = Pm.argmax(1)
    macro = precision_recall_fscore_support(y, pr, average='macro', zero_division=0)
    wtd = precision_recall_fscore_support(y, pr, average='weighted', zero_division=0)
    rows = {"model": name, "accuracy": accuracy_score(y, pr), "balanced_accuracy": balanced_accuracy_score(y, pr),
            "macro_precision": macro[0], "macro_recall": macro[1], "macro_f1": macro[2],
            "weighted_precision": wtd[0], "weighted_recall": wtd[1], "weighted_f1": wtd[2],
            "macro_ovr_roc_auc": roc_auc_score(pd.get_dummies(y), Pm, average='macro', multi_class='ovr'),
            "macro_pr_auc": float(np.mean([average_precision_score((y == c).astype(int), Pm[:, c]) for c in range(4)]))}
    per = []
    pc = precision_recall_fscore_support(y, pr, average=None, zero_division=0)
    for c in range(4):
        yb = (y == c).astype(int)
        per.append({"model": name, "class": CN[c], "support": int(yb.sum()), "prevalence": float(yb.mean()),
                    "precision": pc[0][c], "recall": pc[1][c], "f1": pc[2][c],
                    "ovr_roc_auc": roc_auc_score(yb, Pm[:, c]), "ovr_pr_auc": average_precision_score(yb, Pm[:, c])})
    return rows, per

agg_rows, per_rows = [], []
r, p = multiclass_report(yt, Pte, "XGBoost (paper model)"); agg_rows.append(r); per_rows += p

import torch, torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model); pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :].to(x.dtype)

class HTBT(nn.Module):
    def __init__(self, n_static, seq_len, d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1, num_classes=4):
        super().__init__()
        self.seq_proj = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.pos_enc = PositionalEncoding(d_model, seq_len + 10)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.seq_encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.static_proj = nn.Sequential(nn.Linear(n_static, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.fusion_mlp = nn.Sequential(nn.Linear(d_model * 2, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(d_ff), nn.Linear(d_ff, d_model), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(d_model, d_ff // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff // 2, num_classes))
        self.attn_weights = None
    def forward(self, static_x, seq_x):
        seq = self.pos_enc(self.seq_proj(seq_x.unsqueeze(-1))); seq_enc = self.seq_encoder(seq); seq_pool = seq_enc.mean(1)
        s = self.static_proj(static_x)
        a, w = self.cross_attn(query=s.unsqueeze(1), key=seq_enc, value=seq_enc, need_weights=True, average_attn_weights=False)
        self.attn_weights = w
        return self.classifier(self.fusion_mlp(torch.cat([s, a.squeeze(1)], -1))), seq_pool, s

static_features = [c for c in df.columns if c not in ['id_student', 'code_module', 'code_presentation', target, 'final_result']]
assert len(static_features) == 32
y_arr = y.values.astype(np.int64)
tv_idx, te_idx = train_test_split(np.arange(len(y_arr)), test_size=0.20, stratify=y_arr, random_state=RS)
tr_idx, va_idx = train_test_split(tv_idx, test_size=0.20, stratify=y_arr[tv_idx], random_state=RS)
assert set(te_idx) == set(X_test.index), "HTBT/XGB test partitions differ"
Xs = df[static_features].fillna(0).reset_index(drop=True).copy()
for col in Xs.select_dtypes(include='object').columns:
    le = LabelEncoder(); le.fit(Xs.iloc[tr_idx][col].astype(str)); seen = set(le.classes_)
    Xs[col] = Xs[col].astype(str).apply(lambda x: int(le.transform([x])[0]) if x in seen else -1)
sch = StandardScaler(); Xs_arr = Xs.values.astype(np.float32); sch.fit(Xs_arr[tr_idx]); Xs_all = sch.transform(Xs_arr).astype(np.float32)
print("building 30-day sequences from studentVle ...")
vle = pd.read_csv(os.path.join(BASE, "studentVle.csv"), usecols=['id_student', 'code_module', 'code_presentation', 'date', 'sum_click'])
vle['date'] = pd.to_numeric(vle['date'], errors='coerce').fillna(0).astype(int)
vle = vle[(vle['date'] >= 0) & (vle['date'] < 30)]
def key(fr): return fr['id_student'].astype(str) + '_' + fr['code_module'].astype(str) + '_' + fr['code_presentation'].astype(str)
meta_key = key(df); k2i = {k: i for i, k in enumerate(meta_key)}
vle['k'] = key(vle); g = vle.groupby(['k', 'date'])['sum_click'].sum().reset_index()
seqs = np.zeros((len(df), 30), dtype=np.float32)
ki = g['k'].map(k2i); ok = ki.notna()
seqs[ki[ok].astype(int).values, g.loc[ok, 'date'].values] = g.loc[ok, 'sum_click'].values
seqs = np.log1p(seqs); del vle, g
htbt = HTBT(32, 30)
htbt.load_state_dict(torch.load(os.path.join(BASE, "models_htbt", "htbt_best.pt"), map_location='cpu', weights_only=False)); htbt.eval()
def htbt_probs(idx):
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            b = idx[s:s + 512]; lg, _, _ = htbt(torch.from_numpy(Xs_all[b]), torch.from_numpy(seqs[b])); out.append(torch.softmax(lg, 1).numpy())
    return np.vstack(out)
te_sorted = np.array(X_test.index)
Hte = htbt_probs(te_sorted); Hva = htbt_probs(np.array(X_val.index))
h_acc = accuracy_score(yt, Hte.argmax(1)); print(f"SANITY HTBT test accuracy = {h_acc:.4f} (paper 0.5653)")
r, p = multiclass_report(yt, Hte, "HTBT (paper model)"); agg_rows.append(r); per_rows += p

def brier_mc(yy, Pm): Y = np.eye(4)[yy]; return float(np.mean(np.sum((Pm - Y) ** 2, axis=1)))
def ece_top(yy, Pm, bins=15):
    conf = Pm.max(1); pr = Pm.argmax(1); hit = (pr == yy).astype(float); edges = np.linspace(0, 1, bins + 1); e = 0.0; rows = []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0: continue
        e += m.mean() * abs(hit[m].mean() - conf[m].mean()); rows.append((edges[i], edges[i + 1], int(m.sum()), conf[m].mean(), hit[m].mean()))
    return float(e), rows
cal_rows = []; xgb_bins = None
for name, Pm in [("XGBoost (paper model)", Pte), ("HTBT (paper model)", Hte)]:
    e, bins = ece_top(yt, Pm)
    cal_rows.append({"model": name, "brier_multiclass": brier_mc(yt, Pm), "ece_top_label_15bins": e,
                     "mean_confidence": float(Pm.max(1).mean()), "accuracy": accuracy_score(yt, Pm.argmax(1))})
    if name.startswith("XGBoost"): xgb_bins = bins
pd.DataFrame(cal_rows).to_csv(os.path.join(OUT, "table_calibration.csv"), index=False)

def reliability_curve(yb, pb, bins=10):
    edges = np.linspace(0, 1, bins + 1); xs = []; ys = []; ns = []
    for i in range(bins):
        m = (pb > edges[i]) & (pb <= edges[i + 1])
        if m.sum() >= 20: xs.append(pb[m].mean()); ys.append(yb[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)
C1, C2, C3, GRID = "#2a78d6", "#eb6834", "#e34948", "#e1e0d9"
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ax[0].plot([0, 1], [0, 1], ls='--', lw=1, color='#898781')
ax[0].plot([b[3] for b in xgb_bins], [b[4] for b in xgb_bins], '-o', color=C1, lw=2, ms=5)
ax[0].set(xlabel='predicted confidence (top label)', ylabel='observed accuracy', title='(a) XGBoost top-label reliability', xlim=(0, 1), ylim=(0, 1))
ax[1].plot([0, 1], [0, 1], ls='--', lw=1, color='#898781')
for c, col, lab in [(F, C2, 'Fail'), (D, C3, 'Distinction'), (W, C1, 'Withdrawn')]:
    xs, ys, ns = reliability_curve((yt == c).astype(int), Pte[:, c]); ax[1].plot(xs, ys, '-o', color=col, lw=2, ms=5, label=lab)
ax[1].set(xlabel='predicted class probability', ylabel='observed frequency', title='(b) per-class one-vs-rest reliability', xlim=(0, 1), ylim=(0, 1)); ax[1].legend(frameon=False)
for a in ax: a.grid(color=GRID, lw=0.6); a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_reliability_xgb.png"), dpi=200); plt.close(fig)

def factory(**kw):
    return xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", n_estimators=200, learning_rate=0.1, max_depth=6, random_state=RS, n_jobs=-1, verbosity=0, **kw)
sw = compute_sample_weight('balanced', ytr)
m_w = factory(); m_w.fit(X_train, ytr, sample_weight=sw); Pw = m_w.predict_proba(X_test)
r, p = multiclass_report(yt, Pw, "XGBoost + balanced class weights"); agg_rows.append(r); per_rows += p
m_c = factory(); m_c.fit(X_train, ytr); Pc = m_c.predict_proba(X_test)
print(f"control refit (unweighted) acc={accuracy_score(yt, Pc.argmax(1)):.4f}  weighted acc={accuracy_score(yt, Pw.argmax(1)):.4f}")
pd.DataFrame(agg_rows).to_csv(os.path.join(OUT, "table_multiclass_extended.csv"), index=False)
pd.DataFrame(per_rows).to_csv(os.path.join(OUT, "table_per_class.csv"), index=False)

ytr_b, yv_b, yt_b = [np.isin(a, [F, W]).astype(int) for a in (ytr, yv, yt)]
print(f"at-risk prevalence: train {ytr_b.mean():.3f} val {yv_b.mean():.3f} test {yt_b.mean():.3f}")
def bfactory(**kw):
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=200, learning_rate=0.1, max_depth=6, random_state=RS, n_jobs=-1, verbosity=0, **kw)
mb = bfactory(); mb.fit(X_train, ytr_b); sb_va = mb.predict_proba(X_val)[:, 1]; sb_te = mb.predict_proba(X_test)[:, 1]
mbw = bfactory(); mbw.fit(X_train, ytr_b, sample_weight=compute_sample_weight('balanced', ytr_b))
sbw_va = mbw.predict_proba(X_val)[:, 1]; sbw_te = mbw.predict_proba(X_test)[:, 1]
s4_va = Pva[:, F] + Pva[:, W]; s4_te = Pte[:, F] + Pte[:, W]
sh_va = Hva[:, F] + Hva[:, W]; sh_te = Hte[:, F] + Hte[:, W]
def thr_for_recall(yb, s, target=0.80):

    for t in np.sort(np.unique(s))[::-1]:
        if recall_score(yb, (s >= t).astype(int)) >= target: return float(t)
    return float(s.min())
def binary_report(name, s_va, s_te):
    t80 = thr_for_recall(yv_b, s_va, 0.80); rows = []
    for opname, t in [("default 0.5", 0.5), ("validation-chosen (recall >= 0.80)", t80)]:
        pr = (s_te >= t).astype(int)
        rows.append({"model": name, "operating_point": opname, "threshold": t,
                     "roc_auc": roc_auc_score(yt_b, s_te), "pr_auc": average_precision_score(yt_b, s_te),
                     "balanced_accuracy": balanced_accuracy_score(yt_b, pr), "accuracy": accuracy_score(yt_b, pr),
                     "at_risk_recall": recall_score(yt_b, pr), "at_risk_precision": precision_score(yt_b, pr, zero_division=0),
                     "at_risk_f1": f1_score(yt_b, pr), "flagged_share": float(pr.mean())})
    return rows
brows = []
brows += binary_report("Binary XGBoost (retrained)", sb_va, sb_te)
brows += binary_report("Binary XGBoost + balanced weights", sbw_va, sbw_te)
brows += binary_report("4-class XGBoost, P(Fail)+P(Withdrawn)", s4_va, s4_te)
brows += binary_report("4-class HTBT, P(Fail)+P(Withdrawn)", sh_va, sh_te)
pd.DataFrame(brows).to_csv(os.path.join(OUT, "table_binary_at_risk.csv"), index=False)
cm = confusion_matrix(yt, pred, labels=[0, 1, 2, 3]); err = cm.sum() - np.trace(cm)
grp = np.array([0, 1, 0, 1])
within = int(sum(cm[i, j] for i in range(4) for j in range(4) if i != j and grp[i] == grp[j])); across = int(err - within)
json.dump({"total_errors": int(err), "within_group_errors": within, "across_boundary_errors": across, "within_share": within / err},
          open(os.path.join(OUT, "error_structure.json"), 'w'), indent=1)
print(f"4-class errors: {err} total, {within} within-group ({within / err:.1%}), {across} across the at-risk boundary")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for s, col, lab in [(sb_te, C1, 'binary XGBoost'), (s4_te, C2, '4-class XGBoost, P(Fail)+P(Withdrawn)'), (sh_te, C3, '4-class HTBT, P(Fail)+P(Withdrawn)')]:
    fpr, tpr, _ = roc_curve(yt_b, s); ax[0].plot(fpr, tpr, color=col, lw=2, label=f"{lab} (AUC {roc_auc_score(yt_b, s):.3f})")
    pr_, rc_, _ = precision_recall_curve(yt_b, s); ax[1].plot(rc_, pr_, color=col, lw=2, label=f"{lab} (PR-AUC {average_precision_score(yt_b, s):.3f})")
ax[0].plot([0, 1], [0, 1], ls='--', lw=1, color='#898781'); ax[0].set(xlabel='false positive rate', ylabel='true positive rate', title='(a) ROC: at-risk vs not at-risk')
ax[1].axhline(yt_b.mean(), ls='--', lw=1, color='#898781'); ax[1].set(xlabel='recall (at-risk students caught)', ylabel='precision', title='(b) precision-recall', ylim=(0.4, 1))
ax[0].legend(frameon=False, fontsize=8, loc='lower right'); ax[1].legend(frameon=False, fontsize=8, loc='lower left')
for a in ax: a.grid(color=GRID, lw=0.6); a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_binary_roc_pr.png"), dpi=200); plt.close(fig)

rng = np.random.RandomState(RS); B = 1000; n = len(yt)
boot = {"xgb4_balanced_accuracy": [], "xgb4_macro_f1": [], "bin_roc_auc": [], "bin_pr_auc": [], "s4_roc_auc": []}
for _ in range(B):
    i = rng.randint(0, n, n)
    boot["xgb4_balanced_accuracy"].append(balanced_accuracy_score(yt[i], pred[i])); boot["xgb4_macro_f1"].append(f1_score(yt[i], pred[i], average='macro'))
    boot["bin_roc_auc"].append(roc_auc_score(yt_b[i], sb_te[i])); boot["bin_pr_auc"].append(average_precision_score(yt_b[i], sb_te[i])); boot["s4_roc_auc"].append(roc_auc_score(yt_b[i], s4_te[i]))
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
json.dump(ci, open(os.path.join(OUT, "bootstrap_ci.json"), 'w'), indent=1)

pd.DataFrame({"df_index": te_sorted, "id_student": df.loc[te_sorted, 'id_student'].values, "y_true": yt,
              **{f"xgb4_{CN[c]}": Pte[:, c] for c in range(4)}, **{f"xgbw_{CN[c]}": Pw[:, c] for c in range(4)},
              **{f"htbt_{CN[c]}": Hte[:, c] for c in range(4)}, "bin_xgb_atrisk": sb_te, "bin_xgbw_atrisk": sbw_te}
             ).to_csv(os.path.join(OUT, "test_probabilities.csv"), index=False)
print(f"done in {time.time() - T0:.0f}s -> {OUT}")
