import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
import sys, os, json, math, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = str(DATA_DIR)
PREP = str(PREPROCESSED_CSV)
OUT = os.path.join(str(RESULTS_DIR), "R3-5"); os.makedirs(OUT, exist_ok=True)
RS = 42; SEQ = 30; CN = ['Distinction', 'Fail', 'Pass', 'Withdrawn']
np.random.seed(RS); torch.manual_seed(RS); T0 = time.time()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__(); pe = torch.zeros(max_len, d_model); pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :].to(x.dtype)
class HTBT(nn.Module):
    def __init__(self, n_static, seq_len, d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1, num_classes=4):
        super().__init__(); self.seq_proj = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.LayerNorm(d_model)); self.pos_enc = PositionalEncoding(d_model, seq_len + 10)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True); self.seq_encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.static_proj = nn.Sequential(nn.Linear(n_static, d_model), nn.ReLU(), nn.LayerNorm(d_model)); self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.fusion_mlp = nn.Sequential(nn.Linear(d_model * 2, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(d_ff), nn.Linear(d_ff, d_model), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(d_model, d_ff // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff // 2, num_classes)); self.attn_weights = None
    def forward(self, static_x, seq_x):
        seq = self.pos_enc(self.seq_proj(seq_x.unsqueeze(-1))); seq_enc = self.seq_encoder(seq); seq_pool = seq_enc.mean(1); s = self.static_proj(static_x)
        a, w = self.cross_attn(query=s.unsqueeze(1), key=seq_enc, value=seq_enc, need_weights=True, average_attn_weights=False); self.attn_weights = w
        return self.classifier(self.fusion_mlp(torch.cat([s, a.squeeze(1)], -1))), seq_pool, s

df = pd.read_csv(PREP); target = 'target_result'
static_features = [c for c in df.columns if c not in ['id_student', 'code_module', 'code_presentation', target, 'final_result']]; assert len(static_features) == 32
y = df[target].astype(int).values
tv, te = train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=RS)
tr, va = train_test_split(tv, test_size=0.20, stratify=y[tv], random_state=RS)
Xs = df[static_features].fillna(0).reset_index(drop=True).copy()
for col in Xs.select_dtypes(include='object').columns:
    le = LabelEncoder(); le.fit(Xs.iloc[tr][col].astype(str)); seen = set(le.classes_)
    Xs[col] = Xs[col].astype(str).apply(lambda v: int(le.transform([v])[0]) if v in seen else -1)
sc = StandardScaler(); arr = Xs.values.astype(np.float32); sc.fit(arr[tr]); Xall = sc.transform(arr).astype(np.float32)
vle = pd.read_csv(os.path.join(BASE, "studentVle.csv"), usecols=['id_student', 'code_module', 'code_presentation', 'date', 'sum_click'])
vle['date'] = pd.to_numeric(vle['date'], errors='coerce').fillna(0).astype(int); vle = vle[(vle['date'] >= 0) & (vle['date'] < SEQ)]
key = lambda fr: fr['id_student'].astype(str) + '_' + fr['code_module'].astype(str) + '_' + fr['code_presentation'].astype(str)
k2i = {k: i for i, k in enumerate(key(df))}; vle['k'] = key(vle); g = vle.groupby(['k', 'date'])['sum_click'].sum().reset_index()
seqs = np.zeros((len(df), SEQ), dtype=np.float32); ki = g['k'].map(k2i); ok = ki.notna()
seqs[ki[ok].astype(int).values, g.loc[ok, 'date'].values] = g.loc[ok, 'sum_click'].values; seqs = np.log1p(seqs); del vle, g
model = HTBT(32, SEQ); model.load_state_dict(torch.load(os.path.join(BASE, "models_htbt", "htbt_best.pt"), map_location='cpu', weights_only=False)); model.eval()

te = np.sort(te); Xte, Ste, yte = Xall[te], seqs[te], y[te]; N = len(te)
def fwd(Xb, Sb, want_attn=False):
    out, att = [], []
    with torch.no_grad():
        for s in range(0, len(Xb), 512):
            lg, _, _ = model(torch.from_numpy(Xb[s:s + 512]), torch.from_numpy(Sb[s:s + 512])); out.append(torch.softmax(lg, 1).numpy())
            if want_attn: att.append(model.attn_weights.mean(1).squeeze(1).numpy())
    return (np.vstack(out), np.vstack(att)) if want_attn else np.vstack(out)
P0, A = fwd(Xte, Ste, True); pred = P0.argmax(1); p0 = P0[np.arange(N), pred]
acc = (pred == yte).mean(); print(f"SANITY HTBT test accuracy = {acc:.4f} (paper 0.5653); attention rows sum to {A.sum(1).mean():.4f}")

def occlude(S, days_idx):
    S2 = S.copy(); S2[np.arange(len(S))[:, None], days_idx] = 0.0; return S2
order = np.argsort(A, axis=1)
top3, bot3 = order[:, -3:], order[:, :3]
d_top = p0 - fwd(Xte, occlude(Ste, top3))[np.arange(N), pred]
d_bot = p0 - fwd(Xte, occlude(Ste, bot3))[np.arange(N), pred]
rng = np.random.RandomState(RS); d_rand = np.zeros(N)
for _ in range(5):
    rnd = np.stack([rng.choice(SEQ, 3, replace=False) for _ in range(N)]); d_rand += (p0 - fwd(Xte, occlude(Ste, rnd))[np.arange(N), pred]) / 5

active_top = (np.take_along_axis(Ste, top3, 1) > 0).all(1)
def rb(x, y_): d = x - y_; pos = (d > 0).sum(); neg = (d < 0).sum(); return (pos - neg) / max(1, pos + neg)
def boot_ci(v, B=2000):
    r = np.random.RandomState(RS); m = [r.choice(v, len(v)).mean() for _ in range(B)]; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
occ = []
for name, dv in [("top-3 attended days", d_top), ("bottom-3 attended days", d_bot), ("random 3 days (mean of 5 draws)", d_rand)]:
    lo, hi = boot_ci(dv); occ.append({"condition": name, "n": N, "mean_prob_drop": float(dv.mean()), "ci_low": lo, "ci_high": hi, "median": float(np.median(dv)), "share_positive": float((dv > 0).mean())})
w_tr = wilcoxon(d_top, d_rand, zero_method='pratt'); w_tb = wilcoxon(d_top, d_bot, zero_method='pratt')
tests = {"top_vs_random": {"wilcoxon_p": float(w_tr.pvalue), "rank_biserial": float(rb(d_top, d_rand)), "mean_diff": float((d_top - d_rand).mean())},
         "top_vs_bottom": {"wilcoxon_p": float(w_tb.pvalue), "rank_biserial": float(rb(d_top, d_bot)), "mean_diff": float((d_top - d_bot).mean())},
         "n_with_all_top3_days_active": int(active_top.sum()),
         "top_vs_random_active_only": {"mean_diff": float((d_top - d_rand)[active_top].mean()), "wilcoxon_p": float(wilcoxon(d_top[active_top], d_rand[active_top], zero_method='pratt').pvalue)}}
pd.DataFrame(occ).to_csv(os.path.join(OUT, "table_occlusion.csv"), index=False); json.dump(tests, open(os.path.join(OUT, "occlusion_tests.json"), "w"), indent=1)
print("occlusion:", [(o["condition"], round(o["mean_prob_drop"], 4)) for o in occ], "| top vs random p =", f"{w_tr.pvalue:.2e}")

imp = np.zeros((N, SEQ))
for d in range(SEQ):
    imp[:, d] = np.abs(p0 - fwd(Xte, occlude(Ste, np.full((N, 1), d)))[np.arange(N), pred])
rho = np.array([spearmanr(A[i], imp[i]).correlation if imp[i].std() > 0 else np.nan for i in range(N)])
valid = ~np.isnan(rho); rho_v = rho[valid]
w_rho = wilcoxon(rho_v, zero_method='pratt')
rho_stats = {"n_valid": int(valid.sum()), "n_constant_impact": int((~valid).sum()), "mean_rho": float(rho_v.mean()), "median_rho": float(np.median(rho_v)),
             "iqr": [float(np.percentile(rho_v, 25)), float(np.percentile(rho_v, 75))], "share_positive": float((rho_v > 0).mean()), "wilcoxon_p_vs_zero": float(w_rho.pvalue)}

rho_pop = spearmanr(A.mean(0), imp.mean(0)).correlation; rho_stats["population_level_rho_meanA_vs_meanImpact"] = float(rho_pop)

gxi = np.zeros((N, SEQ)); model.eval()
for s in range(0, N, 512):
    xb = torch.from_numpy(Xte[s:s + 512]); sb = torch.from_numpy(Ste[s:s + 512]).requires_grad_(True)
    lg, _, _ = model(xb, sb); sel = lg[torch.arange(len(sb)), torch.from_numpy(pred[s:s + 512])]
    grads = torch.autograd.grad(sel.sum(), sb)[0]; gxi[s:s + 512] = (grads * sb).detach().numpy()
rho_g_att = np.array([spearmanr(np.abs(gxi[i]), A[i]).correlation if np.abs(gxi[i]).std() > 0 else np.nan for i in range(N)])
rho_g_imp = np.array([spearmanr(np.abs(gxi[i]), imp[i]).correlation if (np.abs(gxi[i]).std() > 0 and imp[i].std() > 0) else np.nan for i in range(N)])
rho_stats["gradxinput_vs_attention_median_rho"] = float(np.nanmedian(rho_g_att)); rho_stats["gradxinput_vs_occlusion_median_rho"] = float(np.nanmedian(rho_g_imp))
json.dump(rho_stats, open(os.path.join(OUT, "rho_stats.json"), "w"), indent=1)
print("rho:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rho_stats.items()})

mA = A.mean(0); top = np.argsort(mA)[::-1][:3]; low = np.argsort(mA)[:3]
km = KMeans(n_clusters=3, random_state=RS, n_init=10).fit(StandardScaler().fit_transform(A)); sizes = np.bincount(km.labels_)
clus = []
for c in range(3):
    idx = km.labels_ == c; a_c = A[idx].mean(0); s_c = Ste[idx]
    clus.append({"cluster": c, "n": int(idx.sum()), "peak_attention_day": int(a_c.argmax()), "mean_active_days": float((s_c > 0).sum(1).mean()),
                 "dominant_true_outcome": CN[int(np.bincount(yte[idx], minlength=4).argmax())], "share_at_risk": float(np.isin(yte[idx], [1, 3]).mean()),
                 "mean_rho": float(np.nanmean(rho[idx]))})
t7 = {"most_attended_positions": top.tolist(), "highest_attention_weights": [float(mA[i]) for i in top], "lowest_attended_positions": low.tolist(),
      "lowest_attention_weights": [float(mA[i]) for i in low], "uniform_reference": 1 / SEQ, "cluster_sizes": sizes.tolist(), "clusters": clus}
json.dump(t7, open(os.path.join(OUT, "table7_attention_definitive.json"), "w"), indent=1)
pd.DataFrame({"day": np.arange(SEQ), "mean_attention": mA, "mean_occlusion_impact": imp.mean(0), "mean_abs_gradxinput": np.abs(gxi).mean(0), "share_active": (Ste > 0).mean(0)}).to_csv(os.path.join(OUT, "table_per_day.csv"), index=False)
print("Table 7:", t7["most_attended_positions"], np.round(t7["highest_attention_weights"], 5), "clusters", sizes)

C1, C2, C3, GRID = "#2a78d6", "#eb6834", "#e34948", "#e1e0d9"
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
ax[0].plot(np.arange(SEQ), mA, "-o", color=C1, ms=3, lw=2, label="mean cross-attention"); ax[0].axhline(1 / SEQ, ls="--", lw=1, color="#898781"); ax[0].set(xlabel="day in window", ylabel="attention", title="(a) attention per day (test set)")
ax0b = ax[0]
ax[1].plot(np.arange(SEQ), imp.mean(0), "-o", color=C2, ms=3, lw=2); ax[1].set(xlabel="day in window", ylabel="mean |Δp| when day occluded", title="(b) single-day occlusion impact")
ax[2].hist(rho_v, bins=30, color=C1, alpha=0.85); ax[2].axvline(0, ls="--", lw=1, color="#898781"); ax[2].axvline(np.median(rho_v), color=C3, lw=2)
ax[2].set(xlabel="per-instance Spearman ρ (attention vs occlusion impact)", ylabel="students", title=f"(c) ρ distribution, median {np.median(rho_v):.2f}")
for a in ax: a.grid(color=GRID, lw=0.6); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_attention_faithfulness.png"), dpi=200); plt.close(fig)
np.save(os.path.join(OUT, "attention_test.npy"), A); np.save(os.path.join(OUT, "occlusion_impact_test.npy"), imp)
print(f"done in {time.time()-T0:.0f}s -> {OUT}")
