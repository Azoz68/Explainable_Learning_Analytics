import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
import sys, os, json, math, time, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
import xgboost as xgb
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = str(DATA_DIR)
PREP = str(PREPROCESSED_CSV)
OUT = os.path.join(str(RESULTS_DIR), "R3-1"); os.makedirs(OUT, exist_ok=True)
RS = 42; SEEDS = [42, 0, 1, 2, 3]; F, W = 1, 3; CN = ['Distinction', 'Fail', 'Pass', 'Withdrawn']
T0 = time.time()
df = pd.read_csv(PREP); target = 'target_result'
CAT = ["gender", "region", "highest_education", "imd_band", "age_band", "disability", "code_module", "code_presentation"]
fc = [c for c in df.columns if c not in {"id_student", "final_result", target}]
y_all = df[target].astype(int).values; groups = df["id_student"].values; pres = df["code_presentation"].astype(str).values

def encode_scale(tr, va, te):
    X = df[fc]; Xtr, Xva, Xte = X.iloc[tr].copy(), X.iloc[va].copy(), X.iloc[te].copy()
    for c in CAT:
        le = LabelEncoder(); le.fit(Xtr[c].astype(str)); seen = set(le.classes_)
        for s in (Xtr, Xva, Xte): s[c] = le.transform(s[c].astype(str).apply(lambda v: v if v in seen else le.classes_[0]))
    Xtr, Xva, Xte = [s.fillna(0).astype(float) for s in (Xtr, Xva, Xte)]
    sc = StandardScaler(); return sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(Xte)
def factory(seed): return xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", n_estimators=200, learning_rate=0.1, max_depth=6, random_state=seed, n_jobs=-1, verbosity=0)
def mets(y, Pm):
    pr = Pm.argmax(1); m = precision_recall_fscore_support(y, pr, average='macro', zero_division=0); yb = np.isin(y, [F, W]).astype(int)
    return {"accuracy": accuracy_score(y, pr), "macro_precision": m[0], "macro_recall": m[1], "macro_f1": m[2],
            "macro_ovr_auc": roc_auc_score(pd.get_dummies(y), Pm, average='macro', multi_class='ovr'), "at_risk_auc": roc_auc_score(yb, Pm[:, F] + Pm[:, W])}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__(); pe = torch.zeros(max_len, d_model); pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); self.register_buffer('pe', pe.unsqueeze(0))
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
static_features = [c for c in df.columns if c not in ['id_student', 'code_module', 'code_presentation', target, 'final_result']]
vle = pd.read_csv(os.path.join(BASE, "studentVle.csv"), usecols=['id_student', 'code_module', 'code_presentation', 'date', 'sum_click'])
vle['date'] = pd.to_numeric(vle['date'], errors='coerce').fillna(0).astype(int); vle = vle[(vle['date'] >= 0) & (vle['date'] < 30)]
key = lambda fr: fr['id_student'].astype(str) + '_' + fr['code_module'].astype(str) + '_' + fr['code_presentation'].astype(str)
k2i = {k: i for i, k in enumerate(key(df))}; vle['k'] = key(vle); g = vle.groupby(['k', 'date'])['sum_click'].sum().reset_index()
seqs = np.zeros((len(df), 30), dtype=np.float32); ki = g['k'].map(k2i); ok = ki.notna(); seqs[ki[ok].astype(int).values, g.loc[ok, 'date'].values] = g.loc[ok, 'sum_click'].values; seqs = np.log1p(seqs); del vle, g
def train_htbt(tr_, va_, te_, seed=42, max_epochs=60, patience=8, tag="grouped"):
    torch.manual_seed(seed); np.random.seed(seed)
    Xs = df[static_features].fillna(0).reset_index(drop=True).copy()
    for col in Xs.select_dtypes(include='object').columns:
        le = LabelEncoder(); le.fit(Xs.iloc[tr_][col].astype(str)); seen = set(le.classes_); Xs[col] = Xs[col].astype(str).apply(lambda v: int(le.transform([v])[0]) if v in seen else -1)
    arr = Xs.values.astype(np.float32); sc = StandardScaler().fit(arr[tr_]); Xa = sc.transform(arr).astype(np.float32)
    mk = lambda idx, sh: DataLoader(TensorDataset(torch.from_numpy(Xa[idx]), torch.from_numpy(seqs[idx]), torch.from_numpy(y_all[idx].astype(np.int64))), batch_size=64, shuffle=sh)
    trl, val, tel = mk(tr_, True), mk(va_, False), mk(te_, False)
    model = HTBT(len(static_features), 30); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs); crit = nn.CrossEntropyLoss(); best, wait, best_state = -1e9, 0, None
    def evaluate(loader):
        model.eval(); P = []; Y = []
        with torch.no_grad():
            for xs, ss, yy in loader: lg, _, _ = model(xs, ss); P.append(torch.softmax(lg, 1).numpy()); Y.append(yy.numpy())
        return np.vstack(P), np.concatenate(Y)
    for ep in range(1, max_epochs + 1):
        model.train(); t1 = time.time()
        for xs, ss, yy in trl:
            opt.zero_grad(); lg, sp, se = model(xs, ss); loss = crit(lg, yy) + 0.3 * nn.functional.mse_loss(sp, se); loss.backward(); opt.step()
        sch.step(); Pv, Yv = evaluate(val); f1v = f1_score(Yv, Pv.argmax(1), average='macro')
        print(f"    HTBT[{tag}] epoch {ep:02d} val_f1={f1v:.4f} ({time.time()-t1:.0f}s)", flush=True)
        if f1v > best: best, wait, best_state = f1v, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: print("    early stopping"); break
    model.load_state_dict(best_state); Pt, Yt = evaluate(tel); return mets(Yt, Pt), ep, best

SEED = int(sys.argv[1]); torch.set_num_threads(max(2, (os.cpu_count() or 4) // 3))
def grouped_split(seed):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed); tv_, te_ = next(outer.split(np.zeros(len(y_all)), y_all, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed); tr_i, va_i = next(inner.split(np.zeros(len(tv_)), y_all[tv_], groups[tv_]))
    tr_, va_ = tv_[tr_i], tv_[va_i]; assert not (set(groups[tr_]) & set(groups[te_])) and not (set(groups[va_]) & set(groups[te_])) and not (set(groups[tr_]) & set(groups[va_]))
    return tr_, va_, te_
tr_, va_, te_ = grouped_split(42); assert len(te_) == 6519
m, ep, best = train_htbt(tr_, va_, te_, seed=SEED, tag=f"grouped42-trainseed{SEED}")
row = {"protocol": "student-grouped split (seed-42 partition)", "train_seed": SEED, "epochs_run": ep, "best_val_macro_f1": best, **m}
fp = os.path.join(OUT, f"table_htbt_grouped_seed{SEED}.csv"); pd.DataFrame([row]).to_csv(fp, index=False)
print("HTBT GROUPED SEED DONE", SEED, {k: round(v, 4) for k, v in m.items()}, "epochs", ep, "best_val", round(best, 4), f"in {time.time()-T0:.0f}s")
