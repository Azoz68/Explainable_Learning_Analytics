import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
import sys, os, warnings, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = os.path.join(str(DATA_DIR), "")
OUT_PATH = os.path.join(str(RESULTS_DIR), "oulad_results", "")
os.makedirs(OUT_PATH, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

print("=" * 60)
print("HTBT - Hybrid Temporal-Behavioural Transformer")
print("=" * 60)

print("\n[1] Rebuilding feature pipeline...")

student_info  = pd.read_csv(DATA_PATH + "studentInfo.csv").replace('?', np.nan)
student_vle   = pd.read_csv(DATA_PATH + "studentVle.csv").replace('?', np.nan)
assessments   = pd.read_csv(DATA_PATH + "assessments.csv").replace('?', np.nan)
student_assess= pd.read_csv(DATA_PATH + "studentAssessment.csv").replace('?', np.nan)
courses       = pd.read_csv(DATA_PATH + "courses.csv").replace('?', np.nan)
student_reg   = pd.read_csv(DATA_PATH + "studentRegistration.csv").replace('?', np.nan)

sv = student_vle.copy()
sv['sum_click'] = pd.to_numeric(sv['sum_click'], errors='coerce').fillna(0)
sv['date']      = pd.to_numeric(sv['date'], errors='coerce')

vle_agg = (sv.groupby(['code_module','code_presentation','id_student'])
             .agg(total_clicks=('sum_click','sum'), avg_clicks=('sum_click','mean'),
                  max_clicks=('sum_click','max'), click_std=('sum_click','std'),
                  distinct_resources=('id_site','nunique'), days_active=('date','nunique'),
                  first_access=('date','min'), last_activity_day=('date','max'))
             .reset_index())
vle_agg['click_std'].fillna(0, inplace=True)
vle_agg['cv_clicks'] = vle_agg['click_std'] / (vle_agg['avg_clicks'] + 1e-9)

def compute_entropy(group):
    clicks = group['sum_click'].values
    if clicks.sum() == 0: return 0.0
    return float(scipy_entropy(clicks / clicks.sum()))

def compute_burstiness(group):
    dates = np.sort(group['date'].dropna().unique().astype(float))
    if len(dates) < 2: return 0.0
    gaps = np.diff(dates)
    mu = gaps.mean()
    return float(gaps.std() / mu) if mu > 0 else 0.0

print("  Computing entropy & burstiness...")
entropy_df = (sv.groupby(['code_module','code_presentation','id_student'])
                .apply(compute_entropy).reset_index().rename(columns={0:'activity_entropy'}))
burst_df   = (sv.groupby(['code_module','code_presentation','id_student'])
                .apply(compute_burstiness).reset_index().rename(columns={0:'burstiness'}))
vle_agg = (vle_agg
           .merge(entropy_df, on=['code_module','code_presentation','id_student'], how='left')
           .merge(burst_df,   on=['code_module','code_presentation','id_student'], how='left'))
vle_agg[['activity_entropy','burstiness']] = vle_agg[['activity_entropy','burstiness']].fillna(0)

af = student_assess.merge(assessments, on='id_assessment', how='left')
af['score']  = pd.to_numeric(af['score'],  errors='coerce')
af['weight'] = pd.to_numeric(af['weight'], errors='coerce')
af['weighted_contrib'] = af['score'] * af['weight'] / 100.0

assess_agg = (af.groupby(['id_student','code_module','code_presentation'])
                .agg(weighted_score=('weighted_contrib','sum'),
                     num_assessments=('id_assessment','count'),
                     mean_score=('score','mean'),
                     last_submission=('date_submitted','max'))
                .reset_index())

def perf_trend(group):
    g = group.sort_values('date_submitted')
    scores = pd.to_numeric(g['score'], errors='coerce').dropna().values
    if len(scores) < 2: return 0.0
    x = np.arange(len(scores), dtype=float)
    try: return float(np.polyfit(x, scores, 1)[0])
    except: return 0.0

trend_df = (af.groupby(['id_student','code_module','code_presentation'])
              .apply(perf_trend).reset_index().rename(columns={0:'perf_trend'}))
assess_agg = assess_agg.merge(trend_df, on=['id_student','code_module','code_presentation'], how='left')
for c in ['weighted_score','mean_score','last_submission','perf_trend']:
    assess_agg[c].fillna(0, inplace=True)

reg = student_reg.copy()
reg['date_registration']   = pd.to_numeric(reg['date_registration'],   errors='coerce')
reg['date_unregistration'] = pd.to_numeric(reg['date_unregistration'], errors='coerce')
reg = reg.merge(courses, on=['code_module','code_presentation'], how='left')
reg['module_presentation_length'] = pd.to_numeric(reg['module_presentation_length'], errors='coerce')
reg['study_duration'] = np.where(reg['date_unregistration'].notna(),
    reg['date_unregistration'] - reg['date_registration'],
    reg['module_presentation_length'] - reg['date_registration']).clip(0)
reg['study_duration'].fillna(0, inplace=True)
reg['first_registration'] = reg['date_registration'].fillna(0)
reg_feat = reg[['code_module','code_presentation','id_student','study_duration','first_registration']]

df = (student_info
      .merge(vle_agg,    on=['code_module','code_presentation','id_student'], how='left')
      .merge(assess_agg, on=['code_module','code_presentation','id_student'], how='left')
      .merge(reg_feat,   on=['code_module','code_presentation','id_student'], how='left'))

df['engagement_efficiency'] = df['weighted_score'] / (df['total_clicks'] + 1)
df['CBII'] = 0.5*df['weighted_score'] + 0.3*df['activity_entropy'] + 0.2*df['perf_trend']
df['TPI']  = df['study_duration'] / (df['num_assessments'].fillna(0) + 1)
df['dropout_risk'] = (1/(df['study_duration']+1) + 1/(df['total_clicks']+1) +
                      1/(df['weighted_score']+1))

print("\n[2] Building 30-day VLE sequences...")
WINDOW = 30

sv2 = student_vle.copy()
sv2['date']      = pd.to_numeric(sv2['date'],      errors='coerce')
sv2['sum_click'] = pd.to_numeric(sv2['sum_click'], errors='coerce').fillna(0)

end_date = (sv2.groupby(['code_module','code_presentation','id_student'])
               ['date'].max().reset_index().rename(columns={'date':'end_date'}))
sv2 = sv2.merge(end_date, on=['code_module','code_presentation','id_student'])
sv2 = sv2.dropna(subset=['date','end_date'])
sv2['days_from_end'] = sv2['end_date'] - sv2['date']
sv_win = sv2[sv2['days_from_end'] < WINDOW]

sv_piv = (sv_win.groupby(['code_module','code_presentation','id_student','days_from_end'])
                ['sum_click'].sum().reset_index())

def build_seq(group):
    seq = np.zeros(WINDOW, dtype=np.float32)
    for _, row in group.iterrows():
        idx = int(row['days_from_end'])
        if 0 <= idx < WINDOW:
            seq[WINDOW - 1 - idx] = row['sum_click']
    return seq

print("  Aggregating (may take a moment)...")
seq_df = (sv_piv.groupby(['code_module','code_presentation','id_student'])
                .apply(build_seq).reset_index().rename(columns={0:'sequence'}))

df = df.merge(seq_df, on=['code_module','code_presentation','id_student'], how='left')
df['sequence'] = df['sequence'].apply(
    lambda x: x if isinstance(x, np.ndarray) else np.zeros(WINDOW, dtype=np.float32))

print("  Encoding features...")

cat_cols = ['gender','region','highest_education','imd_band','age_band',
            'disability','code_module','code_presentation']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

target_le = LabelEncoder()
df['label'] = target_le.fit_transform(df['final_result'].astype(str))
class_names = list(target_le.classes_)

exclude = ['id_student','final_result','label','sequence']
feature_cols = [c for c in df.columns if c not in exclude]
df[feature_cols] = df[feature_cols].fillna(0)

scaler = StandardScaler()
static_X = scaler.fit_transform(df[feature_cols].values)
labels   = df['label'].values
n_classes = len(class_names)
print(f"  Static features: {static_X.shape}, Classes: {class_names}")

df_htbt = df

seqs = np.stack(df_htbt['sequence'].values)
max_c = seqs.max()
if max_c > 0:
    seqs = seqs / max_c

print(f"  Sequences shape: {seqs.shape}")

idx_all = np.arange(len(labels))
idx_temp, idx_te = train_test_split(idx_all, test_size=0.20, random_state=42, stratify=labels)
idx_tr,   idx_vl = train_test_split(idx_temp, test_size=0.20, random_state=42, stratify=labels[idx_temp])

def to_tensors(idx):
    return (torch.tensor(seqs[idx],      dtype=torch.float32).unsqueeze(-1),
            torch.tensor(static_X[idx],  dtype=torch.float32),
            torch.tensor(labels[idx],    dtype=torch.long))

seq_tr, st_tr, lb_tr = to_tensors(idx_tr)
seq_vl, st_vl, lb_vl = to_tensors(idx_vl)
seq_te, st_te, lb_te = to_tensors(idx_te)

train_dl = DataLoader(TensorDataset(seq_tr, st_tr, lb_tr), batch_size=256, shuffle=True)
val_dl   = DataLoader(TensorDataset(seq_vl, st_vl, lb_vl), batch_size=256)
test_dl  = DataLoader(TensorDataset(seq_te, st_te, lb_te), batch_size=256)

class HTBTModel(nn.Module):
    def __init__(self, seq_len, n_static, n_classes,
                 d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.temporal_proj = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.static_mlp = nn.Sequential(
            nn.Linear(n_static, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, d_model),  nn.ReLU())
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model*2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes))

    def forward(self, seq, static):
        t = self.temporal_proj(seq)
        t_enc = self.transformer(t)
        s = self.static_mlp(static)
        s_q = s.unsqueeze(1)
        attn_out, attn_w = self.cross_attn(s_q, t_enc, t_enc)
        attn_out = attn_out.squeeze(1)
        t_pool = t_enc.mean(dim=1)
        align_loss = nn.functional.mse_loss(t_pool, s)
        combined = torch.cat([attn_out, s], dim=1)
        return self.classifier(combined), attn_w, align_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[3] Training HTBT on {device}...")

n_static = static_X.shape[1]
model = HTBTModel(WINDOW, n_static, n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

MAX_EPOCHS, PATIENCE, ALIGN_W = 30, 7, 0.1
best_val_acc = 0; patience_cnt = 0
train_losses, val_accs = [], []

for epoch in range(1, MAX_EPOCHS+1):
    model.train()
    ep_loss = 0
    for sb, stb, lb in train_dl:
        sb, stb, lb = sb.to(device), stb.to(device), lb.to(device)
        optimizer.zero_grad()
        logits, _, al = model(sb, stb)
        loss = criterion(logits, lb) + ALIGN_W * al
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for sb, stb, lb in val_dl:
            logits, _, _ = model(sb.to(device), stb.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(lb.numpy())
    val_acc = accuracy_score(trues, preds)
    train_losses.append(ep_loss / len(train_dl))
    val_accs.append(val_acc)
    scheduler.step(1 - val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUT_PATH + "htbt_best.pt")
        patience_cnt = 0
    else:
        patience_cnt += 1

    if epoch % 5 == 0:
        print(f"  Epoch {epoch:3d}: loss={ep_loss/len(train_dl):.4f}, val_acc={val_acc:.4f}")
    if patience_cnt >= PATIENCE:
        print(f"  Early stop at epoch {epoch}")
        break

model.load_state_dict(torch.load(OUT_PATH + "htbt_best.pt"))
model.eval()
test_preds, test_true, all_probs, all_attn = [], [], [], []

with torch.no_grad():
    for sb, stb, lb in test_dl:
        logits, attn_w, _ = model(sb.to(device), stb.to(device))
        probs = torch.softmax(logits, 1).cpu().numpy()
        test_preds.extend(logits.argmax(1).cpu().numpy())
        test_true.extend(lb.numpy())
        all_probs.extend(probs.tolist())
        all_attn.append(attn_w.squeeze(1).cpu().numpy())

htbt_acc  = accuracy_score(test_true, test_preds)
htbt_f1   = f1_score(test_true, test_preds, average='macro', zero_division=0)
htbt_prec = precision_score(test_true, test_preds, average='macro', zero_division=0)
htbt_rec  = recall_score(test_true, test_preds, average='macro', zero_division=0)
htbt_auc  = roc_auc_score(test_true, np.array(all_probs), multi_class='ovr', average='macro')

print(f"\n  Table 6: HTBT Test Set Performance")
print(f"  Accuracy : {htbt_acc:.4f}")
print(f"  Precision: {htbt_prec:.4f}")
print(f"  Recall   : {htbt_rec:.4f}")
print(f"  Macro-F1 : {htbt_f1:.4f}")
print(f"  ROC-AUC  : {htbt_auc:.4f}")

pd.DataFrame([{'accuracy':htbt_acc,'macro_precision':htbt_prec,
               'macro_recall':htbt_rec,'macro_f1':htbt_f1,'roc_auc':htbt_auc}]
             ).to_csv(OUT_PATH + "table6_htbt_test.csv", index=False)

fig, ax1 = plt.subplots(figsize=(9,5))
ax1.plot(train_losses, color='steelblue', label='Train Loss')
ax1.set_ylabel("Loss", color='steelblue')
ax2 = ax1.twinx()
ax2.plot(val_accs, color='darkorange', label='Val Accuracy')
ax2.set_ylabel("Val Accuracy", color='darkorange')
ax1.set_xlabel("Epoch")
plt.title("Figure 13: HTBT Training Curve", fontsize=13)
lines1,l1 = ax1.get_legend_handles_labels()
lines2,l2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, l1+l2, loc='lower right')
plt.tight_layout()
fig.savefig(OUT_PATH + "figure13_htbt_training.png", dpi=150)
plt.close(fig)
print("  Saved: figure13_htbt_training.png")

print("\n[4] Analysing attention patterns...")
attn_arr = np.concatenate(all_attn, axis=0)

km = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = km.fit_predict(attn_arr)
centers = km.cluster_centers_

persona = ['Early Engagers', 'Late Bloomers', 'Sporadic']
days = np.arange(WINDOW)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ci in range(3):
    mask = cluster_labels == ci
    c_attn = attn_arr[mask]
    m, s = centers[ci], c_attn.std(axis=0)
    axes[ci].plot(days, m, color='navy', label='Mean')
    axes[ci].fill_between(days, m-s, m+s, alpha=0.2, color='navy')
    axes[ci].set_title(f"Cluster {ci+1}: {persona[ci]}\n(n={mask.sum()})", fontsize=10)
    axes[ci].set_xlabel("Day in 30-day window")
    if ci == 0: axes[ci].set_ylabel("Attention weight")
plt.suptitle("Figure 14: Clustered Attention Patterns with Variance (HTBT)", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_PATH + "figure14_attention_clusters.png", dpi=150)
plt.close(fig)
print("  Saved: figure14_attention_clusters.png")

print("\n  Table 7: Attention-Based Behavioural Insights")
rows = []
for ci in range(3):
    mask = cluster_labels == ci
    true_lbs = np.array(test_true)[mask]
    dominant = class_names[np.bincount(true_lbs).argmax()]
    rows.append({'Cluster': f"Cluster {ci+1} ({persona[ci]})",
                 'n_students': int(mask.sum()),
                 'dominant_outcome': dominant,
                 'peak_attention_day': int(np.argmax(centers[ci])),
                 'mean_peak_weight': float(centers[ci].max())})

attn_df = pd.DataFrame(rows).set_index('Cluster')
print(attn_df.to_string())
attn_df.to_csv(OUT_PATH + "table7_attention_insights.csv")

print("\n" + "=" * 60)
print("HTBT COMPLETE")
print("=" * 60)
print(f"  Accuracy={htbt_acc:.4f}, F1={htbt_f1:.4f}, AUC={htbt_auc:.4f}")
print(f"\nOutputs saved to: {OUT_PATH}")
