import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
import sys, os, warnings
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = os.path.join(str(DATA_DIR), "")
OUT_PATH = os.path.join(str(RESULTS_DIR), "oulad_results", "")
WINDOW    = 30
np.random.seed(42); torch.manual_seed(42)

print("=" * 55)
print("HTBT Evaluation (loading saved checkpoint)")
print("=" * 55)

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

def compute_entropy(grp):
    c = grp['sum_click'].values
    return float(scipy_entropy(c/c.sum())) if c.sum()>0 else 0.0
def compute_burstiness(grp):
    d = np.sort(grp['date'].dropna().unique().astype(float))
    if len(d)<2: return 0.0
    g = np.diff(d); m = g.mean()
    return float(g.std()/m) if m>0 else 0.0

entr_df = sv.groupby(['code_module','code_presentation','id_student']).apply(compute_entropy).reset_index().rename(columns={0:'activity_entropy'})
burs_df = sv.groupby(['code_module','code_presentation','id_student']).apply(compute_burstiness).reset_index().rename(columns={0:'burstiness'})
vle_agg = vle_agg.merge(entr_df,on=['code_module','code_presentation','id_student'],how='left').merge(burs_df,on=['code_module','code_presentation','id_student'],how='left')
vle_agg[['activity_entropy','burstiness']] = vle_agg[['activity_entropy','burstiness']].fillna(0)

af = student_assess.merge(assessments, on='id_assessment', how='left')
af['score']  = pd.to_numeric(af['score'],  errors='coerce')
af['weight'] = pd.to_numeric(af['weight'], errors='coerce')
af['weighted_contrib'] = af['score'] * af['weight'] / 100.0
assess_agg = (af.groupby(['id_student','code_module','code_presentation'])
                .agg(weighted_score=('weighted_contrib','sum'), num_assessments=('id_assessment','count'),
                     mean_score=('score','mean'), last_submission=('date_submitted','max'))
                .reset_index())
def perf_trend(grp):
    scores = pd.to_numeric(grp.sort_values('date_submitted')['score'],errors='coerce').dropna().values
    if len(scores)<2: return 0.0
    try: return float(np.polyfit(np.arange(len(scores),dtype=float), scores, 1)[0])
    except: return 0.0
trend_df = af.groupby(['id_student','code_module','code_presentation']).apply(perf_trend).reset_index().rename(columns={0:'perf_trend'})
assess_agg = assess_agg.merge(trend_df, on=['id_student','code_module','code_presentation'], how='left')
assess_agg[['weighted_score','mean_score','last_submission','perf_trend']] = assess_agg[['weighted_score','mean_score','last_submission','perf_trend']].fillna(0)

reg = student_reg.copy()
reg['date_registration']   = pd.to_numeric(reg['date_registration'],   errors='coerce')
reg['date_unregistration'] = pd.to_numeric(reg['date_unregistration'], errors='coerce')
reg = reg.merge(courses, on=['code_module','code_presentation'], how='left')
reg['module_presentation_length'] = pd.to_numeric(reg['module_presentation_length'], errors='coerce')
reg['study_duration'] = np.where(reg['date_unregistration'].notna(),
    reg['date_unregistration'] - reg['date_registration'],
    reg['module_presentation_length'] - reg['date_registration']).clip(0)
reg['study_duration'] = reg['study_duration'].fillna(0)
reg['first_registration'] = reg['date_registration'].fillna(0)
reg_feat = reg[['code_module','code_presentation','id_student','study_duration','first_registration']]

df = (student_info
      .merge(vle_agg,    on=['code_module','code_presentation','id_student'], how='left')
      .merge(assess_agg, on=['code_module','code_presentation','id_student'], how='left')
      .merge(reg_feat,   on=['code_module','code_presentation','id_student'], how='left'))

df['engagement_efficiency'] = df['weighted_score'] / (df['total_clicks'] + 1)
df['CBII'] = 0.5*df['weighted_score'] + 0.3*df['activity_entropy'].fillna(0) + 0.2*df['perf_trend'].fillna(0)
df['TPI']  = df['study_duration'].fillna(0) / (df['num_assessments'].fillna(0) + 1)
df['dropout_risk'] = (1/(df['study_duration'].fillna(0)+1) + 1/(df['total_clicks'].fillna(0)+1) + 1/(df['weighted_score'].fillna(0)+1))

print("[2] Building 30-day sequences...")
sv2 = student_vle.copy()
sv2['date']      = pd.to_numeric(sv2['date'], errors='coerce')
sv2['sum_click'] = pd.to_numeric(sv2['sum_click'], errors='coerce').fillna(0)
end_d = sv2.groupby(['code_module','code_presentation','id_student'])['date'].max().reset_index().rename(columns={'date':'end_date'})
sv2 = sv2.merge(end_d, on=['code_module','code_presentation','id_student']).dropna(subset=['date','end_date'])
sv2['dfe'] = sv2['end_date'] - sv2['date']
sv_w = sv2[sv2['dfe']<WINDOW]
sv_p = sv_w.groupby(['code_module','code_presentation','id_student','dfe'])['sum_click'].sum().reset_index()
def build_seq(g):
    s = np.zeros(WINDOW, dtype=np.float32)
    for _, r in g.iterrows():
        i = int(r['dfe'])
        if 0<=i<WINDOW: s[WINDOW-1-i] = r['sum_click']
    return s
seq_df = sv_p.groupby(['code_module','code_presentation','id_student']).apply(build_seq).reset_index().rename(columns={0:'sequence'})
df = df.merge(seq_df, on=['code_module','code_presentation','id_student'], how='left')
df['sequence'] = df['sequence'].apply(lambda x: x if isinstance(x,np.ndarray) else np.zeros(WINDOW,dtype=np.float32))

for col in ['gender','region','highest_education','imd_band','age_band','disability','code_module','code_presentation']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
target_le = LabelEncoder()
df['label'] = target_le.fit_transform(df['final_result'].astype(str))
class_names = list(target_le.classes_)

excl = ['id_student','final_result','label','sequence']
feature_cols = [c for c in df.columns if c not in excl]
df[feature_cols] = df[feature_cols].fillna(0)
scaler = StandardScaler()
static_X = scaler.fit_transform(df[feature_cols].values)
labels   = df['label'].values
n_classes = len(class_names)
seqs = np.stack(df['sequence'].values)
mc = seqs.max()
if mc>0: seqs = seqs/mc
print(f"   static_X={static_X.shape}, seqs={seqs.shape}, classes={class_names}")

idx_all = np.arange(len(labels))
idx_temp, idx_te = train_test_split(idx_all, test_size=0.20, random_state=42, stratify=labels)
idx_tr,   idx_vl = train_test_split(idx_temp, test_size=0.20, random_state=42, stratify=labels[idx_temp])

def to_tensors(idx):
    return (torch.tensor(seqs[idx], dtype=torch.float32).unsqueeze(-1),
            torch.tensor(static_X[idx], dtype=torch.float32),
            torch.tensor(labels[idx], dtype=torch.long))

seq_te, st_te, lb_te = to_tensors(idx_te)
test_dl = DataLoader(TensorDataset(seq_te, st_te, lb_te), batch_size=256)

class HTBTModel(nn.Module):
    def __init__(self, seq_len, n_static, n_classes, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.temporal_proj = nn.Linear(1, d_model)
        enc_l = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_ff,dropout=dropout,batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_l, num_layers=num_layers)
        self.static_mlp = nn.Sequential(nn.Linear(n_static,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,d_model),nn.ReLU())
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model,num_heads=nhead,dropout=dropout,batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(d_model*2,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,n_classes))
    def forward(self, seq, static):
        t = self.temporal_proj(seq)
        t_enc = self.transformer(t)
        s = self.static_mlp(static)
        attn_out, attn_w = self.cross_attn(s.unsqueeze(1), t_enc, t_enc)
        attn_out = attn_out.squeeze(1)
        align_loss = nn.functional.mse_loss(t_enc.mean(1), s)
        return self.classifier(torch.cat([attn_out, s], dim=1)), attn_w, align_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_static = static_X.shape[1]
model = HTBTModel(WINDOW, n_static, n_classes).to(device)
model.load_state_dict(torch.load(OUT_PATH + "htbt_best.pt", map_location=device))
model.eval()
print(f"\n[3] Model loaded from htbt_best.pt | device={device}")

test_preds, test_true, all_probs, all_attn = [], [], [], []
with torch.no_grad():
    for sb, stb, lb in test_dl:
        logits, attn_w, _ = model(sb.to(device), stb.to(device))
        probs = torch.softmax(logits,1).cpu().numpy()
        test_preds.extend(logits.argmax(1).cpu().numpy())
        test_true.extend(lb.numpy())
        all_probs.extend(probs.tolist())
        all_attn.append(attn_w.squeeze(1).cpu().numpy())

acc  = accuracy_score(test_true, test_preds)
f1   = f1_score(test_true, test_preds, average='macro', zero_division=0)
prec = precision_score(test_true, test_preds, average='macro', zero_division=0)
rec  = recall_score(test_true, test_preds, average='macro', zero_division=0)
auc  = roc_auc_score(test_true, np.array(all_probs), multi_class='ovr', average='macro')

print(f"\n  Table 6: HTBT Test Set Performance")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  Macro-F1 : {f1:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
pd.DataFrame([{'accuracy':acc,'macro_precision':prec,'macro_recall':rec,'macro_f1':f1,'roc_auc':auc}]).to_csv(OUT_PATH+"table6_htbt_test.csv",index=False)

print("\n[4] Attention cluster analysis...")
attn_arr = np.concatenate(all_attn, axis=0)
km = KMeans(n_clusters=3, random_state=42, n_init=10)
cl = km.fit_predict(attn_arr)
centers = km.cluster_centers_
persona = ['Early Engagers', 'Late Bloomers', 'Sporadic']
days = np.arange(WINDOW)

fig, axes = plt.subplots(1,3,figsize=(15,5),sharey=True)
for ci in range(3):
    m = attn_arr[cl==ci]; mn,sd = centers[ci], m.std(axis=0)
    axes[ci].plot(days,mn,color='navy',label='Mean')
    axes[ci].fill_between(days,mn-sd,mn+sd,alpha=0.2,color='navy')
    axes[ci].set_title(f"Cluster {ci+1}: {persona[ci]}\n(n={(cl==ci).sum()})",fontsize=10)
    axes[ci].set_xlabel("Day in 30-day window")
    if ci==0: axes[ci].set_ylabel("Attention weight")
plt.suptitle("Figure 14: Clustered Attention Patterns (HTBT)",fontsize=12)
plt.tight_layout()
fig.savefig(OUT_PATH+"figure14_attention_clusters.png",dpi=150)
plt.close(fig)
print("  Saved: figure14_attention_clusters.png")

rows=[]
for ci in range(3):
    mask=(cl==ci); tl=np.array(test_true)[mask]
    rows.append({'Cluster':f"Cluster {ci+1} ({persona[ci]})",
                 'n_students':int(mask.sum()),
                 'dominant_outcome':class_names[np.bincount(tl).argmax()],
                 'peak_attention_day':int(np.argmax(centers[ci])),
                 'mean_peak_weight':float(centers[ci].max())})
attn_df = pd.DataFrame(rows).set_index('Cluster')
print("\n  Table 7: Attention-Based Behavioural Insights")
print(attn_df.to_string())
attn_df.to_csv(OUT_PATH+"table7_attention_insights.csv")

print("\n" + "="*55)
print("HTBT EVALUATION COMPLETE")
print("="*55)
print(f"  Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
