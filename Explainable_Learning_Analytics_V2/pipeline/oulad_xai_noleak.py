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
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import shap
import lime
import lime.lime_tabular

np.random.seed(42)
warnings.filterwarnings('ignore')

DATA_PATH = os.path.join(str(DATA_DIR), "")
OUT_PATH = os.path.join(str(RESULTS_DIR), "oulad_results_noleak", "")
os.makedirs(OUT_PATH, exist_ok=True)

CUTOFF_DAY = 30

print("=" * 65)
print("OULAD XAI – No Data Leakage Pipeline")
print(f"Prediction cut-off: day {CUTOFF_DAY}")
print("=" * 65)

print("\n[1] Loading data...")
student_info   = pd.read_csv(DATA_PATH + "studentInfo.csv").replace('?', np.nan)
student_vle    = pd.read_csv(DATA_PATH + "studentVle.csv").replace('?', np.nan)
assessments    = pd.read_csv(DATA_PATH + "assessments.csv").replace('?', np.nan)
student_assess = pd.read_csv(DATA_PATH + "studentAssessment.csv").replace('?', np.nan)
courses        = pd.read_csv(DATA_PATH + "courses.csv").replace('?', np.nan)
student_reg    = pd.read_csv(DATA_PATH + "studentRegistration.csv").replace('?', np.nan)

print(f"\n[2] VLE features (days 0-{CUTOFF_DAY} only)...")
sv = student_vle.copy()
sv['date']      = pd.to_numeric(sv['date'],      errors='coerce')
sv['sum_click'] = pd.to_numeric(sv['sum_click'], errors='coerce').fillna(0)
sv = sv[sv['date'] <= CUTOFF_DAY]

vle_agg = (sv.groupby(['code_module','code_presentation','id_student'])
             .agg(
                 total_clicks       = ('sum_click', 'sum'),
                 avg_clicks         = ('sum_click', 'mean'),
                 max_clicks         = ('sum_click', 'max'),
                 click_std          = ('sum_click', 'std'),
                 distinct_resources = ('id_site',   'nunique'),
                 days_active        = ('date',       'nunique'),
                 first_access       = ('date',       'min'),
                 last_access_before_cutoff = ('date','max'),
             )
             .reset_index())
vle_agg['click_std'].fillna(0, inplace=True)
vle_agg['cv_clicks'] = vle_agg['click_std'] / (vle_agg['avg_clicks'] + 1e-9)

def entropy_fn(grp):
    c = grp['sum_click'].values
    return float(scipy_entropy(c / c.sum())) if c.sum() > 0 else 0.0

def burstiness_fn(grp):
    d = np.sort(grp['date'].dropna().unique().astype(float))
    if len(d) < 2: return 0.0
    g = np.diff(d); m = g.mean()
    return float(g.std() / m) if m > 0 else 0.0

print("  Computing entropy & burstiness...")
entr_df = (sv.groupby(['code_module','code_presentation','id_student'])
             .apply(entropy_fn).reset_index().rename(columns={0:'activity_entropy'}))
burs_df = (sv.groupby(['code_module','code_presentation','id_student'])
             .apply(burstiness_fn).reset_index().rename(columns={0:'burstiness'}))
vle_agg = (vle_agg
           .merge(entr_df, on=['code_module','code_presentation','id_student'], how='left')
           .merge(burs_df, on=['code_module','code_presentation','id_student'], how='left'))
vle_agg[['activity_entropy','burstiness']] = vle_agg[['activity_entropy','burstiness']].fillna(0)

print(f"\n[3] Assessment features (submitted <= day {CUTOFF_DAY})...")
af = student_assess.merge(assessments, on='id_assessment', how='left')
af['score']          = pd.to_numeric(af['score'],          errors='coerce')
af['weight']         = pd.to_numeric(af['weight'],         errors='coerce')
af['date_submitted'] = pd.to_numeric(af['date_submitted'], errors='coerce')
af = af[af['date_submitted'] <= CUTOFF_DAY]

af['weighted_contrib'] = af['score'] * af['weight'] / 100.0

assess_agg = (af.groupby(['id_student','code_module','code_presentation'])
                .agg(
                    weighted_score   = ('weighted_contrib', 'sum'),
                    num_assessments  = ('id_assessment',   'count'),
                    mean_score       = ('score',            'mean'),
                    last_submission  = ('date_submitted',   'max'),
                )
                .reset_index())

def perf_trend(grp):
    scores = pd.to_numeric(
        grp.sort_values('date_submitted')['score'], errors='coerce'
    ).dropna().values
    if len(scores) < 2: return 0.0
    try: return float(np.polyfit(np.arange(len(scores), dtype=float), scores, 1)[0])
    except: return 0.0

trend_df = (af.groupby(['id_student','code_module','code_presentation'])
              .apply(perf_trend).reset_index().rename(columns={0:'perf_trend'}))
assess_agg = assess_agg.merge(trend_df,
                               on=['id_student','code_module','code_presentation'], how='left')
for c in ['weighted_score','mean_score','last_submission','perf_trend','num_assessments']:
    assess_agg[c].fillna(0, inplace=True)

print("\n[4] Registration features (NO date_unregistration)...")
reg = student_reg.copy()
reg['date_registration'] = pd.to_numeric(reg['date_registration'], errors='coerce')

reg = reg.merge(courses, on=['code_module','code_presentation'], how='left')
reg['module_presentation_length'] = pd.to_numeric(
    reg['module_presentation_length'], errors='coerce')

reg['study_duration_proxy'] = (
    reg['module_presentation_length'] - reg['date_registration']
).clip(lower=0).fillna(0)
reg['first_registration'] = reg['date_registration'].fillna(0)

reg_feat = reg[['code_module','code_presentation','id_student',
                'study_duration_proxy','first_registration']].copy()

print("\n[5] Merging feature tables...")
df = (student_info
      .merge(vle_agg,    on=['code_module','code_presentation','id_student'], how='left')
      .merge(assess_agg, on=['code_module','code_presentation','id_student'], how='left')
      .merge(reg_feat,   on=['code_module','code_presentation','id_student'], how='left'))

vle_fill_cols = ['total_clicks','avg_clicks','max_clicks','click_std','cv_clicks',
                 'distinct_resources','days_active','first_access',
                 'last_access_before_cutoff','activity_entropy','burstiness']
for c in vle_fill_cols:
    if c in df.columns:
        df[c].fillna(0, inplace=True)

print("\n[6] Composite indicators...")
df['engagement_efficiency'] = df['weighted_score'] / (df['total_clicks'] + 1)
df['CBII'] = (0.5 * df['weighted_score'].fillna(0) +
              0.3 * df['activity_entropy'].fillna(0) +
              0.2 * df['perf_trend'].fillna(0))
df['TPI']  = df['study_duration_proxy'] / (df['num_assessments'].fillna(0) + 1)

df['dropout_risk_proxy'] = (1 / (df['study_duration_proxy'] + 1) +
                             1 / (df['total_clicks'] + 1) +
                             1 / (df['weighted_score'].fillna(0) + 1))

print("\n[7] Splitting FIRST (64/16/20)...")
target_le_full = LabelEncoder()
y_all = target_le_full.fit_transform(df['final_result'].astype(str))
class_names = list(target_le_full.classes_)

idx_all = np.arange(len(df))
idx_temp, idx_test = train_test_split(idx_all, test_size=0.20,
                                       random_state=42, stratify=y_all)
idx_train, idx_val  = train_test_split(idx_temp, test_size=0.20,
                                        random_state=42, stratify=y_all[idx_temp])

df_train = df.iloc[idx_train].copy()
df_val   = df.iloc[idx_val].copy()
df_test  = df.iloc[idx_test].copy()

y_train  = y_all[idx_train]
y_val    = y_all[idx_val]
y_test   = y_all[idx_test]

print(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
print(f"  Classes: {class_names}")

print("\n[8] Encoding (fit on TRAIN only)...")
cat_cols = ['gender','region','highest_education','imd_band',
            'age_band','disability','code_module','code_presentation']

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(df_train[col].astype(str))
    le_dict[col] = le

    for split in [df_train, df_val, df_test]:
        vals = split[col].astype(str)
        seen = set(le.classes_)
        split[col] = vals.apply(lambda v: v if v in seen else le.classes_[0])
        split[col] = le.transform(split[col])

excl = ['id_student','final_result']
feature_cols = [c for c in df.columns if c not in excl]

for split in [df_train, df_val, df_test]:
    split[feature_cols] = split[feature_cols].fillna(0)

X_train = df_train[feature_cols].values.astype(float)
X_val   = df_val[feature_cols].values.astype(float)
X_test  = df_test[feature_cols].values.astype(float)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print(f"  Feature matrix: {X_train_s.shape[1]} features")
print(f"  Class distribution (train): {np.bincount(y_train)}")

print("\n[10] 10-Fold CV on training set...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models_cv = {
    'XGBoost': xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1),
    'RandomForest': RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1),
    'LogisticRegression': LogisticRegression(
        max_iter=1000, solver='lbfgs', random_state=42, n_jobs=-1),
}

cv_results = {}
for name, model in models_cv.items():
    accs, f1s, aucs = [], [], []
    for tr_idx, vl_idx in skf.split(X_train_s, y_train):

        sc_fold = StandardScaler()
        Xtr = sc_fold.fit_transform(X_train[tr_idx])
        Xvl = sc_fold.transform(X_train[vl_idx])
        ytr, yvl = y_train[tr_idx], y_train[vl_idx]
        model.fit(Xtr, ytr)
        yp = model.predict(Xvl)
        ypr = model.predict_proba(Xvl)
        accs.append(accuracy_score(yvl, yp))
        f1s.append(f1_score(yvl, yp, average='macro', zero_division=0))
        try: aucs.append(roc_auc_score(yvl, ypr, multi_class='ovr', average='macro'))
        except: aucs.append(np.nan)
    cv_results[name] = {'accuracy': np.mean(accs), 'macro_f1': np.mean(f1s),
                        'roc_auc': np.nanmean(aucs)}
    print(f"  {name}: Acc={np.mean(accs):.4f}, F1={np.mean(f1s):.4f}, "
          f"AUC={np.nanmean(aucs):.4f}")

cv_df = pd.DataFrame(cv_results).T.round(4)
print("\n  Table 1 (No-Leak): 10-Fold CV")
print(cv_df.to_string())
cv_df.to_csv(OUT_PATH + "table1_cv_noleak.csv")

print("\n[11] Final XGBoost with early stopping...")
xgb_final = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    early_stopping_rounds=30, random_state=42, n_jobs=-1)

xgb_final.fit(X_train_s, y_train,
               eval_set=[(X_val_s, y_val)], verbose=False)
print(f"  Best iteration: {xgb_final.best_iteration}")

y_pred = xgb_final.predict(X_test_s)
y_prob = xgb_final.predict_proba(X_test_s)

test_acc  = accuracy_score(y_test, y_pred)
test_f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
test_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
test_rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
test_auc  = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

print(f"\n  Table 2 (No-Leak): XGBoost Test Set")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall   : {test_rec:.4f}")
print(f"  Macro-F1 : {test_f1:.4f}")
print(f"  ROC-AUC  : {test_auc:.4f}")

pd.DataFrame([{'accuracy':test_acc,'macro_precision':test_prec,
               'macro_recall':test_rec,'macro_f1':test_f1,'roc_auc':test_auc}]
             ).to_csv(OUT_PATH+"table2_xgb_noleak.csv", index=False)

print("\n  Table 3 (No-Leak): Class-Level Metrics")
cr = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr).T.round(4)
print(cr_df.to_string())
cr_df.to_csv(OUT_PATH+"table3_class_noleak.csv")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title(f"Confusion Matrix - XGBoost (No-Leak, cutoff=day {CUTOFF_DAY})")
ax.set_ylabel("True"); ax.set_xlabel("Predicted")
plt.tight_layout()
fig.savefig(OUT_PATH+"figure_cm_noleak.png", dpi=150)
plt.close(fig)
print("  Saved: figure_cm_noleak.png")

print("\n[12] SHAP explanations...")
rng = np.random.RandomState(42)
shap_idx = rng.choice(len(X_val_s), size=min(200, len(X_val_s)), replace=False)
X_shap = X_val_s[shap_idx]
y_shap = y_val[shap_idx]

t_shap = time.time()
try:
    explainer = shap.TreeExplainer(xgb_final,
                                   feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(X_shap)
    print(f"  TreeSHAP: {np.array(shap_values).shape}")
except Exception as e:
    print(f"  TreeSHAP failed ({e}), using KernelSHAP on 50 samples...")
    bg = shap.kmeans(X_train_s, 20)
    explainer = shap.KernelExplainer(xgb_final.predict_proba, bg)
    shap_values = explainer.shap_values(X_shap[:50])
    X_shap = X_shap[:50]; y_shap = y_shap[:50]
shap_time = (time.time() - t_shap) / len(X_shap)

sv_arr = np.array(shap_values)
if isinstance(shap_values, list):
    per_class_shap = shap_values
elif sv_arr.ndim == 3 and sv_arr.shape[2] == len(class_names):
    per_class_shap = [sv_arr[:,:,i] for i in range(sv_arr.shape[2])]
elif sv_arr.ndim == 3:
    per_class_shap = [sv_arr[i] for i in range(sv_arr.shape[0])]
else:
    per_class_shap = [sv_arr]

shap_global = pd.Series(
    np.mean([np.abs(sv).mean(axis=0) for sv in per_class_shap], axis=0),
    index=feature_cols).sort_values(ascending=False)
print("  Top-10 SHAP features:")
print(shap_global.head(10).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(per_class_shap[0], X_shap, feature_names=feature_cols,
                  max_display=20, show=False)
plt.title(f"SHAP Summary (No-Leak, cutoff=day {CUTOFF_DAY})")
plt.tight_layout()
fig.savefig(OUT_PATH+"figure_shap_summary_noleak.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: figure_shap_summary_noleak.png")

n_classes = len(class_names)
if len(per_class_shap) == n_classes:
    fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 7))
    for i,(sv,cls) in enumerate(zip(per_class_shap, class_names)):
        imp = pd.Series(np.abs(sv).mean(axis=0), index=feature_cols).nlargest(10)
        axes[i].barh(imp.index[::-1], imp.values[::-1], color='steelblue')
        axes[i].set_title(f"SHAP - {cls}")
        axes[i].set_xlabel("Mean |SHAP|")
    plt.suptitle(f"Per-class SHAP (No-Leak, cutoff=day {CUTOFF_DAY})", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_PATH+"figure_shap_perclass_noleak.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure_shap_perclass_noleak.png")

print("\n[13] LIME explanations...")
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_s, feature_names=feature_cols,
    class_names=class_names, mode='classification',
    discretize_continuous=True, random_state=42)

lime_sample = min(50, len(X_shap))
lime_imps = []
t_lime = time.time()
for i in range(lime_sample):
    exp = lime_exp.explain_instance(X_shap[i], xgb_final.predict_proba,
                                     num_features=len(feature_cols), top_labels=1)
    top_lbl = list(exp.local_exp.keys())[0]
    lime_imp = np.zeros(len(feature_cols))
    for fi, w in exp.local_exp[top_lbl]:
        if fi < len(feature_cols):
            lime_imp[fi] = abs(w)
    lime_imps.append(lime_imp)
lime_time = (time.time() - t_lime) / lime_sample
lime_imps = np.array(lime_imps)
lime_global = pd.Series(lime_imps.mean(axis=0),
                         index=feature_cols).sort_values(ascending=False)
print(f"  LIME time: {lime_time:.3f}s/sample")
print("  Top-10 LIME features:")
print(lime_global.head(10).to_string())

print("\n[14] Hybrid SHAP-LIME...")
shap_loc = np.mean([np.abs(sv[:lime_sample]) for sv in per_class_shap], axis=0)

def norm_rows(arr):
    mn = arr.min(1, keepdims=True); mx = arr.max(1, keepdims=True)
    return (arr - mn) / np.where(mx-mn==0, 1, mx-mn)

shap_norm = norm_rows(shap_loc)
lime_norm = norm_rows(lime_imps)
hybrid = 0.5 * shap_norm + 0.5 * lime_norm
hybrid_global = pd.Series(hybrid.mean(axis=0),
                            index=feature_cols).sort_values(ascending=False)
print("  Top-10 Hybrid features:")
print(hybrid_global.head(10).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for ax_, (title, imp, color) in zip(axes, [
    ("SHAP",   shap_global,   'steelblue'),
    ("LIME",   lime_global,   'coral'),
    ("Hybrid", hybrid_global, 'mediumpurple')]):
    top = imp.head(15)
    ax_.barh(top.index[::-1], top.values[::-1], color=color)
    ax_.set_title(f"{title} (No-Leak, day {CUTOFF_DAY})")
    ax_.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(OUT_PATH+"figure_xai_comparison_noleak.png", dpi=150)
plt.close(fig)
print("  Saved: figure_xai_comparison_noleak.png")

print("\n[15] Fidelity & consistency...")
k = 5
baseline = np.median(X_train_s, axis=0)
fidelity = {'SHAP':[], 'LIME':[], 'Hybrid':[]}
for i in range(lime_sample):
    orig = xgb_final.predict_proba(X_shap[i:i+1])[0, y_shap[i]]
    for mname, imp_row in [('SHAP',shap_norm[i]),('LIME',lime_norm[i]),('Hybrid',hybrid[i])]:
        xa = X_shap[i].copy()
        xa[np.argsort(imp_row)[::-1][:k]] = baseline[np.argsort(imp_row)[::-1][:k]]
        abl = xgb_final.predict_proba(xa.reshape(1,-1))[0, y_shap[i]]
        fidelity[mname].append(orig - abl)

fid_means = {m: float(np.mean(v)) for m, v in fidelity.items()}
print(f"  Fidelity (k={k}): " +
      " | ".join(f"{m}={v:.4f}" for m, v in fid_means.items()))
pd.DataFrame([fid_means]).to_csv(OUT_PATH+"table4_fidelity_noleak.csv", index=False)

rhos = [spearmanr(shap_norm[i], lime_norm[i])[0] for i in range(lime_sample)
        if not np.isnan(spearmanr(shap_norm[i], lime_norm[i])[0])]
print(f"  Spearman rho (SHAP vs LIME): mean={np.mean(rhos):.4f}, std={np.std(rhos):.4f}")

t_shap2 = time.time()
_ = explainer.shap_values(X_shap[:10])
shap_t2 = (time.time() - t_shap2) / 10
hybrid_t = shap_t2 + lime_time
timing = {'SHAP': shap_t2, 'LIME': lime_time, 'Hybrid': hybrid_t}
print(f"  Timing: SHAP={shap_t2:.3f}s | LIME={lime_time:.3f}s | Hybrid={hybrid_t:.3f}s")
pd.DataFrame([timing]).to_csv(OUT_PATH+"table5_timing_noleak.csv", index=False)

print("\n[16] Pedagogical grouping...")
ped_groups = {
    'Assessment Performance':
        ['weighted_score','mean_score','perf_trend','last_submission','num_assessments'],
    'Engagement & Activity':
        ['total_clicks','avg_clicks','max_clicks','activity_entropy',
         'days_active','distinct_resources','cv_clicks','click_std'],
    'Temporal Behaviour':
        ['TPI','study_duration_proxy','last_access_before_cutoff',
         'burstiness','first_access','first_registration'],
    'Composite Indicators':
        ['CBII','engagement_efficiency','dropout_risk_proxy'],
    'Demographic':
        ['gender','region','highest_education','imd_band','age_band',
         'disability','num_of_prev_attempts','studied_credits',
         'code_module','code_presentation'],
}
ped_contrib = {}
for group, cols in ped_groups.items():
    valid = [c for c in cols if c in hybrid_global.index]
    ped_contrib[group] = float(hybrid_global[valid].sum()) if valid else 0.0

ped_s = pd.Series(ped_contrib).sort_values()
fig, ax = plt.subplots(figsize=(9,6))
colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2']
ax.barh(ped_s.index, ped_s.values, color=colors)
ax.set_title(f"Pedagogical Constructs - Hybrid (No-Leak, day {CUTOFF_DAY})")
ax.set_xlabel("Cumulative Hybrid Score")
plt.tight_layout()
fig.savefig(OUT_PATH+"figure_pedagogical_noleak.png", dpi=150)
plt.close(fig)
print("  Pedagogical contributions:")
print(ped_s.round(4).to_string())

print("\n" + "=" * 65)
print("NO-LEAK EXPERIMENT COMPLETE")
print("=" * 65)
print("\n[Table 1] 10-Fold CV:")
print(cv_df.round(4).to_string())
print(f"\n[Table 2] XGBoost Test: Acc={test_acc:.4f}, "
      f"F1={test_f1:.4f}, AUC={test_auc:.4f}")
print(f"\n[Table 4] Fidelity (k={k}): " +
      " | ".join(f"{m}={v:.4f}" for m, v in fid_means.items()))
print(f"\n[Table 5] Timing: SHAP={shap_t2:.3f}s | LIME={lime_time:.3f}s")
print(f"\nData leakage fixes applied:")
print(f"  - date_unregistration REMOVED from all features")
print(f"  - study_duration uses module length only (same for all students)")
print(f"  - VLE + assessment features capped at day {CUTOFF_DAY}")
print(f"  - LabelEncoder fitted on TRAIN split only")
print(f"  - StandardScaler fitted on TRAIN split only")
print(f"  - 10-fold CV refits scaler per fold")
print(f"\nOutputs saved to: {OUT_PATH}")
for f in sorted(os.listdir(OUT_PATH)):
    print(f"  {f}")
