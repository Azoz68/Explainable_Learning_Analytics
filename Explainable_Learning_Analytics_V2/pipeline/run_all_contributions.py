import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import os
import warnings
warnings.filterwarnings('ignore')

OUT_PATH = os.path.join(str(RESULTS_DIR), "contribution_results", "")
DATA_PATH = os.path.join(str(DATA_DIR), "")
os.makedirs(OUT_PATH, exist_ok=True)

import time

def section(title):
    print("\n" + "="*65)
    print(f"  {title}")
    print("="*65)

section("Contribution 2: Temporal Prediction Window Analysis")
t0 = time.time()
from contribution2_temporal_window import run_temporal_analysis
res2 = run_temporal_analysis()
print(f"  [Done in {time.time()-t0:.1f}s]")

section("Contribution 3: Formal Leakage Taxonomy")
t0 = time.time()
from contribution3_leakage_taxonomy import run_leakage_taxonomy
res3 = run_leakage_taxonomy()
print(f"  [Done in {time.time()-t0:.1f}s]")

section("Contribution 4: Explanation Stability Analysis")
t0 = time.time()
from contribution4_stability_analysis import run_stability_analysis
res4, reliable = run_stability_analysis()
print(f"  [Done in {time.time()-t0:.1f}s]")

section("Contribution 1: Adaptive Fidelity-Weighted Hybrid XAI")
t0 = time.time()

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap, lime, lime.lime_tabular
from scipy.stats import entropy as scipy_entropy

CUTOFF_DAY = 30
np.random.seed(42)

print("  Loading data for Contribution 1...")
student_info   = pd.read_csv(DATA_PATH+"studentInfo.csv").replace('?', np.nan)
student_vle    = pd.read_csv(DATA_PATH+"studentVle.csv").replace('?', np.nan)
assessments    = pd.read_csv(DATA_PATH+"assessments.csv").replace('?', np.nan)
student_assess = pd.read_csv(DATA_PATH+"studentAssessment.csv").replace('?', np.nan)
courses        = pd.read_csv(DATA_PATH+"courses.csv").replace('?', np.nan)
student_reg    = pd.read_csv(DATA_PATH+"studentRegistration.csv").replace('?', np.nan)

sv = student_vle.copy()
sv['date']      = pd.to_numeric(sv['date'],      errors='coerce')
sv['sum_click'] = pd.to_numeric(sv['sum_click'], errors='coerce').fillna(0)
sv = sv[sv['date'] <= CUTOFF_DAY]

vle_agg = (sv.groupby(['code_module','code_presentation','id_student'])
             .agg(total_clicks=('sum_click','sum'),
                  avg_clicks  =('sum_click','mean'),
                  days_active =('date','nunique'),
                  last_access =('date','max'))
             .reset_index())

def entr(g):
    c = g['sum_click'].values
    return float(scipy_entropy(c/c.sum())) if c.sum()>0 else 0.0
entr_df = (sv.groupby(['code_module','code_presentation','id_student'])
             .apply(entr).reset_index().rename(columns={0:'activity_entropy'}))
vle_agg = vle_agg.merge(entr_df, on=['code_module','code_presentation','id_student'], how='left')
vle_agg['activity_entropy'] = vle_agg['activity_entropy'].fillna(0)

af = student_assess.merge(assessments, on='id_assessment', how='left')
af['score']          = pd.to_numeric(af['score'],          errors='coerce')
af['weight']         = pd.to_numeric(af['weight'],         errors='coerce')
af['date_submitted'] = pd.to_numeric(af['date_submitted'], errors='coerce')
af = af[af['date_submitted'] <= CUTOFF_DAY]
af['wc'] = af['score'] * af['weight'] / 100.0

if len(af) > 0:
    assess_agg = (af.groupby(['id_student','code_module','code_presentation'])
                    .agg(weighted_score =('wc','sum'),
                         num_assessments=('id_assessment','count'),
                         mean_score     =('score','mean'))
                    .reset_index())
else:
    assess_agg = pd.DataFrame(columns=['id_student','code_module','code_presentation',
                                        'weighted_score','num_assessments','mean_score'])

reg = student_reg.copy()
reg['date_registration'] = pd.to_numeric(reg['date_registration'], errors='coerce')
reg = reg.merge(courses, on=['code_module','code_presentation'], how='left')
reg['module_presentation_length'] = pd.to_numeric(reg['module_presentation_length'], errors='coerce')
reg['study_duration_proxy'] = (
    reg['module_presentation_length'] - reg['date_registration']
).clip(0).fillna(0)
reg['first_registration'] = reg['date_registration'].fillna(0)
reg_feat = reg[['code_module','code_presentation','id_student',
                'study_duration_proxy','first_registration']]

df = (student_info
      .merge(vle_agg,    on=['code_module','code_presentation','id_student'], how='left')
      .merge(assess_agg, on=['code_module','code_presentation','id_student'], how='left')
      .merge(reg_feat,   on=['code_module','code_presentation','id_student'], how='left'))

for c in ['total_clicks','avg_clicks','days_active','last_access','activity_entropy',
          'weighted_score','num_assessments','mean_score','study_duration_proxy','first_registration']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

df['engagement_efficiency'] = df['weighted_score'] / (df['total_clicks']+1)
df['CBII'] = 0.5*df['weighted_score'] + 0.3*df['activity_entropy']
df['TPI']  = df['study_duration_proxy'] / (df['num_assessments']+1)

le_tgt = LabelEncoder()
y_all  = le_tgt.fit_transform(df['final_result'].astype(str))
class_names = list(le_tgt.classes_)

excl     = ['id_student','final_result']
cat_cols = ['gender','region','highest_education','imd_band',
            'age_band','disability','code_module','code_presentation']
feat_cols = [c for c in df.columns if c not in excl]

idx = np.arange(len(df))
idx_tr, idx_te = train_test_split(idx, test_size=0.20, random_state=42, stratify=y_all)
y_tr = y_all[idx_tr]; y_te = y_all[idx_te]

df_tr = df.iloc[idx_tr].copy()
df_te = df.iloc[idx_te].copy()
for col in cat_cols:
    le = LabelEncoder().fit(df_tr[col].astype(str))
    seen = set(le.classes_)
    for sp in [df_tr, df_te]:
        sp[col] = sp[col].astype(str).apply(lambda v: v if v in seen else le.classes_[0])
        sp[col] = le.transform(sp[col])

df_tr[feat_cols] = df_tr[feat_cols].fillna(0)
df_te[feat_cols] = df_te[feat_cols].fillna(0)

X_tr = df_tr[feat_cols].values.astype(float)
X_te = df_te[feat_cols].values.astype(float)

sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr)
X_te_s = sc.transform(X_te)

print("  Training XGBoost for Contribution 1...")
clf = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=5,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, n_jobs=-1, verbosity=0)
clf.fit(X_tr_s, y_tr)

print("  Computing SHAP (50 samples)...")
try:
    exp_shap = shap.TreeExplainer(clf)
    sv_raw   = exp_shap.shap_values(X_te_s[:50])
    sv_arr   = np.array(sv_raw)
    if sv_arr.ndim == 3 and sv_arr.shape[2] == len(class_names):

        shap_abs = np.abs(sv_arr).mean(axis=2)
    else:
        shap_abs = np.abs(sv_arr)
    shap_norm = shap_abs / (shap_abs.sum(axis=1, keepdims=True) + 1e-9)
except Exception as e:
    print(f"  TreeSHAP failed ({e}), using KernelSHAP...")
    bg = shap.kmeans(X_tr_s, 20)
    exp_shap  = shap.KernelExplainer(clf.predict_proba, bg)
    sv_raw    = exp_shap.shap_values(X_te_s[:50], nsamples=50)
    sv_arr    = np.abs(np.array(sv_raw))
    if sv_arr.ndim == 3:
        sv_arr = sv_arr.mean(axis=2)
    shap_norm = sv_arr / (sv_arr.sum(axis=1, keepdims=True) + 1e-9)

print("  Computing LIME (50 samples)...")
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_tr_s, feature_names=feat_cols,
    class_names=class_names,
    discretize_continuous=True, random_state=42)
lime_mat = np.zeros((50, len(feat_cols)))
for i in range(50):
    exp = lime_exp.explain_instance(X_te_s[i], clf.predict_proba,
                                     num_features=len(feat_cols))
    for fidx, w in exp.as_map()[exp.available_labels()[0]]:
        lime_mat[i, fidx] = abs(w)
lime_norm = lime_mat / (lime_mat.sum(axis=1, keepdims=True) + 1e-9)

y_sample = y_te[:50]

from contribution1_adaptive_hybrid import evaluate_hybrid_methods
hybrid_adp, weights, fid_results, stat_results = evaluate_hybrid_methods(
    X_te_s[:50], y_sample, shap_norm, lime_norm,
    clf, k=5, baseline=None, n_bootstrap=200,
    feature_names=feat_cols, out_path=OUT_PATH)

print(f"  [Done in {time.time()-t0:.1f}s]")

section("ALL CONTRIBUTIONS COMPLETE")
print(f"  Output directory: {OUT_PATH}")
for f in sorted(os.listdir(OUT_PATH)):
    print(f"    {f}")
