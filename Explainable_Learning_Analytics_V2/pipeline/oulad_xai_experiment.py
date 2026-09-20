import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
import sys, os, warnings, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr, entropy
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

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_PATH = os.path.join(str(DATA_DIR), "")
OUT_PATH = os.path.join(str(RESULTS_DIR), "oulad_results", "")
os.makedirs(OUT_PATH, exist_ok=True)

print("=" * 70)
print("OULAD XAI Experiment - Full Pipeline")
print("=" * 70)

print("\n[1] Loading OULAD tables...")

student_info  = pd.read_csv(DATA_PATH + "studentInfo.csv")
student_vle   = pd.read_csv(DATA_PATH + "studentVle.csv")
vle           = pd.read_csv(DATA_PATH + "vle.csv")
assessments   = pd.read_csv(DATA_PATH + "assessments.csv")
student_assess= pd.read_csv(DATA_PATH + "studentAssessment.csv")
courses       = pd.read_csv(DATA_PATH + "courses.csv")
student_reg   = pd.read_csv(DATA_PATH + "studentRegistration.csv")

print(f"  studentInfo:      {student_info.shape}")
print(f"  studentVle:       {student_vle.shape}")
print(f"  assessments:      {assessments.shape}")
print(f"  studentAssessment:{student_assess.shape}")
print(f"  studentReg:       {student_reg.shape}")

for df in [student_info, student_vle, vle, assessments, student_assess, courses, student_reg]:
    df.replace('?', np.nan, inplace=True)

print("\n[2] Engineering VLE behavioural features...")

vle_agg = (student_vle
           .groupby(['code_module', 'code_presentation', 'id_student'])
           .agg(
               total_clicks    = ('sum_click', 'sum'),
               avg_clicks      = ('sum_click', 'mean'),
               max_clicks      = ('sum_click', 'max'),
               click_std       = ('sum_click', 'std'),
               distinct_resources = ('id_site', 'nunique'),
               days_active     = ('date', 'nunique'),
               first_access    = ('date', 'min'),
               last_activity_day = ('date', 'max'),
           )
           .reset_index()
           )

vle_agg['click_std'].fillna(0, inplace=True)
vle_agg['cv_clicks'] = vle_agg['click_std'] / (vle_agg['avg_clicks'] + 1e-9)

def compute_entropy(group):
    clicks = group['sum_click'].values
    if clicks.sum() == 0:
        return 0.0
    probs = clicks / clicks.sum()
    return float(entropy(probs))

print("  Computing activity entropy (may take a moment)...")
entropy_df = (student_vle
              .groupby(['code_module', 'code_presentation', 'id_student'])
              .apply(compute_entropy)
              .reset_index()
              .rename(columns={0: 'activity_entropy'}))

vle_agg = vle_agg.merge(entropy_df,
                         on=['code_module', 'code_presentation', 'id_student'],
                         how='left')
vle_agg['activity_entropy'].fillna(0, inplace=True)

def compute_burstiness(group):
    dates = np.sort(group['date'].unique().astype(float))
    if len(dates) < 2:
        return 0.0
    gaps = np.diff(dates)
    mu, sigma = gaps.mean(), gaps.std()
    if mu == 0:
        return 0.0
    return float(sigma / mu)

print("  Computing burstiness...")
burst_df = (student_vle
            .groupby(['code_module', 'code_presentation', 'id_student'])
            .apply(compute_burstiness)
            .reset_index()
            .rename(columns={0: 'burstiness'}))

vle_agg = vle_agg.merge(burst_df,
                          on=['code_module', 'code_presentation', 'id_student'],
                          how='left')
vle_agg['burstiness'].fillna(0, inplace=True)

print("\n[3] Engineering assessment features...")

assess_full = student_assess.merge(assessments, on='id_assessment', how='left')
assess_full['score'] = pd.to_numeric(assess_full['score'], errors='coerce')
assess_full['weight'] = pd.to_numeric(assess_full['weight'], errors='coerce')

assess_full['weighted_contrib'] = assess_full['score'] * assess_full['weight'] / 100.0

assess_agg = (assess_full
              .groupby(['id_student', 'code_module', 'code_presentation'])
              .agg(
                  weighted_score   = ('weighted_contrib', 'sum'),
                  num_assessments  = ('id_assessment', 'count'),
                  mean_score       = ('score', 'mean'),
                  last_submission  = ('date_submitted', 'max'),
              )
              .reset_index())

def performance_trend(group):
    group = group.sort_values('date_submitted')
    scores = pd.to_numeric(group['score'], errors='coerce').dropna().values
    if len(scores) < 2:
        return 0.0
    x = np.arange(len(scores), dtype=float)
    try:
        slope = np.polyfit(x, scores, 1)[0]
        return float(slope)
    except Exception:
        return 0.0

trend_df = (assess_full
            .groupby(['id_student', 'code_module', 'code_presentation'])
            .apply(performance_trend)
            .reset_index()
            .rename(columns={0: 'perf_trend'}))

assess_agg = assess_agg.merge(trend_df,
                               on=['id_student', 'code_module', 'code_presentation'],
                               how='left')
assess_agg['perf_trend'].fillna(0, inplace=True)
assess_agg['weighted_score'].fillna(0, inplace=True)
assess_agg['mean_score'].fillna(0, inplace=True)
assess_agg['last_submission'].fillna(0, inplace=True)

print("\n[4] Engineering temporal / registration features...")

reg = student_reg.copy()
reg['date_registration']   = pd.to_numeric(reg['date_registration'],   errors='coerce')
reg['date_unregistration'] = pd.to_numeric(reg['date_unregistration'], errors='coerce')

reg = reg.merge(courses, on=['code_module', 'code_presentation'], how='left')
reg['module_presentation_length'] = pd.to_numeric(reg['module_presentation_length'], errors='coerce')

reg['study_duration'] = np.where(
    reg['date_unregistration'].notna(),
    reg['date_unregistration'] - reg['date_registration'],
    reg['module_presentation_length'] - reg['date_registration']
)
reg['study_duration'] = reg['study_duration'].clip(lower=0).fillna(0)
reg['first_registration'] = reg['date_registration'].fillna(0)

reg_feat = reg[['code_module', 'code_presentation', 'id_student',
                'study_duration', 'first_registration']].copy()

print("\n[5] Merging all feature tables...")

df = student_info.copy()
df = df.merge(vle_agg,    on=['code_module', 'code_presentation', 'id_student'], how='left')
df = df.merge(assess_agg, on=['code_module', 'code_presentation', 'id_student'], how='left')
df = df.merge(reg_feat,   on=['code_module', 'code_presentation', 'id_student'], how='left')

vle_cols = ['total_clicks','avg_clicks','max_clicks','click_std','cv_clicks',
            'distinct_resources','days_active','first_access','last_activity_day',
            'activity_entropy','burstiness']
for c in vle_cols:
    if c in df.columns:
        df[c].fillna(0, inplace=True)

print("\n[6] Computing composite indicators...")

df['engagement_efficiency'] = df['weighted_score'] / (df['total_clicks'] + 1)

df['CBII'] = (0.5 * df['weighted_score'] +
              0.3 * df['activity_entropy'] +
              0.2 * df['perf_trend'])

df['TPI'] = df['study_duration'] / (df['num_assessments'].fillna(0) + 1)

df['dropout_risk'] = (1 / (df['study_duration'] + 1) +
                      1 / (df['total_clicks'] + 1) +
                      1 / (df['weighted_score'] + 1))

print("\n[7] Encoding and scaling features...")

cat_cols = ['gender', 'region', 'highest_education', 'imd_band',
            'age_band', 'disability', 'code_module', 'code_presentation']

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

target_le = LabelEncoder()
df['final_result_enc'] = target_le.fit_transform(df['final_result'].astype(str))
class_names = list(target_le.classes_)
print(f"  Classes: {class_names}")

exclude_cols = ['id_student', 'final_result', 'final_result_enc']
feature_cols = [c for c in df.columns if c not in exclude_cols]

df[feature_cols] = df[feature_cols].fillna(0)

X = df[feature_cols].values
y = df['final_result_enc'].values

print(f"  Feature matrix: {X.shape}, Target: {y.shape}")
print(f"  Class distribution: {np.bincount(y)}")

print("\n[8] Splitting data (64% train, 16% val, 20% test)...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp)

print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print("\n[9] 10-Fold Cross-Validation on Training Set...")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models_cv = {
    'XGBoost': xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1),
    'RandomForest': RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
    'LogisticRegression': LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1, solver='lbfgs'),
}

cv_results = {}
for name, model in models_cv.items():
    accs, f1s, precs, recs, aucs = [], [], [], [], []
    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_train_s, y_train), 1):
        Xtr, Xvl = X_train_s[tr_idx], X_train_s[vl_idx]
        ytr, yvl = y_train[tr_idx],   y_train[vl_idx]
        model.fit(Xtr, ytr)
        ypred = model.predict(Xvl)
        yprob = model.predict_proba(Xvl)
        accs.append(accuracy_score(yvl, ypred))
        f1s.append(f1_score(yvl, ypred, average='macro', zero_division=0))
        precs.append(precision_score(yvl, ypred, average='macro', zero_division=0))
        recs.append(recall_score(yvl, ypred, average='macro', zero_division=0))
        try:
            aucs.append(roc_auc_score(yvl, yprob, multi_class='ovr', average='macro'))
        except Exception:
            aucs.append(np.nan)
    cv_results[name] = {
        'accuracy': np.mean(accs), 'macro_f1': np.mean(f1s),
        'macro_precision': np.mean(precs), 'macro_recall': np.mean(recs),
        'roc_auc': np.nanmean(aucs)
    }
    print(f"  {name}: Acc={np.mean(accs):.4f}, F1={np.mean(f1s):.4f}, "
          f"AUC={np.nanmean(aucs):.4f}")

print("\n  Table 1: 10-Fold CV Results (Training Set)")
cv_df = pd.DataFrame(cv_results).T.round(4)
print(cv_df.to_string())
cv_df.to_csv(OUT_PATH + "table1_cv_results.csv")

print("\n[10] Training final XGBoost model with early stopping...")

xgb_final = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, n_jobs=-1,
    early_stopping_rounds=30
)
xgb_final.fit(
    X_train_s, y_train,
    eval_set=[(X_val_s, y_val)],
    verbose=False
)
print(f"  Best iteration: {xgb_final.best_iteration}")

y_pred_test = xgb_final.predict(X_test_s)
y_prob_test = xgb_final.predict_proba(X_test_s)

test_acc  = accuracy_score(y_test, y_pred_test)
test_f1   = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
test_prec = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
test_rec  = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
test_auc  = roc_auc_score(y_test, y_prob_test, multi_class='ovr', average='macro')

print(f"\n  Table 2: Final XGBoost Test Set Performance")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall   : {test_rec:.4f}")
print(f"  Macro-F1 : {test_f1:.4f}")
print(f"  ROC-AUC  : {test_auc:.4f}")

pd.DataFrame([{'accuracy': test_acc, 'macro_precision': test_prec,
               'macro_recall': test_rec, 'macro_f1': test_f1,
               'roc_auc': test_auc}]).to_csv(OUT_PATH + "table2_xgb_test.csv", index=False)

print("\n  Table 3: Class-Level Metrics (Test Set)")
cr = classification_report(y_test, y_pred_test, target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr).T
print(cr_df.round(4).to_string())
cr_df.to_csv(OUT_PATH + "table3_class_metrics.csv")

cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title("Confusion Matrix - XGBoost (Test Set)", fontsize=13)
ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
plt.tight_layout()
fig.savefig(OUT_PATH + "figure6_confusion_matrix.png", dpi=150)
plt.close(fig)
print("  Saved: figure6_confusion_matrix.png")

print("\n[11] Computing SHAP explanations...")

shap_sample_size = min(200, len(X_val_s))
rng = np.random.RandomState(42)
shap_idx = rng.choice(len(X_val_s), size=shap_sample_size, replace=False)
X_shap = X_val_s[shap_idx]
y_shap = y_val[shap_idx]

shap_time_start = time.time()
try:
    explainer_shap = shap.TreeExplainer(
        xgb_final,
        feature_perturbation='tree_path_dependent')
    shap_values = explainer_shap.shap_values(X_shap)
    shap_time_per = (time.time() - shap_time_start) / shap_sample_size
    print(f"  TreeSHAP used. Shape: {np.array(shap_values).shape}, "
          f"{shap_time_per:.3f}s/sample")
except Exception as e:
    print(f"  TreeSHAP failed ({e}), using KernelSHAP on 50-sample background...")
    bg = shap.kmeans(X_train_s, 20)
    explainer_shap = shap.KernelExplainer(xgb_final.predict_proba, bg)
    shap_values = explainer_shap.shap_values(X_shap[:50])
    X_shap = X_shap[:50]; y_shap = y_shap[:50]
    shap_time_per = (time.time() - shap_time_start) / 50
    print(f"  KernelSHAP used. {shap_time_per:.3f}s/sample")

sv_arr = np.array(shap_values)
if isinstance(shap_values, list):
    per_class_shap = shap_values
    sv_for_summary = shap_values[0]
elif sv_arr.ndim == 3:

    if sv_arr.shape[2] == len(class_names):

        per_class_shap = [sv_arr[:, :, i] for i in range(sv_arr.shape[2])]
    else:

        per_class_shap = [sv_arr[i] for i in range(sv_arr.shape[0])]
    sv_for_summary = per_class_shap[0]
else:
    per_class_shap = [sv_arr]
    sv_for_summary = sv_arr

shap_global_arr = np.mean([np.abs(sv).mean(axis=0) for sv in per_class_shap], axis=0)
shap_global_imp = pd.Series(shap_global_arr, index=feature_cols).sort_values(ascending=False)

print("  Top-10 SHAP features:")
print(shap_global_imp.head(10).to_string())

n_classes = len(class_names)
if len(per_class_shap) == n_classes:
    fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 7), sharey=False)
    for i, (sv, cls) in enumerate(zip(per_class_shap, class_names)):
        imp = pd.Series(np.abs(sv).mean(axis=0), index=feature_cols)
        top = imp.nlargest(10)
        axes[i].barh(top.index[::-1], top.values[::-1], color='steelblue')
        axes[i].set_title(f"SHAP - {cls}", fontsize=11)
        axes[i].set_xlabel("Mean |SHAP|")
    plt.suptitle("Figure 7: Per-class SHAP Feature Importance", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_PATH + "figure7_shap_per_class.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure7_shap_per_class.png")

sv_for_summary = per_class_shap[0]
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv_for_summary, X_shap, feature_names=feature_cols,
                  max_display=20, show=False)
plt.title("Figure 8: SHAP Summary Plot", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_PATH + "figure8_shap_summary.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: figure8_shap_summary.png")

print("\n[12] Computing LIME explanations...")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_s,
    feature_names=feature_cols,
    class_names=class_names,
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

lime_sample_size = min(50, len(X_shap))
lime_importances = []

t0 = time.time()
for i in range(lime_sample_size):
    exp = lime_explainer.explain_instance(
        X_shap[i],
        xgb_final.predict_proba,
        num_features=len(feature_cols),
        top_labels=1
    )
    top_label = list(exp.local_exp.keys())[0]
    feat_weights = dict(exp.local_exp[top_label])

    lime_imp = np.zeros(len(feature_cols))
    for feat_idx, weight in feat_weights.items():
        if feat_idx < len(feature_cols):
            lime_imp[feat_idx] = abs(weight)
    lime_importances.append(lime_imp)

lime_time = (time.time() - t0) / lime_sample_size
lime_importances = np.array(lime_importances)
lime_global = pd.Series(lime_importances.mean(axis=0), index=feature_cols).sort_values(ascending=False)

print(f"  LIME avg time per instance: {lime_time:.3f}s")
print("  Top-10 LIME features:")
print(lime_global.head(10).to_string())

fig, ax = plt.subplots(figsize=(9, 7))
top_lime = lime_global.head(15)
ax.barh(top_lime.index[::-1], top_lime.values[::-1], color='coral')
ax.set_title("Figure 9: LIME Global Feature Importance", fontsize=13)
ax.set_xlabel("Mean |LIME weight|")
plt.tight_layout()
fig.savefig(OUT_PATH + "figure9_lime_importance.png", dpi=150)
plt.close(fig)
print("  Saved: figure9_lime_importance.png")

print("\n[13] Computing Hybrid SHAP-LIME scores...")

shap_local = np.mean([np.abs(sv[:lime_sample_size]) for sv in per_class_shap], axis=0)

def normalise_rows(arr):
    mins = arr.min(axis=1, keepdims=True)
    maxs = arr.max(axis=1, keepdims=True)
    denom = np.where(maxs - mins == 0, 1, maxs - mins)
    return (arr - mins) / denom

shap_norm = normalise_rows(shap_local)
lime_norm = normalise_rows(lime_importances)

alpha = 0.5
hybrid_scores = alpha * shap_norm + (1 - alpha) * lime_norm

hybrid_global = pd.Series(hybrid_scores.mean(axis=0), index=feature_cols).sort_values(ascending=False)

print("  Top-10 Hybrid SHAP-LIME features:")
print(hybrid_global.head(10).to_string())

fig, ax = plt.subplots(figsize=(9, 7))
top_hybrid = hybrid_global.head(15)
ax.barh(top_hybrid.index[::-1], top_hybrid.values[::-1], color='mediumpurple')
ax.set_title("Figure 10: Hybrid SHAP+LIME Feature Importance", fontsize=13)
ax.set_xlabel("Mean Hybrid Score")
plt.tight_layout()
fig.savefig(OUT_PATH + "figure10_hybrid_importance.png", dpi=150)
plt.close(fig)
print("  Saved: figure10_hybrid_importance.png")

print("\n[14] Explainability validation (fidelity, consistency, timing)...")

k_ablate = 5

def ablation_fidelity(X_sample, y_sample, shap_imp, lime_imp, hybrid_imp, model, k=5):
    baseline = np.median(X_train_s, axis=0)
    results = {'SHAP': [], 'LIME': [], 'Hybrid': []}

    for i in range(len(X_sample)):
        orig_prob = model.predict_proba(X_sample[i:i+1])[0, y_sample[i]]

        for method, imp in [('SHAP', shap_imp[i]), ('LIME', lime_imp[i]), ('Hybrid', hybrid_imp[i])]:
            top_k_idx = np.argsort(imp)[::-1][:k]
            x_ablated = X_sample[i].copy()
            x_ablated[top_k_idx] = baseline[top_k_idx]
            abl_prob = model.predict_proba(x_ablated.reshape(1, -1))[0, y_sample[i]]
            results[method].append(orig_prob - abl_prob)

    return {m: np.mean(v) for m, v in results.items()}

fidelity = ablation_fidelity(
    X_shap[:lime_sample_size], y_shap[:lime_sample_size],
    shap_norm, lime_norm, hybrid_scores,
    xgb_final, k=k_ablate
)

print(f"  Table 4: Fidelity (mean prob drop, k={k_ablate})")
for m, v in fidelity.items():
    print(f"    {m}: {v:.4f}")
pd.DataFrame([fidelity]).to_csv(OUT_PATH + "table4_fidelity.csv", index=False)

rhos = []
for i in range(lime_sample_size):
    rho, _ = spearmanr(shap_norm[i], lime_norm[i])
    if not np.isnan(rho):
        rhos.append(rho)
print(f"\n  Consistency (Spearman rho SHAP vs LIME): mean={np.mean(rhos):.4f}, "
      f"std={np.std(rhos):.4f}")

t0 = time.time()
_ = explainer_shap.shap_values(X_shap[:10])
shap_time = (time.time() - t0) / 10

hybrid_time = shap_time + lime_time

print(f"\n  Table 5: Timing (avg per instance)")
print(f"    SHAP:   {shap_time:.3f}s")
print(f"    LIME:   {lime_time:.3f}s")
print(f"    Hybrid: {hybrid_time:.3f}s")
pd.DataFrame([{'SHAP': shap_time, 'LIME': lime_time, 'Hybrid': hybrid_time}
              ]).to_csv(OUT_PATH + "table5_timing.csv", index=False)

print("\n[15] Pedagogical construct mapping...")

ped_groups = {
    'Assessment Performance':
        ['weighted_score', 'mean_score', 'perf_trend', 'last_submission', 'num_assessments'],
    'Engagement & Activity':
        ['total_clicks', 'avg_clicks', 'max_clicks', 'activity_entropy',
         'days_active', 'distinct_resources', 'cv_clicks', 'click_std'],
    'Temporal Behaviour':
        ['TPI', 'study_duration', 'last_activity_day', 'burstiness',
         'first_access', 'first_registration'],
    'Composite Indicators':
        ['CBII', 'engagement_efficiency', 'dropout_risk'],
    'Demographic':
        ['gender', 'region', 'highest_education', 'imd_band', 'age_band',
         'disability', 'num_of_prev_attempts', 'studied_credits',
         'code_module', 'code_presentation'],
}

ped_contrib = {}
for group, cols in ped_groups.items():
    valid = [c for c in cols if c in hybrid_global.index]
    ped_contrib[group] = hybrid_global[valid].sum() if valid else 0.0

ped_series = pd.Series(ped_contrib).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2']
ax.barh(ped_series.index, ped_series.values, color=colors)
ax.set_title("Figure 11: Pedagogical Construct Contributions (Hybrid SHAP+LIME)", fontsize=13)
ax.set_xlabel("Cumulative Hybrid Score")
plt.tight_layout()
fig.savefig(OUT_PATH + "figure11_pedagogical_grouping.png", dpi=150)
plt.close(fig)
print("  Saved: figure11_pedagogical_grouping.png")
print("  Pedagogical contributions:")
print(ped_series.round(4).to_string())

print("\n[16] Building HTBT (Hybrid Temporal-Behavioural Transformer)...")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("  PyTorch not available. Installing...")

if not TORCH_AVAILABLE:
    import subprocess
    subprocess.run(['pip', 'install', 'torch', '--quiet'], check=False)
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        TORCH_AVAILABLE = True
        print("  PyTorch installed successfully.")
    except ImportError:
        print("  PyTorch unavailable - skipping HTBT.")

if TORCH_AVAILABLE:

    print("  Building 30-day VLE click sequences...")
    WINDOW = 30

    sv = student_vle.copy()
    sv['date'] = pd.to_numeric(sv['date'], errors='coerce')
    sv['sum_click'] = pd.to_numeric(sv['sum_click'], errors='coerce').fillna(0)

    sv_end = (sv.groupby(['code_module','code_presentation','id_student'])
               ['date'].max().reset_index().rename(columns={'date':'end_date'}))
    sv = sv.merge(sv_end, on=['code_module','code_presentation','id_student'])
    sv = sv.dropna(subset=['date','end_date'])
    sv['days_from_end'] = sv['end_date'] - sv['date']
    sv_window = sv[sv['days_from_end'] < WINDOW]

    sv_pivot = (sv_window
                .groupby(['code_module','code_presentation','id_student','days_from_end'])
                ['sum_click'].sum().reset_index())

    def build_seq(group):
        seq = np.zeros(WINDOW, dtype=np.float32)
        for _, row in group.iterrows():
            idx = int(row['days_from_end'])
            if 0 <= idx < WINDOW:
                seq[WINDOW - 1 - idx] = row['sum_click']
        return seq

    print("  Aggregating sequences (may take a moment)...")
    seq_df = (sv_pivot
              .groupby(['code_module','code_presentation','id_student'])
              .apply(build_seq)
              .reset_index()
              .rename(columns={0: 'sequence'}))

    df_htbt = df.merge(seq_df, on=['code_module','code_presentation','id_student'], how='left')
    df_htbt['sequence'] = df_htbt['sequence'].apply(
        lambda x: x if isinstance(x, np.ndarray) else np.zeros(WINDOW, dtype=np.float32))

    max_click = max(arr.max() for arr in df_htbt['sequence'] if arr.max() > 0) or 1.0
    seqs = np.stack(df_htbt['sequence'].values) / max_click

    static_X = scaler.transform(df[feature_cols].fillna(0).values)
    labels_htbt = df['final_result_enc'].values

    idx_all = np.arange(len(labels_htbt))
    idx_temp, idx_test_h = train_test_split(
        idx_all, test_size=0.20, random_state=42, stratify=labels_htbt)
    idx_train_h, idx_val_h = train_test_split(
        idx_temp, test_size=0.20, random_state=42, stratify=labels_htbt[idx_temp])

    def to_tensors(idx):
        seq_t    = torch.tensor(seqs[idx], dtype=torch.float32).unsqueeze(-1)
        static_t = torch.tensor(static_X[idx], dtype=torch.float32)
        labels_t = torch.tensor(labels_htbt[idx], dtype=torch.long)
        return seq_t, static_t, labels_t

    seq_tr, st_tr, lb_tr = to_tensors(idx_train_h)
    seq_vl, st_vl, lb_vl = to_tensors(idx_val_h)
    seq_te, st_te, lb_te = to_tensors(idx_test_h)

    train_ds = TensorDataset(seq_tr, st_tr, lb_tr)
    val_ds   = TensorDataset(seq_vl, st_vl, lb_vl)
    test_ds  = TensorDataset(seq_te, st_te, lb_te)

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=256)
    test_dl  = DataLoader(test_ds,  batch_size=256)

    class HTBTModel(nn.Module):
        def __init__(self, seq_len, n_static, n_classes,
                     d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
            super().__init__()

            self.temporal_proj = nn.Linear(1, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.static_mlp = nn.Sequential(
                nn.Linear(n_static, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, d_model),  nn.ReLU()
            )

            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead,
                dropout=dropout, batch_first=True)

            self.classifier = nn.Sequential(
                nn.Linear(d_model * 2, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, n_classes)
            )

            self.align_proj = nn.Linear(seq_len * d_model, d_model)
            self.seq_len = seq_len
            self.d_model = d_model

        def forward(self, seq, static):

            t = self.temporal_proj(seq)
            t_enc = self.transformer(t)

            s = self.static_mlp(static)
            s_q = s.unsqueeze(1)

            attn_out, attn_weights = self.cross_attn(s_q, t_enc, t_enc)
            attn_out = attn_out.squeeze(1)

            t_pool = t_enc.mean(dim=1)
            align_loss = nn.functional.mse_loss(t_pool, s)

            combined = torch.cat([attn_out, s], dim=1)
            logits = self.classifier(combined)

            return logits, attn_weights, align_loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    n_static = static_X.shape[1]
    n_classes = len(class_names)
    model_htbt = HTBTModel(
        seq_len=WINDOW, n_static=n_static, n_classes=n_classes,
        d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1
    ).to(device)

    optimizer = optim.Adam(model_htbt.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print("  Training HTBT...")
    best_val_acc = 0
    patience_count = 0
    MAX_EPOCHS = 30
    PATIENCE = 7
    ALIGN_WEIGHT = 0.1

    train_losses, val_accs = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        model_htbt.train()
        epoch_loss = 0
        for seq_b, st_b, lb_b in train_dl:
            seq_b, st_b, lb_b = seq_b.to(device), st_b.to(device), lb_b.to(device)
            optimizer.zero_grad()
            logits, _, align_loss = model_htbt(seq_b, st_b)
            loss = criterion(logits, lb_b) + ALIGN_WEIGHT * align_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model_htbt.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model_htbt.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for seq_b, st_b, lb_b in val_dl:
                seq_b, st_b = seq_b.to(device), st_b.to(device)
                logits, _, _ = model_htbt(seq_b, st_b)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_true.extend(lb_b.numpy())

        val_acc = accuracy_score(val_true, val_preds)
        train_losses.append(epoch_loss / len(train_dl))
        val_accs.append(val_acc)
        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model_htbt.state_dict(), OUT_PATH + "htbt_best.pt")
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: loss={epoch_loss/len(train_dl):.4f}, "
                  f"val_acc={val_acc:.4f}")

        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    model_htbt.load_state_dict(torch.load(OUT_PATH + "htbt_best.pt"))
    model_htbt.eval()

    test_preds, test_true, all_attn = [], [], []
    test_probs_htbt = []
    with torch.no_grad():
        for seq_b, st_b, lb_b in test_dl:
            seq_b, st_b = seq_b.to(device), st_b.to(device)
            logits, attn_w, _ = model_htbt(seq_b, st_b)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_true.extend(lb_b.numpy())
            test_probs_htbt.extend(probs.tolist())
            all_attn.append(attn_w.squeeze(1).cpu().numpy())

    htbt_acc  = accuracy_score(test_true, test_preds)
    htbt_f1   = f1_score(test_true, test_preds, average='macro', zero_division=0)
    htbt_prec = precision_score(test_true, test_preds, average='macro', zero_division=0)
    htbt_rec  = recall_score(test_true, test_preds, average='macro', zero_division=0)
    htbt_auc  = roc_auc_score(
        test_true, np.array(test_probs_htbt), multi_class='ovr', average='macro')

    print(f"\n  Table 6: HTBT Test Set Performance")
    print(f"  Accuracy : {htbt_acc:.4f}")
    print(f"  Precision: {htbt_prec:.4f}")
    print(f"  Recall   : {htbt_rec:.4f}")
    print(f"  Macro-F1 : {htbt_f1:.4f}")
    print(f"  ROC-AUC  : {htbt_auc:.4f}")
    pd.DataFrame([{'accuracy': htbt_acc, 'macro_precision': htbt_prec,
                   'macro_recall': htbt_rec, 'macro_f1': htbt_f1,
                   'roc_auc': htbt_auc}]
                 ).to_csv(OUT_PATH + "table6_htbt_test.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(train_losses, label='Train Loss', color='steelblue')
    ax1.set_ylabel("Loss", color='steelblue')
    ax2 = ax1.twinx()
    ax2.plot(val_accs, label='Val Accuracy', color='darkorange')
    ax2.set_ylabel("Val Accuracy", color='darkorange')
    ax1.set_xlabel("Epoch")
    plt.title("Figure 13: HTBT Training Curve", fontsize=13)
    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2, loc='lower right')
    plt.tight_layout()
    fig.savefig(OUT_PATH + "figure13_htbt_training.png", dpi=150)
    plt.close(fig)
    print("  Saved: figure13_htbt_training.png")

    all_attn_arr = np.concatenate(all_attn, axis=0)
    mean_attn    = all_attn_arr.mean(axis=0)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(all_attn_arr)
    cluster_centers = km.cluster_centers_

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    persona_names = ['Early Engagers', 'Late Bloomers', 'Sporadic']
    days = np.arange(WINDOW)

    for ci in range(3):
        mask = cluster_labels == ci
        cluster_attn = all_attn_arr[mask]
        mean_c = cluster_centers[ci]
        std_c  = cluster_attn.std(axis=0)
        axes[ci].plot(days, mean_c, label='Mean', color='navy')
        axes[ci].fill_between(days, mean_c - std_c, mean_c + std_c, alpha=0.2, color='navy')
        axes[ci].set_title(f"Cluster {ci+1}: {persona_names[ci]}\n(n={mask.sum()})", fontsize=10)
        axes[ci].set_xlabel("Day in 30-day window")
        if ci == 0:
            axes[ci].set_ylabel("Attention weight")

    plt.suptitle("Figure 14: Clustered Avg Attention Patterns with Variance (HTBT)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_PATH + "figure14_attention_clusters.png", dpi=150)
    plt.close(fig)
    print("  Saved: figure14_attention_clusters.png")

    attn_insights = {}
    for ci in range(3):
        mask = cluster_labels == ci
        true_lbs = np.array(test_true)[mask]
        dominant = class_names[np.bincount(true_lbs).argmax()]
        peak_day = int(np.argmax(cluster_centers[ci]))
        attn_insights[f"Cluster {ci+1} ({persona_names[ci]})"] = {
            'n_students': int(mask.sum()),
            'dominant_outcome': dominant,
            'peak_attention_day': peak_day,
            'mean_peak_weight': float(cluster_centers[ci].max()),
        }
    attn_df = pd.DataFrame(attn_insights).T
    print("\n  Table 7: Attention-Based Behavioural Insights")
    print(attn_df.to_string())
    attn_df.to_csv(OUT_PATH + "table7_attention_insights.csv")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE - Summary")
print("=" * 70)

print("\n[Table 1] 10-Fold CV Results:")
print(cv_df.round(4).to_string())

print(f"\n[Table 2] XGBoost Test Set: Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")

print(f"\n[Table 4] Fidelity (k={k_ablate}):")
for m, v in fidelity.items():
    print(f"  {m}: {v:.4f}")

print(f"\n[Table 5] Timing:")
print(f"  SHAP:{shap_time:.3f}s | LIME:{lime_time:.3f}s | Hybrid:{hybrid_time:.3f}s")

if TORCH_AVAILABLE:
    print(f"\n[Table 6] HTBT Test Set: Acc={htbt_acc:.4f}, F1={htbt_f1:.4f}, AUC={htbt_auc:.4f}")

print(f"\nAll outputs saved to: {OUT_PATH}")
print("Files:")
for f in sorted(os.listdir(OUT_PATH)):
    print(f"  {f}")
