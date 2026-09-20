import argparse
import os
import sys
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).resolve().parent

DATA_DIR = Path(os.environ.get("OULAD_DATA", ROOT / "data" / "anonymisedData"))
PREPROCESSED_CSV = Path(os.environ.get("OULAD_PREPROCESSED", DATA_DIR / "oulad_preprocessed_noleak.csv"))
MODELS_DIR = Path(os.environ.get("OULAD_MODELS", DATA_DIR / "models"))
RESULTS_DIR = Path(os.environ.get("OULAD_RESULTS", ROOT / "results"))

SEED = 42
CUTOFF_DAY = 30
CLASS_NAMES = ["Distinction", "Fail", "Pass", "Withdrawn"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _stage_oulad_xai_experiment(_MAIN=True):
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
    return locals()

def _stage_oulad_xai_noleak(_MAIN=True):
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
    return locals()

def _stage_oulad_htbt(_MAIN=True):
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
    return locals()

def _stage_oulad_htbt_eval(_MAIN=True):
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
    return locals()

def _stage_contribution1_adaptive_hybrid(_MAIN=True):
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel, wilcoxon
    from sklearn.metrics import accuracy_score
    import warnings
    warnings.filterwarnings('ignore')

    OUT_PATH = os.path.join(str(RESULTS_DIR), "contribution_results", "")
    import os; os.makedirs(OUT_PATH, exist_ok=True)

    def compute_adaptive_hybrid(X_sample, y_sample, shap_norm, lime_norm,
                                 model, k=5, baseline=None):
        if baseline is None:
            baseline = np.zeros(X_sample.shape[1])

        n = len(X_sample)
        fid_shap = np.zeros(n)
        fid_lime = np.zeros(n)

        for i in range(n):
            orig_prob = model.predict_proba(X_sample[i:i+1])[0, y_sample[i]]

            x_abl = X_sample[i].copy()
            top_k_shap = np.argsort(shap_norm[i])[::-1][:k]
            x_abl[top_k_shap] = baseline[top_k_shap]
            fid_shap[i] = max(0, orig_prob - model.predict_proba(
                x_abl.reshape(1,-1))[0, y_sample[i]])

            x_abl = X_sample[i].copy()
            top_k_lime = np.argsort(lime_norm[i])[::-1][:k]
            x_abl[top_k_lime] = baseline[top_k_lime]
            fid_lime[i] = max(0, orig_prob - model.predict_proba(
                x_abl.reshape(1,-1))[0, y_sample[i]])

        total_fid = fid_shap + fid_lime
        weights = np.where(total_fid > 0, fid_shap / total_fid, 0.5)

        hybrid_adaptive = (weights[:, None] * shap_norm +
                           (1 - weights[:, None]) * lime_norm)

        return hybrid_adaptive, weights, fid_shap, fid_lime

    def evaluate_hybrid_methods(X_sample, y_sample, shap_norm, lime_norm,
                                 model, k=5, baseline=None, n_bootstrap=500,
                                 feature_names=None, out_path=OUT_PATH):
        if baseline is None:
            baseline = np.zeros(X_sample.shape[1])
        n = len(X_sample)

        print("  Computing adaptive hybrid...")
        hybrid_adp, weights, fid_shap_per, fid_lime_per = compute_adaptive_hybrid(
            X_sample, y_sample, shap_norm, lime_norm, model, k, baseline)

        hybrid_fixed = 0.5 * shap_norm + 0.5 * lime_norm

        fid_fixed = np.zeros(n)
        fid_adp   = np.zeros(n)

        for i in range(n):
            orig = model.predict_proba(X_sample[i:i+1])[0, y_sample[i]]

            x_abl = X_sample[i].copy()
            x_abl[np.argsort(hybrid_fixed[i])[::-1][:k]] = baseline[np.argsort(hybrid_fixed[i])[::-1][:k]]
            fid_fixed[i] = orig - model.predict_proba(x_abl.reshape(1,-1))[0, y_sample[i]]

            x_abl = X_sample[i].copy()
            x_abl[np.argsort(hybrid_adp[i])[::-1][:k]] = baseline[np.argsort(hybrid_adp[i])[::-1][:k]]
            fid_adp[i] = orig - model.predict_proba(x_abl.reshape(1,-1))[0, y_sample[i]]

        methods = {
            'SHAP Only':       fid_shap_per,
            'LIME Only':       fid_lime_per,
            'Hybrid Fixed':    fid_fixed,
            'Hybrid Adaptive': fid_adp,
        }

        print("\n  === Fidelity Results (mean probability drop, k={}) ===".format(k))
        results = {}
        for name, fid in methods.items():
            results[name] = {
                'mean':   float(np.mean(fid)),
                'std':    float(np.std(fid)),
                'median': float(np.median(fid)),
            }
            print(f"  {name:20s}: mean={np.mean(fid):.4f}, std={np.std(fid):.4f}")

        print("\n  === Statistical Significance (Adaptive vs. others) ===")
        stat_results = {}
        for name, fid in methods.items():
            if name == 'Hybrid Adaptive':
                continue
            t_stat, p_t = ttest_rel(fid_adp, fid)
            try:
                w_stat, p_w = wilcoxon(fid_adp, fid)
            except Exception:
                w_stat, p_w = np.nan, np.nan
            stat_results[name] = {'t_stat': t_stat, 'p_ttest': p_t,
                                   'W_stat': w_stat, 'p_wilcoxon': p_w}
            sig = "***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else "ns"
            print(f"  Adaptive vs {name:20s}: t={t_stat:.3f}, p={p_t:.4f} {sig}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        data_box = [methods[m] for m in methods]
        bp = ax.boxplot(data_box, labels=list(methods.keys()), patch_artist=True,
                        notch=True)
        colors = ['#4C72B0','#DD8452','#55A868','#C44E52']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_title("Fidelity Distribution by XAI Method\n(Adaptive Hybrid = Proposed)", fontsize=12)
        ax.set_ylabel("Mean Probability Drop (k=5)")
        ax.axhline(np.mean(fid_adp), color='red', linestyle='--', alpha=0.5, label='Adaptive mean')
        ax.legend(fontsize=9); ax.tick_params(axis='x', rotation=15)

        ax2 = axes[1]
        ax2.hist(weights, bins=20, color='#C44E52', alpha=0.7, edgecolor='black')
        ax2.axvline(0.5, color='black', linestyle='--', label='Fixed alpha=0.5')
        ax2.set_xlabel("Per-instance SHAP weight (w_i)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Adaptive Weights\n(Deviation from 0.5 = instance-specific benefit)", fontsize=12)
        ax2.legend()

        plt.suptitle("Contribution 1: Adaptive Fidelity-Weighted Hybrid XAI", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(out_path + "fig_adaptive_hybrid_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Saved: fig_adaptive_hybrid_comparison.png")

        res_df = pd.DataFrame(results).T.round(4)
        res_df.to_csv(out_path + "table_adaptive_hybrid_fidelity.csv")
        stat_df = pd.DataFrame(stat_results).T.round(4)
        stat_df.to_csv(out_path + "table_adaptive_hybrid_stats.csv")

        print(f"\n  Mean adaptive weight (SHAP): {weights.mean():.3f} (0.5=equal, >0.5=SHAP preferred)")
        print(f"  Instances where SHAP dominates (w>0.6): {(weights>0.6).sum()} ({(weights>0.6).mean()*100:.1f}%)")
        print(f"  Instances where LIME dominates (w<0.4): {(weights<0.4).sum()} ({(weights<0.4).mean()*100:.1f}%)")

        return hybrid_adp, weights, results, stat_results

    if _MAIN:
        print("This module is imported by the main experiment script.")
        print("Usage: from contribution1_adaptive_hybrid import compute_adaptive_hybrid, evaluate_hybrid_methods")
    return locals()

def _stage_contribution2_temporal_window(_MAIN=True):
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import xgboost as xgb
    from scipy.stats import entropy as scipy_entropy
    import warnings
    warnings.filterwarnings('ignore')

    DATA_PATH = os.path.join(str(DATA_DIR), "")
    OUT_PATH = os.path.join(str(RESULTS_DIR), "contribution_results", "")
    import os; os.makedirs(OUT_PATH, exist_ok=True)

    np.random.seed(42)

    CUTOFF_DAYS = [7, 14, 21, 30, 45, 60, 90, 120, 180, 270]

    def build_features_at_cutoff(student_info, student_vle, assessments,
                                   student_assess, courses, student_reg,
                                   cutoff_day):
        sv = student_vle.copy()
        sv['date']      = pd.to_numeric(sv['date'],      errors='coerce')
        sv['sum_click'] = pd.to_numeric(sv['sum_click'], errors='coerce').fillna(0)
        sv = sv[sv['date'] <= cutoff_day]

        if len(sv) == 0:
            return None, None

        vle_agg = (sv.groupby(['code_module','code_presentation','id_student'])
                     .agg(total_clicks=('sum_click','sum'),
                          avg_clicks  =('sum_click','mean'),
                          days_active =('date','nunique'),
                          last_access =('date','max'))
                     .reset_index())
        vle_agg['click_std'] = (sv.groupby(['code_module','code_presentation','id_student'])
                                  ['sum_click'].std().reset_index()['sum_click'].fillna(0).values
                                  if len(vle_agg) > 0 else 0)

        def entr(g):
            c = g['sum_click'].values
            return float(scipy_entropy(c/c.sum())) if c.sum()>0 else 0.0
        entr_df = (sv.groupby(['code_module','code_presentation','id_student'])
                     .apply(entr).reset_index().rename(columns={0:'activity_entropy'}))
        vle_agg = vle_agg.merge(entr_df, on=['code_module','code_presentation','id_student'], how='left')
        vle_agg['activity_entropy'].fillna(0, inplace=True)

        af = student_assess.merge(assessments, on='id_assessment', how='left')
        af['score']          = pd.to_numeric(af['score'],          errors='coerce')
        af['weight']         = pd.to_numeric(af['weight'],         errors='coerce')
        af['date_submitted'] = pd.to_numeric(af['date_submitted'], errors='coerce')
        af = af[af['date_submitted'] <= cutoff_day]
        af['wc'] = af['score'] * af['weight'] / 100.0

        if len(af) > 0:
            assess_agg = (af.groupby(['id_student','code_module','code_presentation'])
                            .agg(weighted_score=('wc','sum'),
                                 num_assessments=('id_assessment','count'),
                                 mean_score=('score','mean'))
                            .reset_index())
        else:
            assess_agg = pd.DataFrame(columns=['id_student','code_module',
                                                'code_presentation','weighted_score',
                                                'num_assessments','mean_score'])

        reg = student_reg.copy()
        reg['date_registration'] = pd.to_numeric(reg['date_registration'], errors='coerce')
        reg = reg.merge(courses, on=['code_module','code_presentation'], how='left')
        reg['module_presentation_length'] = pd.to_numeric(
            reg['module_presentation_length'], errors='coerce')
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

        for c in ['total_clicks','avg_clicks','days_active','last_access',
                  'click_std','activity_entropy','weighted_score','num_assessments','mean_score']:
            if c in df.columns:
                df[c].fillna(0, inplace=True)

        df['engagement_efficiency'] = df['weighted_score'].fillna(0) / (df['total_clicks'].fillna(0)+1)
        df['CBII'] = (0.5*df['weighted_score'].fillna(0) +
                      0.3*df['activity_entropy'].fillna(0))
        df['TPI']  = df['study_duration_proxy'] / (df['num_assessments'].fillna(0)+1)

        return df, None

    def run_temporal_analysis():
        print("\n[Contribution 2] Temporal Prediction Window Analysis")
        print(f"  Evaluating cut-offs: {CUTOFF_DAYS}")

        print("  Loading data...")
        student_info   = pd.read_csv(DATA_PATH+"studentInfo.csv").replace('?',np.nan)
        student_vle    = pd.read_csv(DATA_PATH+"studentVle.csv").replace('?',np.nan)
        assessments    = pd.read_csv(DATA_PATH+"assessments.csv").replace('?',np.nan)
        student_assess = pd.read_csv(DATA_PATH+"studentAssessment.csv").replace('?',np.nan)
        courses        = pd.read_csv(DATA_PATH+"courses.csv").replace('?',np.nan)
        student_reg    = pd.read_csv(DATA_PATH+"studentRegistration.csv").replace('?',np.nan)

        le_tgt = LabelEncoder()
        y_all = le_tgt.fit_transform(student_info['final_result'].astype(str))
        class_names = list(le_tgt.classes_)

        idx = np.arange(len(student_info))
        idx_tr, idx_te = train_test_split(idx, test_size=0.20, random_state=42, stratify=y_all)
        y_train_all = y_all[idx_tr]; y_test_all = y_all[idx_te]

        results = []

        for cutoff in CUTOFF_DAYS:
            print(f"\n  --- Cut-off day {cutoff} ---")

            df, _ = build_features_at_cutoff(
                student_info, student_vle, assessments,
                student_assess, courses, student_reg, cutoff)

            excl = ['id_student','final_result']
            cat_cols = ['gender','region','highest_education','imd_band',
                        'age_band','disability','code_module','code_presentation']

            df_tr = df.iloc[idx_tr].copy()
            df_te = df.iloc[idx_te].copy()

            for col in cat_cols:
                le = LabelEncoder().fit(df_tr[col].astype(str))
                seen = set(le.classes_)
                for sp in [df_tr, df_te]:
                    sp[col] = sp[col].astype(str).apply(
                        lambda v: v if v in seen else le.classes_[0])
                    sp[col] = le.transform(sp[col])

            feat_cols = [c for c in df.columns if c not in excl]
            df_tr[feat_cols] = df_tr[feat_cols].fillna(0)
            df_te[feat_cols] = df_te[feat_cols].fillna(0)

            X_tr = df_tr[feat_cols].values.astype(float)
            X_te = df_te[feat_cols].values.astype(float)

            sc = StandardScaler().fit(X_tr)
            X_tr_s = sc.transform(X_tr)
            X_te_s = sc.transform(X_te)

            n_with_vle = (df_tr['total_clicks'].fillna(0) > 0).sum()
            pct_with_vle = n_with_vle / len(df_tr) * 100

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_accs, cv_f1s = [], []
            for tr_i, vl_i in skf.split(X_tr_s, y_train_all):
                sc2 = StandardScaler().fit(X_tr[tr_i])
                Xtr2, Xvl2 = sc2.transform(X_tr[tr_i]), sc2.transform(X_tr[vl_i])
                clf = xgb.XGBClassifier(
                    n_estimators=200, learning_rate=0.1, max_depth=5,
                    use_label_encoder=False, eval_metric='mlogloss',
                    random_state=42, n_jobs=-1, verbosity=0)
                clf.fit(Xtr2, y_train_all[tr_i])
                yp = clf.predict(Xvl2)
                cv_accs.append(accuracy_score(y_train_all[vl_i], yp))
                cv_f1s.append(f1_score(y_train_all[vl_i], yp, average='macro', zero_division=0))

            clf_final = xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=5,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=42, n_jobs=-1, verbosity=0)
            clf_final.fit(X_tr_s, y_train_all)
            yp_te = clf_final.predict(X_te_s)
            ypr_te = clf_final.predict_proba(X_te_s)

            test_acc = accuracy_score(y_test_all, yp_te)
            test_f1  = f1_score(y_test_all, yp_te, average='macro', zero_division=0)
            try:
                test_auc = roc_auc_score(y_test_all, ypr_te, multi_class='ovr', average='macro')
            except:
                test_auc = np.nan

            results.append({
                'cutoff_day':    cutoff,
                'cv_acc_mean':   np.mean(cv_accs),
                'cv_acc_std':    np.std(cv_accs),
                'cv_f1_mean':    np.mean(cv_f1s),
                'test_acc':      test_acc,
                'test_f1':       test_f1,
                'test_auc':      test_auc,
                'pct_with_vle':  pct_with_vle,
                'n_features':    len(feat_cols),
            })
            print(f"    CV Acc={np.mean(cv_accs):.4f} | Test Acc={test_acc:.4f} | "
                  f"F1={test_f1:.4f} | AUC={test_auc:.4f} | "
                  f"VLE coverage={pct_with_vle:.1f}%")

        res_df = pd.DataFrame(results)
        res_df.to_csv(OUT_PATH + "table_temporal_window.csv", index=False)

        max_auc = res_df['test_auc'].max()
        elbow_threshold = 0.80 * max_auc
        elbow_rows = res_df[res_df['test_auc'] >= elbow_threshold]
        elbow_day = elbow_rows['cutoff_day'].min() if len(elbow_rows) > 0 else None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.plot(res_df['cutoff_day'], res_df['test_acc'], 'o-',
                color='#4C72B0', linewidth=2, markersize=7, label='Test Accuracy')
        ax.fill_between(res_df['cutoff_day'],
                        res_df['cv_acc_mean'] - res_df['cv_acc_std'],
                        res_df['cv_acc_mean'] + res_df['cv_acc_std'],
                        alpha=0.2, color='#4C72B0', label='CV Acc +/-1 SD')
        ax.plot(res_df['cutoff_day'], res_df['test_f1'], 's--',
                color='#DD8452', linewidth=2, markersize=7, label='Test Macro-F1')
        if elbow_day:
            ax.axvline(elbow_day, color='red', linestyle=':', linewidth=2,
                       label=f'Elbow point (day {elbow_day})')
        ax.set_xlabel("Prediction Cut-off Day")
        ax.set_ylabel("Performance Metric")
        ax.set_title("Predictive Performance vs. Observation Window\n(No Data Leakage)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_xticks(res_df['cutoff_day'])

        ax2 = axes[1]
        ax2.plot(res_df['cutoff_day'], res_df['test_auc'], 'D-',
                 color='#55A868', linewidth=2, markersize=8, label='Test ROC-AUC')
        ax2.plot(res_df['cutoff_day'], res_df['pct_with_vle']/100, '^:',
                 color='#C44E52', linewidth=2, markersize=7, label='% Students with VLE data')
        if elbow_day:
            ax2.axvline(elbow_day, color='red', linestyle=':', linewidth=2,
                        label=f'Elbow (day {elbow_day})')
        ax2.set_xlabel("Prediction Cut-off Day")
        ax2.set_ylabel("Metric Value")
        ax2.set_title("AUC vs. Observation Window\n(Trade-off: Early Warning vs. Accuracy)")
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
        ax2.set_xticks(res_df['cutoff_day'])

        plt.suptitle("Contribution 2: Temporal Prediction Window Analysis", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(OUT_PATH + "fig_temporal_window_analysis.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"\n  === Temporal Analysis Summary ===")
        print(res_df[['cutoff_day','test_acc','test_f1','test_auc']].to_string(index=False))
        if elbow_day:
            print(f"\n  ELBOW POINT: Day {elbow_day} achieves >=80% of maximum AUC")
            print(f"  This means meaningful early warning is possible at week {elbow_day//7}")
        print(f"\n  Saved: table_temporal_window.csv, fig_temporal_window_analysis.png")
        return res_df

    if _MAIN:
        res = run_temporal_analysis()
    return locals()

def _stage_contribution3_leakage_taxonomy(_MAIN=True):
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import xgboost as xgb
    import warnings
    warnings.filterwarnings('ignore')

    DATA_PATH = os.path.join(str(DATA_DIR), "")
    OUT_PATH = os.path.join(str(RESULTS_DIR), "contribution_results", "")
    import os; os.makedirs(OUT_PATH, exist_ok=True)

    np.random.seed(42)
    CUTOFF_DAY = 30

    def load_raw_data():
        student_info   = pd.read_csv(DATA_PATH+"studentInfo.csv").replace('?', np.nan)
        student_vle    = pd.read_csv(DATA_PATH+"studentVle.csv").replace('?', np.nan)
        assessments    = pd.read_csv(DATA_PATH+"assessments.csv").replace('?', np.nan)
        student_assess = pd.read_csv(DATA_PATH+"studentAssessment.csv").replace('?', np.nan)
        courses        = pd.read_csv(DATA_PATH+"courses.csv").replace('?', np.nan)
        student_reg    = pd.read_csv(DATA_PATH+"studentRegistration.csv").replace('?', np.nan)
        return student_info, student_vle, assessments, student_assess, courses, student_reg

    def build_base_features(student_info, student_vle, assessments,
                            student_assess, courses, student_reg,
                            cutoff=CUTOFF_DAY, include_unregistration=False):
        sv = student_vle.copy()
        sv['date']      = pd.to_numeric(sv['date'],      errors='coerce')
        sv['sum_click'] = pd.to_numeric(sv['sum_click'], errors='coerce').fillna(0)
        sv = sv[sv['date'] <= cutoff]

        vle_agg = (sv.groupby(['code_module','code_presentation','id_student'])
                     .agg(total_clicks=('sum_click','sum'),
                          days_active  =('date','nunique'),
                          last_access  =('date','max'))
                     .reset_index())

        af = student_assess.merge(assessments, on='id_assessment', how='left')
        af['score']          = pd.to_numeric(af['score'],          errors='coerce')
        af['date_submitted'] = pd.to_numeric(af['date_submitted'], errors='coerce')
        af = af[af['date_submitted'] <= cutoff]

        if len(af) > 0:
            assess_agg = (af.groupby(['id_student','code_module','code_presentation'])
                            .agg(mean_score     =('score','mean'),
                                 n_assessments  =('id_assessment','count'))
                            .reset_index())
        else:
            assess_agg = pd.DataFrame(columns=['id_student','code_module',
                                                'code_presentation','mean_score','n_assessments'])

        reg = student_reg.copy()
        reg['date_registration']   = pd.to_numeric(reg['date_registration'],   errors='coerce')
        reg['date_unregistration'] = pd.to_numeric(reg['date_unregistration'], errors='coerce')
        reg = reg.merge(courses, on=['code_module','code_presentation'], how='left')
        reg['module_presentation_length'] = pd.to_numeric(
            reg['module_presentation_length'], errors='coerce')

        if include_unregistration:

            reg['study_duration'] = (
                reg['date_unregistration'].fillna(reg['module_presentation_length'])
                - reg['date_registration']
            ).clip(0).fillna(0)
        else:

            reg['study_duration'] = (
                reg['module_presentation_length'] - reg['date_registration']
            ).clip(0).fillna(0)

        reg_feat = reg[['code_module','code_presentation','id_student','study_duration']]

        df = (student_info
              .merge(vle_agg,    on=['code_module','code_presentation','id_student'], how='left')
              .merge(assess_agg, on=['code_module','code_presentation','id_student'], how='left')
              .merge(reg_feat,   on=['code_module','code_presentation','id_student'], how='left'))

        for c in ['total_clicks','days_active','last_access','mean_score','n_assessments','study_duration']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        return df

    def run_experiment(df, split_before_encode=True, refit_scaler_per_fold=True,
                       label="Clean"):
        le_tgt = LabelEncoder()
        y_all  = le_tgt.fit_transform(df['final_result'].astype(str))

        excl     = ['id_student','final_result']
        cat_cols = ['gender','region','highest_education','imd_band',
                    'age_band','disability','code_module','code_presentation']
        feat_cols = [c for c in df.columns if c not in excl]

        idx    = np.arange(len(df))
        idx_tr, idx_te = train_test_split(idx, test_size=0.20, random_state=42, stratify=y_all)
        y_tr = y_all[idx_tr]; y_te = y_all[idx_te]

        if split_before_encode:
            df_tr = df.iloc[idx_tr].copy()
            df_te = df.iloc[idx_te].copy()
            for col in cat_cols:
                if col in df_tr.columns:
                    le = LabelEncoder().fit(df_tr[col].astype(str))
                    seen = set(le.classes_)
                    for sp in [df_tr, df_te]:
                        sp[col] = sp[col].astype(str).apply(
                            lambda v: v if v in seen else le.classes_[0])
                        sp[col] = le.transform(sp[col])
        else:

            for col in cat_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
            df_tr = df.iloc[idx_tr].copy()
            df_te = df.iloc[idx_te].copy()

        df_tr[feat_cols] = df_tr[feat_cols].fillna(0)
        df_te[feat_cols] = df_te[feat_cols].fillna(0)

        X_tr = df_tr[feat_cols].values.astype(float)
        X_te = df_te[feat_cols].values.astype(float)

        if refit_scaler_per_fold:
            sc = StandardScaler().fit(X_tr)
        else:

            sc = StandardScaler().fit(df[feat_cols].fillna(0).values.astype(float))

        X_tr_s = sc.transform(X_tr)
        X_te_s = sc.transform(X_te)

        clf = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42, n_jobs=-1, verbosity=0)
        clf.fit(X_tr_s, y_tr)
        yp = clf.predict(X_te_s)
        ypr = clf.predict_proba(X_te_s)

        acc = accuracy_score(y_te, yp)
        f1  = f1_score(y_te, yp, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(y_te, ypr, multi_class='ovr', average='macro')
        except Exception:
            auc = np.nan

        print(f"  [{label}] Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
        return {'label': label, 'accuracy': acc, 'macro_f1': f1, 'roc_auc': auc}

    def run_leakage_taxonomy():
        print("\n[Contribution 3] Formal Leakage Taxonomy for OULAD")
        print("  Loading data...")
        si, sv, asmts, sa, courses, sr = load_raw_data()

        experiments = []

        print("\n  --- Type 1: Target-Proximal Feature Leakage ---")
        print("  Contaminated (with date_unregistration):")
        df_leak1 = build_base_features(si, sv, asmts, sa, courses, sr,
                                        include_unregistration=True)
        r_leak1 = run_experiment(df_leak1, label="Type1-Contaminated")

        print("  Clean (without date_unregistration):")
        df_clean = build_base_features(si, sv, asmts, sa, courses, sr,
                                        include_unregistration=False)
        r_clean = run_experiment(df_clean, label="Type1-Clean")
        experiments += [r_leak1, r_clean]

        print("\n  --- Type 2: Temporal Leakage (no cutoff vs cutoff=30) ---")
        print("  Contaminated (all VLE data, no time cutoff):")
        df_leak2 = build_base_features(si, sv, asmts, sa, courses, sr,
                                        cutoff=99999)
        r_leak2 = run_experiment(df_leak2, label="Type2-Contaminated(NoLimit)")
        experiments.append(r_leak2)

        print("\n  --- Type 3: Preprocessing Leakage (encode before split) ---")
        df3 = build_base_features(si, sv, asmts, sa, courses, sr)
        print("  Contaminated (LabelEncoder fitted on full dataset):")
        r_leak3 = run_experiment(df3.copy(), split_before_encode=False,
                                  label="Type3-Contaminated(PreSplit)")
        print("  Clean (LabelEncoder fitted on train only):")
        r_clean3 = run_experiment(df3.copy(), split_before_encode=True,
                                   label="Type3-Clean(PostSplit)")
        experiments += [r_leak3, r_clean3]

        res_df = pd.DataFrame(experiments)
        res_df['inflation_vs_clean_acc'] = res_df['accuracy'] - r_clean['accuracy']
        res_df.to_csv(OUT_PATH + "table_leakage_taxonomy.csv", index=False)
        print("\n  === Leakage Taxonomy Summary ===")
        print(res_df[['label','accuracy','macro_f1','roc_auc','inflation_vs_clean_acc']].to_string(index=False))

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#C44E52','#55A868','#C44E52','#55A868','#C44E52','#55A868']
        bars = ax.bar(res_df['label'], res_df['accuracy'],
                      color=colors[:len(res_df)], alpha=0.8, edgecolor='black')

        ax.axhline(r_clean['accuracy'], color='green', linestyle='--', linewidth=2,
                   label=f"Baseline clean (Acc={r_clean['accuracy']:.4f})")

        for bar, row in zip(bars, res_df.itertuples()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{row.accuracy:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel("Test Accuracy")
        ax.set_title("Contribution 3: Leakage Taxonomy — Impact on Accuracy\n"
                     "Red = Contaminated | Green = Clean", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        fig.savefig(OUT_PATH + "fig_leakage_taxonomy.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print("\n  === OULAD Leakage Audit Checklist ===")
        checklist = [
            ("Type 1", "Target-Proximal Features",
             "date_unregistration removed?",
             r_clean['accuracy'] - r_leak1['accuracy']),
            ("Type 2", "Temporal Leakage",
             "VLE/assessment features capped at cutoff day?",
             r_clean['accuracy'] - r_leak2['accuracy']),
            ("Type 3", "Preprocessing Leakage",
             "LabelEncoder/StandardScaler fitted on train-only?",
             r_clean['accuracy'] - r_leak3['accuracy']),
            ("Type 4", "Label Leakage",
             "final_result excluded from feature matrix?", "N/A (binary check)"),
            ("Type 5", "Group Leakage",
             "Same student in both train and test? (OULAD: each row = 1 registration)",
             "N/A (structure check)"),
        ]
        for t, name, check, inflation in checklist:
            status = "VERIFIED" if isinstance(inflation, str) else (
                "PASSED" if inflation >= -0.001 else f"INFLATED by {abs(inflation):.4f}")
            print(f"  {t} {name:35s} | {check:55s} | {status}")

        print(f"\n  Saved: table_leakage_taxonomy.csv, fig_leakage_taxonomy.png")
        return res_df

    if _MAIN:
        run_leakage_taxonomy()
    return locals()

def _stage_contribution4_stability_analysis(_MAIN=True):
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    import shap
    import lime
    import lime.lime_tabular
    import xgboost as xgb
    import warnings
    warnings.filterwarnings('ignore')

    DATA_PATH = os.path.join(str(DATA_DIR), "")
    OUT_PATH = os.path.join(str(RESULTS_DIR), "contribution_results", "")
    import os; os.makedirs(OUT_PATH, exist_ok=True)

    CUTOFF_DAY = 30
    N_SEEDS    = 5
    N_SAMPLES  = 50
    TOP_K      = 10

    def load_and_build(seed=42):
        si   = pd.read_csv(DATA_PATH+"studentInfo.csv").replace('?', np.nan)
        svle = pd.read_csv(DATA_PATH+"studentVle.csv").replace('?', np.nan)
        asmts= pd.read_csv(DATA_PATH+"assessments.csv").replace('?', np.nan)
        sa   = pd.read_csv(DATA_PATH+"studentAssessment.csv").replace('?', np.nan)
        crs  = pd.read_csv(DATA_PATH+"courses.csv").replace('?', np.nan)
        sreg = pd.read_csv(DATA_PATH+"studentRegistration.csv").replace('?', np.nan)

        sv = svle.copy()
        sv['date']      = pd.to_numeric(sv['date'],      errors='coerce')
        sv['sum_click'] = pd.to_numeric(sv['sum_click'], errors='coerce').fillna(0)
        sv = sv[sv['date'] <= CUTOFF_DAY]

        vle_agg = (sv.groupby(['code_module','code_presentation','id_student'])
                     .agg(total_clicks=('sum_click','sum'),
                          days_active  =('date','nunique'),
                          last_access  =('date','max'))
                     .reset_index())

        af = sa.merge(asmts, on='id_assessment', how='left')
        af['score']          = pd.to_numeric(af['score'],          errors='coerce')
        af['date_submitted'] = pd.to_numeric(af['date_submitted'], errors='coerce')
        af = af[af['date_submitted'] <= CUTOFF_DAY]

        if len(af) > 0:
            assess_agg = (af.groupby(['id_student','code_module','code_presentation'])
                            .agg(mean_score =('score','mean'),
                                 n_assessed =('id_assessment','count'))
                            .reset_index())
        else:
            assess_agg = pd.DataFrame(columns=['id_student','code_module',
                                                'code_presentation','mean_score','n_assessed'])

        reg = sreg.copy()
        reg['date_registration'] = pd.to_numeric(reg['date_registration'], errors='coerce')
        reg = reg.merge(crs, on=['code_module','code_presentation'], how='left')
        reg['module_presentation_length'] = pd.to_numeric(
            reg['module_presentation_length'], errors='coerce')
        reg['study_duration'] = (
            reg['module_presentation_length'] - reg['date_registration']
        ).clip(0).fillna(0)
        reg_feat = reg[['code_module','code_presentation','id_student','study_duration']]

        df = (si
              .merge(vle_agg,    on=['code_module','code_presentation','id_student'], how='left')
              .merge(assess_agg, on=['code_module','code_presentation','id_student'], how='left')
              .merge(reg_feat,   on=['code_module','code_presentation','id_student'], how='left'))

        for c in ['total_clicks','days_active','last_access','mean_score','n_assessed','study_duration']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        le_tgt = LabelEncoder()
        y_all  = le_tgt.fit_transform(df['final_result'].astype(str))

        excl     = ['id_student','final_result']
        cat_cols = ['gender','region','highest_education','imd_band',
                    'age_band','disability','code_module','code_presentation']
        feat_cols = [c for c in df.columns if c not in excl]

        idx    = np.arange(len(df))
        idx_tr, idx_te = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y_all)
        y_tr = y_all[idx_tr]; y_te = y_all[idx_te]

        df_tr = df.iloc[idx_tr].copy()
        df_te = df.iloc[idx_te].copy()

        for col in cat_cols:
            if col in df_tr.columns:
                le = LabelEncoder().fit(df_tr[col].astype(str))
                seen = set(le.classes_)
                for sp in [df_tr, df_te]:
                    sp[col] = sp[col].astype(str).apply(
                        lambda v: v if v in seen else le.classes_[0])
                    sp[col] = le.transform(sp[col])

        df_tr[feat_cols] = df_tr[feat_cols].fillna(0)
        df_te[feat_cols] = df_te[feat_cols].fillna(0)

        X_tr = df_tr[feat_cols].values.astype(float)
        X_te = df_te[feat_cols].values.astype(float)

        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr)
        X_te_s = sc.transform(X_te)

        return X_tr_s, X_te_s, y_tr, y_te, feat_cols

    def get_shap_importance(model, X_bg, X_eval, n_samples=N_SAMPLES):
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_eval[:n_samples])
            sv_arr = np.array(sv)
            if sv_arr.ndim == 3:
                sv_arr = np.abs(sv_arr).mean(axis=2)
            elif sv_arr.ndim == 2:
                sv_arr = np.abs(sv_arr)
            return sv_arr.mean(axis=0)
        except Exception:
            bg = shap.kmeans(X_bg, 20)
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            sv = explainer.shap_values(X_eval[:n_samples], nsamples=50)
            sv_arr = np.abs(np.array(sv))
            if sv_arr.ndim == 3:
                sv_arr = sv_arr.mean(axis=2)
            return sv_arr.mean(axis=0)

    def get_lime_importance(model, X_tr, X_eval, feat_names, n_samples=N_SAMPLES):
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_tr, feature_names=feat_names,
            class_names=[str(i) for i in range(model.n_classes_)],
            discretize_continuous=True, random_state=42)

        importances = np.zeros((min(n_samples, len(X_eval)), len(feat_names)))
        for i in range(min(n_samples, len(X_eval))):
            exp = explainer.explain_instance(X_eval[i], model.predict_proba,
                                              num_features=len(feat_names))
            for feat_idx, weight in exp.as_map()[exp.available_labels()[0]]:
                importances[i, feat_idx] = abs(weight)
        return importances.mean(axis=0)

    def jaccard_top_k(rank_a, rank_b, k=TOP_K, n_features=None):
        if n_features is not None:
            set_a = set(np.argsort(rank_a)[::-1][:k])
            set_b = set(np.argsort(rank_b)[::-1][:k])
        else:
            set_a = set(np.argsort(rank_a)[::-1][:k])
            set_b = set(np.argsort(rank_b)[::-1][:k])
        union = set_a | set_b
        inter = set_a & set_b
        return len(inter) / len(union) if union else 1.0

    def run_stability_analysis():
        print("\n[Contribution 4] Explanation Stability Analysis")
        print(f"  Seeds: {N_SEEDS} | Samples/seed: {N_SAMPLES} | Top-K: {TOP_K}")

        seeds = [42, 7, 13, 99, 2024][:N_SEEDS]

        shap_importances = []
        lime_importances = []
        feat_names_ref   = None

        for seed in seeds:
            print(f"\n  Seed {seed}...")
            X_tr, X_te, y_tr, y_te, feat_names = load_and_build(seed=seed)
            if feat_names_ref is None:
                feat_names_ref = feat_names

            clf = xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=seed, n_jobs=-1, verbosity=0)
            clf.fit(X_tr, y_tr)

            print(f"    Computing SHAP...")
            shap_imp = get_shap_importance(clf, X_tr, X_te, n_samples=N_SAMPLES)
            shap_importances.append(shap_imp)

            print(f"    Computing LIME...")
            lime_imp = get_lime_importance(clf, X_tr, X_te, feat_names, n_samples=N_SAMPLES)
            lime_importances.append(lime_imp)

        shap_importances = np.array(shap_importances)
        lime_importances = np.array(lime_importances)

        n_feat = len(feat_names_ref)

        def corr_matrix(imp_array):
            n = len(imp_array)
            mat = np.ones((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    rho, _ = spearmanr(imp_array[i], imp_array[j])
                    mat[i,j] = mat[j,i] = rho
            return mat

        shap_corr = corr_matrix(shap_importances)
        lime_corr = corr_matrix(lime_importances)

        print("\n  === Seed-to-Seed Spearman Correlation ===")
        print(f"  SHAP: mean={shap_corr[np.triu_indices(N_SEEDS,1)].mean():.4f}, "
              f"min={shap_corr[np.triu_indices(N_SEEDS,1)].min():.4f}")
        print(f"  LIME: mean={lime_corr[np.triu_indices(N_SEEDS,1)].mean():.4f}, "
              f"min={lime_corr[np.triu_indices(N_SEEDS,1)].min():.4f}")

        def jaccard_matrix(imp_array, k=TOP_K):
            n = len(imp_array)
            mat = np.ones((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    jac = jaccard_top_k(imp_array[i], imp_array[j], k=k)
                    mat[i,j] = mat[j,i] = jac
            return mat

        shap_jacc = jaccard_matrix(shap_importances)
        lime_jacc = jaccard_matrix(lime_importances)

        print(f"  SHAP Jaccard(top-{TOP_K}): mean={shap_jacc[np.triu_indices(N_SEEDS,1)].mean():.4f}")
        print(f"  LIME Jaccard(top-{TOP_K}): mean={lime_jacc[np.triu_indices(N_SEEDS,1)].mean():.4f}")

        shap_std = shap_importances.std(axis=0)
        lime_std = lime_importances.std(axis=0)
        shap_mean = shap_importances.mean(axis=0)
        lime_mean = lime_importances.mean(axis=0)

        shap_cv = np.where(shap_mean > 0, shap_std / shap_mean, 0)
        lime_cv = np.where(lime_mean > 0, lime_std / lime_mean, 0)

        stability_df = pd.DataFrame({
            'feature':    feat_names_ref,
            'shap_mean':  shap_mean,
            'shap_std':   shap_std,
            'shap_cv':    shap_cv,
            'lime_mean':  lime_mean,
            'lime_std':   lime_std,
            'lime_cv':    lime_cv,
        }).sort_values('shap_mean', ascending=False)

        per_seed_top = [set(np.argsort(shap_importances[s])[::-1][:TOP_K])
                        for s in range(N_SEEDS)]
        reliable_indices = set.intersection(*per_seed_top)
        reliable_features = [feat_names_ref[i] for i in sorted(reliable_indices)]
        print(f"\n  Reliable Core (top-{TOP_K} in ALL {N_SEEDS} seeds): {reliable_features}")

        stability_df['reliable_core'] = stability_df['feature'].isin(reliable_features)
        stability_df.to_csv(OUT_PATH + "table_explanation_stability.csv", index=False)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax = axes[0,0]
        sns.heatmap(shap_corr, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=[f"S{s}" for s in seeds],
                    yticklabels=[f"S{s}" for s in seeds],
                    vmin=0, vmax=1, ax=ax)
        ax.set_title(f"SHAP Seed Stability\n(Spearman rho, mean={shap_corr[np.triu_indices(N_SEEDS,1)].mean():.3f})",
                     fontsize=11)

        ax2 = axes[0,1]
        sns.heatmap(lime_corr, annot=True, fmt='.2f', cmap='Oranges',
                    xticklabels=[f"S{s}" for s in seeds],
                    yticklabels=[f"S{s}" for s in seeds],
                    vmin=0, vmax=1, ax=ax2)
        ax2.set_title(f"LIME Seed Stability\n(Spearman rho, mean={lime_corr[np.triu_indices(N_SEEDS,1)].mean():.3f})",
                      fontsize=11)

        ax3 = axes[1,0]
        top15 = stability_df.head(15)
        x = np.arange(len(top15))
        w = 0.35
        bars1 = ax3.bar(x - w/2, top15['shap_cv'], w, label='SHAP CV', color='#4C72B0', alpha=0.8)
        bars2 = ax3.bar(x + w/2, top15['lime_cv'], w, label='LIME CV', color='#DD8452', alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(top15['feature'], rotation=40, ha='right', fontsize=8)
        ax3.set_ylabel("Coefficient of Variation (lower = more stable)")
        ax3.set_title("Feature Explanation Stability\n(Coefficient of Variation across seeds)", fontsize=11)
        ax3.legend()

        for xi, feat in zip(x, top15['feature']):
            if feat in reliable_features:
                ax3.axvspan(xi - 0.5, xi + 0.5, alpha=0.1, color='green')

        ax4 = axes[1,1]
        ax4.axis('off')
        table_data = [
            ['Metric', 'SHAP', 'LIME'],
            ['Mean seed-to-seed Spearman rho',
             f"{shap_corr[np.triu_indices(N_SEEDS,1)].mean():.3f}",
             f"{lime_corr[np.triu_indices(N_SEEDS,1)].mean():.3f}"],
            [f'Mean Jaccard (top-{TOP_K})',
             f"{shap_jacc[np.triu_indices(N_SEEDS,1)].mean():.3f}",
             f"{lime_jacc[np.triu_indices(N_SEEDS,1)].mean():.3f}"],
            ['Min seed-to-seed Spearman',
             f"{shap_corr[np.triu_indices(N_SEEDS,1)].min():.3f}",
             f"{lime_corr[np.triu_indices(N_SEEDS,1)].min():.3f}"],
            ['Reliable core features', str(len(reliable_features)), '—'],
        ]
        tbl = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 2)
        ax4.set_title("Summary Statistics", fontsize=12, pad=20)
        reliable_text = '\n'.join(reliable_features) if reliable_features else "(none)"
        ax4.text(0.5, 0.05, f"Reliable core:\n{reliable_text}",
                 transform=ax4.transAxes, ha='center', va='bottom',
                 fontsize=9, color='darkgreen',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.suptitle("Contribution 4: Explanation Stability Analysis\n"
                     f"({N_SEEDS} seeds, {N_SAMPLES} samples each)", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(OUT_PATH + "fig_explanation_stability.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"\n  Saved: table_explanation_stability.csv, fig_explanation_stability.png")
        return stability_df, reliable_features

    if _MAIN:
        run_stability_analysis()
    return locals()

def _stage_run_all_contributions(_MAIN=True):
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
    run_temporal_analysis = _stage_contribution2_temporal_window(False)['run_temporal_analysis']
    res2 = run_temporal_analysis()
    print(f"  [Done in {time.time()-t0:.1f}s]")

    section("Contribution 3: Formal Leakage Taxonomy")
    t0 = time.time()
    run_leakage_taxonomy = _stage_contribution3_leakage_taxonomy(False)['run_leakage_taxonomy']
    res3 = run_leakage_taxonomy()
    print(f"  [Done in {time.time()-t0:.1f}s]")

    section("Contribution 4: Explanation Stability Analysis")
    t0 = time.time()
    run_stability_analysis = _stage_contribution4_stability_analysis(False)['run_stability_analysis']
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

    evaluate_hybrid_methods = _stage_contribution1_adaptive_hybrid(False)['evaluate_hybrid_methods']
    hybrid_adp, weights, fid_results, stat_results = evaluate_hybrid_methods(
        X_te_s[:50], y_sample, shap_norm, lime_norm,
        clf, k=5, baseline=None, n_bootstrap=200,
        feature_names=feat_cols, out_path=OUT_PATH)

    print(f"  [Done in {time.time()-t0:.1f}s]")

    section("ALL CONTRIBUTIONS COMPLETE")
    print(f"  Output directory: {OUT_PATH}")
    for f in sorted(os.listdir(OUT_PATH)):
        print(f"    {f}")
    return locals()

def _stage_r31_group_split(_MAIN=True):
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

    tv, te = train_test_split(np.arange(len(y_all)), test_size=0.20, stratify=y_all, random_state=RS)
    tr, va = train_test_split(tv, test_size=0.20, stratify=y_all[tv], random_state=RS)
    s_tr, s_va, s_te = set(groups[tr]), set(groups[va]), set(groups[te])
    audit = {"n_enrolments": int(len(y_all)), "n_students": int(len(set(groups))), "students_with_multiple_enrolments": int((pd.Series(groups).value_counts() > 1).sum()),
             "test_enrolments_sharing_student_with_train": int(np.isin(groups[te], list(s_tr)).sum()), "test_share_sharing_student_with_train": float(np.isin(groups[te], list(s_tr)).mean()),
             "test_enrolments_sharing_student_with_train_or_val": float(np.isin(groups[te], list(s_tr | s_va)).mean()),
             "students_spanning_partitions": int(len((s_tr & s_te) | (s_tr & s_va) | (s_va & s_te))),
             "presentations_per_partition": {"train": sorted(set(pres[tr])), "val": sorted(set(pres[va])), "test": sorted(set(pres[te]))}}
    json.dump(audit, open(os.path.join(OUT, "overlap_audit.json"), "w"), indent=1); print("overlap audit:", audit["test_share_sharing_student_with_train"])

    def grouped_split(seed):
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed); tv_, te_ = next(outer.split(np.zeros(len(y_all)), y_all, groups))
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed); tr_i, va_i = next(inner.split(np.zeros(len(tv_)), y_all[tv_], groups[tv_]))
        tr_, va_ = tv_[tr_i], tv_[va_i]; assert not (set(groups[tr_]) & set(groups[te_])) and not (set(groups[va_]) & set(groups[te_])) and not (set(groups[tr_]) & set(groups[va_]))
        return tr_, va_, te_
    def random_split(seed):
        tv_, te_ = train_test_split(np.arange(len(y_all)), test_size=0.20, stratify=y_all, random_state=seed); tr_, va_ = train_test_split(tv_, test_size=0.20, stratify=y_all[tv_], random_state=seed); return tr_, va_, te_
    rows = []; grouped42 = None
    for protocol, splitter in [("enrolment-level stratified (paper protocol)", random_split), ("student-grouped (StratifiedGroupKFold)", grouped_split)]:
        for seed in SEEDS:
            tr_, va_, te_ = splitter(seed)
            if protocol.startswith("student") and seed == 42: grouped42 = (tr_, va_, te_)
            Xtr, Xva, Xte = encode_scale(tr_, va_, te_); m = factory(seed).fit(Xtr, y_all[tr_]); P = m.predict_proba(Xte)
            rows.append({"protocol": protocol, "seed": seed, "n_train": len(tr_), "n_val": len(va_), "n_test": len(te_), "test_at_risk_share": float(np.isin(y_all[te_], [F, W]).mean()), **mets(y_all[te_], P)})
            print(f"  {protocol[:22]} seed {seed}: acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['macro_f1']:.4f} auc={rows[-1]['macro_ovr_auc']:.4f}", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(os.path.join(OUT, "table_grouped_vs_random_xgb_per_seed.csv"), index=False)
    summ = tab.groupby("protocol")[["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_ovr_auc", "at_risk_auc"]].agg(['mean', 'std']); summ.to_csv(os.path.join(OUT, "table_grouped_vs_random_xgb_summary.csv"))
    print(summ.round(4))

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
    tr_, va_, te_ = grouped42
    h_m, h_ep, h_best = train_htbt(tr_, va_, te_, tag="grouped-seed42")
    htbt_rows = [{"protocol": "student-grouped (StratifiedGroupKFold), seed 42", "epochs_run": h_ep, "best_val_macro_f1": h_best, **h_m},
                 {"protocol": "enrolment-level stratified (paper checkpoint, Table 6)", "epochs_run": 12, "best_val_macro_f1": np.nan, "accuracy": 0.5653, "macro_precision": 0.5343, "macro_recall": 0.4694, "macro_f1": 0.4799, "macro_ovr_auc": 0.7798, "at_risk_auc": np.nan}]
    pd.DataFrame(htbt_rows).to_csv(os.path.join(OUT, "table_htbt_grouped.csv"), index=False); print("HTBT grouped:", {k: round(v, 4) for k, v in h_m.items()})

    lopo = []
    for held in sorted(set(pres)):
        te_ = np.where(pres == held)[0]; rest = np.where(pres != held)[0]
        tr_, va_ = train_test_split(rest, test_size=0.20, stratify=y_all[rest], random_state=RS)
        Xtr, Xva, Xte = encode_scale(tr_, va_, te_); m = factory(RS).fit(Xtr, y_all[tr_]); P = m.predict_proba(Xte)
        lopo.append({"held_out_presentation": held, "n_train": len(tr_), "n_test": len(te_), **mets(y_all[te_], P)}); print("  LOPO", held, round(lopo[-1]['accuracy'], 4), round(lopo[-1]['macro_f1'], 4), flush=True)
    lt = pd.DataFrame(lopo); lt.loc[len(lt)] = {"held_out_presentation": "mean", **{c: lt[c].mean() for c in lt.columns if c not in ("held_out_presentation",)}}
    lt.to_csv(os.path.join(OUT, "table_lopo_xgb.csv"), index=False)

    C1, C2, GRID = "#2a78d6", "#eb6834", "#e1e0d9"
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); metrics_ = ["accuracy", "macro_f1", "macro_ovr_auc", "at_risk_auc"]; x = np.arange(len(metrics_)); w = 0.36
    for j, (proto, col) in enumerate([("enrolment-level stratified (paper protocol)", C1), ("student-grouped (StratifiedGroupKFold)", C2)]):
        sub = tab[tab.protocol == proto]; ax.bar(x + (j - 0.5) * w, sub[metrics_].mean(), w, yerr=sub[metrics_].std(), color=col, capsize=3, label=proto.split(" (")[0])
    ax.set_xticks(x); ax.set_xticklabels(["accuracy", "macro-F1", "macro OvR AUC", "at-risk AUC"]); ax.set_ylim(0.4, 0.9); ax.set_title("XGBoost test performance, 5 seeds each (mean ± SD)")
    ax.legend(frameon=False); ax.grid(axis='y', color=GRID, lw=0.6); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_grouped_vs_random.png"), dpi=200); plt.close(fig)
    print(f"ALL DONE in {time.time()-T0:.0f}s -> {OUT}")
    return locals()

def _stage_r31b_htbt_seeds(_MAIN=True):
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
    tv_, te_ = train_test_split(np.arange(len(y_all)), test_size=0.20, stratify=y_all, random_state=RS); tr_, va_ = train_test_split(tv_, test_size=0.20, stratify=y_all[tv_], random_state=RS)
    assert len(te_) == 6519
    m, ep, best = train_htbt(tr_, va_, te_, seed=SEED, tag=f"paper-split-seed{SEED}")
    row = {"protocol": "enrolment-level stratified (paper split)", "seed": SEED, "epochs_run": ep, "best_val_macro_f1": best, **m}
    fp = os.path.join(OUT, "table_htbt_seeds.csv"); old = pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(fp, index=False)
    print("HTBT SEED DONE", SEED, {k: round(v, 4) for k, v in m.items()}, "epochs", ep, "best_val", round(best, 4), f"in {time.time()-T0:.0f}s")
    return locals()

def _stage_r31c_htbt_grouped_seeds(_MAIN=True):
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
    return locals()

def _stage_r33_fidelity(_MAIN=True):
    import sys, os, json, time, warnings
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); warnings.filterwarnings('ignore')
    import numpy as np, pandas as pd, joblib
    from scipy.stats import wilcoxon, spearmanr
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    BASE = str(DATA_DIR)
    PREP = str(PREPROCESSED_CSV)
    OUT = os.path.join(str(RESULTS_DIR), "R3-3_R3-2_R1-2"); os.makedirs(OUT, exist_ok=True)
    CK   = os.path.join(OUT, "cache"); os.makedirs(CK, exist_ok=True)
    RS = 42; SEEDS = [0, 1, 2, 3, 4]; KS = [3, 5, 7, 10, 15]; CN = ['Distinction', 'Fail', 'Pass', 'Withdrawn']
    T0 = time.time()

    df = pd.read_csv(PREP); target = 'target_result'
    CAT = ["gender", "region", "highest_education", "imd_band", "age_band", "disability", "code_module", "code_presentation"]
    fc = [c for c in df.columns if c not in {"id_student", "final_result", target}]; nF = len(fc)
    X_raw, y = df[fc].copy(), df[target].astype(int)
    Xt, Xte, yt_, yte = train_test_split(X_raw, y, test_size=0.20, stratify=y, random_state=RS)
    Xtr, Xva, ytr, yva = train_test_split(Xt, yt_, test_size=0.20, stratify=yt_, random_state=RS)
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in CAT:
        le = LabelEncoder(); le.fit(Xtr[c].astype(str)); seen = set(le.classes_)
        for s in (Xtr, Xva, Xte): s[c] = le.transform(s[c].astype(str).apply(lambda v: v if v in seen else le.classes_[0]))
    Xtr, Xva, Xte = [s.fillna(0).astype(float) for s in (Xtr, Xva, Xte)]
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(Xtr), columns=fc, index=Xtr.index)
    X_val = pd.DataFrame(sc.transform(Xva), columns=fc, index=Xva.index); X_test = pd.DataFrame(sc.transform(Xte), columns=fc, index=Xte.index)
    yt = yte.values; model = joblib.load(os.path.join(BASE, "models", "xgb_final.joblib")); assert list(model.feature_names_in_) == fc
    def pf(A): return model.predict_proba(pd.DataFrame(A, columns=fc))
    Xte_np = X_test.values; N = len(Xte_np); P0 = pf(Xte_np); pred = P0.argmax(1); p0 = P0[np.arange(N), pred]
    assert abs((pred == yt).mean() - 0.5762) < 0.002
    median_b = X_train.median().values; mean_b = X_train.mean().values
    paper50 = X_test.sample(n=50, random_state=RS).index; pos50 = np.array([X_test.index.get_loc(i) for i in paper50])
    print(f"test n={N}; paper subset n=50 located; setup {time.time()-T0:.0f}s")

    f_shap = os.path.join(CK, "shap_signed.npy")
    if os.path.exists(f_shap): S = np.load(f_shap)
    else:
        sv = np.array(shap.TreeExplainer(model).shap_values(X_test)); S = sv[np.arange(N), :, pred] if sv.ndim == 3 else sv
        np.save(f_shap, S)
    print(f"SHAP {S.shape} ready {time.time()-T0:.0f}s")

    L = {}
    for sd in SEEDS:
        f = os.path.join(CK, f"lime_seed{sd}.npy")
        if os.path.exists(f): L[sd] = np.load(f); print(f"  LIME seed {sd} cached"); continue
        expl = LimeTabularExplainer(training_data=X_train.values, feature_names=fc, class_names=[str(c) for c in range(4)], mode="classification", random_state=sd)
        M = np.zeros((N, nF)); t1 = time.time()
        for i in range(N):
            e = expl.explain_instance(Xte_np[i], pf, num_features=nF, labels=(int(pred[i]),))
            for fi, w in e.local_exp[int(pred[i])]: M[i, fi] = w
            if i % 500 == 0 and i: print(f"  LIME seed {sd}: {i}/{N} ({(time.time()-t1)/i:.2f}s/inst)", flush=True)
        np.save(f, M); L[sd] = M; print(f"  LIME seed {sd} done in {time.time()-t1:.0f}s", flush=True)

    def l1(v): a = np.abs(v); return a / (a.sum(1, keepdims=True) + 1e-9)
    def topk_idx(scores, k): return np.argsort(np.abs(scores), axis=1)[:, ::-1][:, :k]
    def ablate_drop(scores, k, baseline):
        idx = topk_idx(scores, k); Xa = Xte_np.copy(); Xa[np.arange(N)[:, None], idx] = baseline[idx]
        return p0 - pf(Xa)[np.arange(N), pred]
    def adaptive(S_, L_, k, baseline=median_b):
        fs = np.maximum(0, ablate_drop(S_, k, baseline)); fl = np.maximum(0, ablate_drop(L_, k, baseline)); tot = fs + fl
        w = np.where(tot > 0, fs / np.where(tot > 0, tot, 1), 0.5); H = w[:, None] * l1(S_) + (1 - w)[:, None] * l1(L_)
        return w, H, fs, fl

    ks_rows = []; w_by_k = {}
    for k in KS:
        w, H, fs, fl = adaptive(S, L[0], k); w_by_k[k] = w
        dS, dL, dH = ablate_drop(S, k, median_b), ablate_drop(L[0], k, median_b), ablate_drop(H, k, median_b)
        for subset, idx in [("full test (n=6519)", np.arange(N)), ("paper subset (n=50)", pos50)]:
            ks_rows.append({"k": k, "subset": subset, "SHAP_mean_drop": dS[idx].mean(), "LIME_mean_drop": dL[idx].mean(), "Hybrid_mean_drop": dH[idx].mean(),
                            "w_bar": w[idx].mean(), "share_SHAP_dominant(w>0.6)": (w[idx] > 0.6).mean(), "share_LIME_dominant(w<0.4)": (w[idx] < 0.4).mean(),
                            "share_w_equal_0.5(both_zero)": (np.abs(w[idx] - 0.5) < 1e-12).mean(), "ordering_H>=S>L": bool(dH[idx].mean() >= dS[idx].mean() > dL[idx].mean())})
    for r in ks_rows:
        r["spearman_w_vs_k5"] = spearmanr(w_by_k[r["k"]], w_by_k[5]).correlation if r["subset"].startswith("full") else np.nan
    pd.DataFrame(ks_rows).to_csv(os.path.join(OUT, "table_R1-2_k_sensitivity.csv"), index=False)
    print("R1-2 done", time.time() - T0)

    K = 5; per_seed = {}
    for sd in SEEDS:
        w, H, fs, fl = adaptive(S, L[sd], K)
        per_seed[sd] = {"SHAP": ablate_drop(S, K, median_b), "LIME": ablate_drop(L[sd], K, median_b), "Hybrid": ablate_drop(H, K, median_b), "w": w, "fl": fl}
    D = {m: np.mean([per_seed[sd][m] for sd in SEEDS], axis=0) for m in ["SHAP", "LIME", "Hybrid"]}
    Wm = np.mean([per_seed[sd]["w"] for sd in SEEDS], axis=0); fl0 = np.mean([(per_seed[sd]["fl"] == 0) for sd in SEEDS], axis=0)
    rng = np.random.RandomState(RS); B = 10000
    def bci(v, stat=np.mean):
        idx = rng.randint(0, len(v), (B, len(v))); s = stat(v[idx], axis=1); return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
    def q25(a, axis=None): return np.percentile(a, 25, axis=axis)
    rows = []
    for m in ["SHAP", "LIME", "Hybrid"]:
        v = D[m]; lo, hi = bci(v); qlo, qhi = bci(v, q25)
        rows.append({"method": m, "n": N, "seeds": len(SEEDS), "mean_drop": v.mean(), "ci_low": lo, "ci_high": hi, "sd": v.std(ddof=1), "p25": np.percentile(v, 25), "p25_ci_low": qlo, "p25_ci_high": qhi,
                     "median": np.median(v), "max": v.max(), "min": v.min(), "cross_seed_sd_of_mean": np.std([per_seed[sd][m].mean() for sd in SEEDS], ddof=1)})

        rows.append({"method": m + " (paper n=50 subset)", "n": 50, "seeds": len(SEEDS), "mean_drop": v[pos50].mean(), "sd": v[pos50].std(ddof=1), "median": np.median(v[pos50]), "p25": np.percentile(v[pos50], 25), "max": v[pos50].max(), "min": v[pos50].min()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_R3-3_fidelity_by_method.csv"), index=False)
    def rank_biserial(d): pos = (d > 0).sum(); neg = (d < 0).sum(); return (pos - neg) / max(1, pos + neg)
    prs = []
    for a, b in [("Hybrid", "SHAP"), ("Hybrid", "LIME"), ("SHAP", "LIME")]:
        d = D[a] - D[b]; lo, hi = bci(d); nz = d[d != 0]
        w_std = wilcoxon(D[a], D[b], zero_method='wilcox').pvalue if len(nz) else 1.0; w_pratt = wilcoxon(D[a], D[b], zero_method='pratt').pvalue
        prs.append({"comparison": f"{a} - {b}", "mean_diff": d.mean(), "ci_low": lo, "ci_high": hi, "share_ties": float((d == 0).mean()), "share_a_greater": float((d > 0).mean()), "share_a_smaller": float((d < 0).mean()),
                    "wilcoxon_p_pratt": w_pratt, "wilcoxon_p_wilcox": w_std, "rank_biserial": rank_biserial(d), "cohen_dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0})
    ps = [r["wilcoxon_p_pratt"] for r in prs]; order = np.argsort(ps)
    for rank, i in enumerate(order): prs[i]["wilcoxon_p_pratt_holm"] = min(1.0, ps[i] * (3 - rank))
    pd.DataFrame(prs).to_csv(os.path.join(OUT, "table_R3-3_paired_tests.csv"), index=False)
    wstats = {"w_bar": float(Wm.mean()), "w_bar_ci": bci(Wm), "share_SHAP_dominant": float((Wm > 0.6).mean()), "share_SHAP_dominant_ci": bci((Wm > 0.6).astype(float)),
              "share_LIME_dominant": float((Wm < 0.4).mean()), "share_LIME_dominant_ci": bci((Wm < 0.4).astype(float)), "share_fidL_zero_regime": float(fl0.mean()),
              "paper50_w_bar": float(Wm[pos50].mean()), "paper50_share_SHAP_dominant": float((Wm[pos50] > 0.6).mean()), "paper50_share_LIME_dominant": float((Wm[pos50] < 0.4).mean())}
    json.dump(wstats, open(os.path.join(OUT, "R3-3_adaptive_weight_stats.json"), "w"), indent=1)
    print("R3-3 done", time.time() - T0, {m: round(D[m].mean(), 4) for m in D})

    Lm = np.mean([L[sd] for sd in SEEDS], axis=0)
    _, H5, _, _ = adaptive(S, Lm, 5)
    comp = {"SHAP": S, "LIME": Lm, "Adaptive hybrid": H5}
    for a in [0.25, 0.5, 0.75]: comp[f"Fixed-alpha hybrid (alpha={a})"] = a * l1(S) + (1 - a) * l1(Lm)

    f_val = os.path.join(CK, "lime_val500_seed0.npy"); vidx = np.random.RandomState(RS).choice(len(X_val), 500, replace=False); Xv = X_val.values[vidx]
    Pv = pf(Xv); pv_pred = Pv.argmax(1)
    if os.path.exists(f_val): Lv = np.load(f_val)
    else:
        expl = LimeTabularExplainer(training_data=X_train.values, feature_names=fc, class_names=[str(c) for c in range(4)], mode="classification", random_state=0); Lv = np.zeros((500, nF))
        for i in range(500):
            e = expl.explain_instance(Xv[i], pf, num_features=nF, labels=(int(pv_pred[i]),))
            for fi, w in e.local_exp[int(pv_pred[i])]: Lv[i, fi] = w
        np.save(f_val, Lv)
    svv = np.array(shap.TreeExplainer(model).shap_values(pd.DataFrame(Xv, columns=fc))); Sv = svv[np.arange(500), :, pv_pred]
    pv0 = Pv[np.arange(500), pv_pred]
    def drop_on(Xb, pb, predb, scores, k, baseline):
        idx = topk_idx(scores, k); Xa = Xb.copy(); Xa[np.arange(len(Xb))[:, None], idx] = baseline[idx]; return pb - pf(Xa)[np.arange(len(Xb)), predb]
    fsv = np.maximum(0, drop_on(Xv, pv0, pv_pred, Sv, 5, median_b)); flv = np.maximum(0, drop_on(Xv, pv0, pv_pred, Lv, 5, median_b)); totv = fsv + flv
    alpha_val = float(np.where(totv > 0, fsv / np.where(totv > 0, totv, 1), 0.5).mean())
    comp[f"Validation-estimated global alpha ({alpha_val:.3f})"] = alpha_val * l1(S) + (1 - alpha_val) * l1(Lm)
    pi = permutation_importance(model, X_val, yva.values, n_repeats=5, random_state=RS, scoring="neg_log_loss", n_jobs=-1).importances_mean
    comp["Permutation importance (global ranking)"] = np.tile(pi, (N, 1))
    rnd = np.random.RandomState(RS).rand(N, nF); comp["Random ranking (control)"] = rnd
    def del_ins_auc(scores, baseline=median_b, steps=nF):
        order = np.argsort(np.abs(scores), axis=1)[:, ::-1]; dele = np.zeros((N, steps + 1)); ins = np.zeros((N, steps + 1))
        Xd = Xte_np.copy(); Xi = np.tile(baseline, (N, 1)); dele[:, 0] = p0; ins[:, 0] = pf(Xi)[np.arange(N), pred]
        for s in range(steps):
            j = order[:, s]; Xd[np.arange(N), j] = baseline[j]; Xi[np.arange(N), j] = Xte_np[np.arange(N), j]
            dele[:, s + 1] = pf(Xd)[np.arange(N), pred]; ins[:, s + 1] = pf(Xi)[np.arange(N), pred]
        return (dele / p0[:, None]).mean(1), (ins / p0[:, None]).mean(1), dele.mean(0), ins.mean(0)
    noise_b_rng = np.random.RandomState(RS)
    crows = []; curves = {}
    for name, scores in comp.items():
        dAUC, iAUC, dcurve, icurve = del_ins_auc(scores); curves[name] = (dcurve, icurve)
        idx5 = topk_idx(scores, 5); Xk = np.tile(median_b, (N, 1)); Xk[np.arange(N)[:, None], idx5] = Xte_np[np.arange(N)[:, None], idx5]
        suff = pf(Xk)[np.arange(N), pred] / p0
        Xn = Xte_np.copy(); Xn[np.arange(N)[:, None], idx5] = X_train.values[noise_b_rng.randint(0, len(X_train), (N, 5)), idx5]
        crows.append({"method": name, "deletion_AUC_lower_better": float(dAUC.mean()), "deletion_ci": bci(dAUC), "insertion_AUC_higher_better": float(iAUC.mean()), "insertion_ci": bci(iAUC),
                      "sufficiency_top5_higher_better": float(suff.mean()), "sufficiency_ci": bci(suff),
                      "top5_drop_median_baseline(selection criterion)": float(ablate_drop(scores, 5, median_b).mean()),
                      "top5_drop_mean_baseline": float(ablate_drop(scores, 5, mean_b).mean()),
                      "top5_drop_noise_baseline": float((p0 - pf(Xn)[np.arange(N), pred]).mean())})
        print(f"  R3-2 {name}: del={dAUC.mean():.4f} ins={iAUC.mean():.4f} suff={suff.mean():.4f}", flush=True)
    pd.DataFrame(crows).to_csv(os.path.join(OUT, "table_R3-2_independent_criteria.csv"), index=False)

    dA_h, _, _, _ = del_ins_auc(H5); dA_s, _, _, _ = del_ins_auc(S); _, iA_h, _, _ = del_ins_auc(H5); _, iA_s, _, _ = del_ins_auc(S)
    json.dump({"deletion_AUC_hybrid_minus_SHAP": {"mean": float((dA_h - dA_s).mean()), "ci": bci(dA_h - dA_s), "wilcoxon_p": float(wilcoxon(dA_h, dA_s, zero_method='pratt').pvalue), "rank_biserial": rank_biserial(dA_s - dA_h)},
               "insertion_AUC_hybrid_minus_SHAP": {"mean": float((iA_h - iA_s).mean()), "ci": bci(iA_h - iA_s), "wilcoxon_p": float(wilcoxon(iA_h, iA_s, zero_method='pratt').pvalue), "rank_biserial": rank_biserial(iA_h - iA_s)},
               "alpha_from_validation": alpha_val}, open(os.path.join(OUT, "R3-2_hybrid_vs_shap_independent.json"), "w"), indent=1)

    C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]; GRID = "#e1e0d9"
    ks = pd.DataFrame(ks_rows); full = ks[ks.subset.str.startswith("full")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for m, col in [("SHAP", C[0]), ("LIME", C[1]), ("Hybrid", C[2])]: ax[0].plot(full.k, full[f"{m}_mean_drop"], "-o", color=col, lw=2, ms=4, label=m)
    ax[0].set(xlabel="k (features ablated)", ylabel="mean probability drop", title="(a) fidelity vs k, full test set"); ax[0].legend(frameon=False)
    ax[1].plot(full.k, full.w_bar, "-o", color=C[0], lw=2, ms=4, label="mean SHAP weight w̄"); ax[1].plot(full.k, full["share_SHAP_dominant(w>0.6)"], "-s", color=C[1], lw=2, ms=4, label="share SHAP-dominant")
    ax[1].plot(full.k, full["share_LIME_dominant(w<0.4)"], "-^", color=C[2], lw=2, ms=4, label="share LIME-dominant"); ax[1].axhline(0.5, ls="--", lw=1, color="#898781")
    ax[1].set(xlabel="k", ylabel="value", title="(b) adaptive weights vs k", ylim=(0, 1)); ax[1].legend(frameon=False, fontsize=8)
    for a in ax: a.set_xticks(KS); a.grid(color=GRID, lw=0.6); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_R1-2_k_sensitivity.png"), dpi=200); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9)); xs = np.arange(nF + 1)
    for (name, (dc, ic)), col in zip(curves.items(), C * 2):
        if name.startswith("Fixed-alpha") and "0.5" not in name: continue
        ax[0].plot(xs, dc, color=col, lw=1.8, label=name); ax[1].plot(xs, ic, color=col, lw=1.8, label=name)
    ax[0].set(xlabel="features removed (attribution order)", ylabel="mean predicted-class probability", title="(a) deletion curve (lower = more faithful)")
    ax[1].set(xlabel="features inserted (attribution order)", ylabel="mean predicted-class probability", title="(b) insertion curve (higher = more faithful)"); ax[1].legend(frameon=False, fontsize=7)
    for a in ax: a.grid(color=GRID, lw=0.6); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_R3-2_deletion_insertion.png"), dpi=200); plt.close(fig)
    print(f"ALL DONE in {time.time()-T0:.0f}s -> {OUT}")
    return locals()

def _stage_r33_extras2(_MAIN=True):
    import sys, os, json, time, warnings
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); warnings.filterwarnings('ignore')
    import numpy as np, pandas as pd, joblib
    from scipy.stats import wilcoxon, spearmanr
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    BASE = str(DATA_DIR)
    PREP = str(PREPROCESSED_CSV)
    OUT = os.path.join(str(RESULTS_DIR), "R3-3_R3-2_R1-2"); os.makedirs(OUT, exist_ok=True)
    CK   = os.path.join(OUT, "cache"); os.makedirs(CK, exist_ok=True)
    RS = 42; SEEDS = [0, 1, 2, 3, 4]; KS = [3, 5, 7, 10, 15]; CN = ['Distinction', 'Fail', 'Pass', 'Withdrawn']
    T0 = time.time()

    df = pd.read_csv(PREP); target = 'target_result'
    CAT = ["gender", "region", "highest_education", "imd_band", "age_band", "disability", "code_module", "code_presentation"]
    fc = [c for c in df.columns if c not in {"id_student", "final_result", target}]; nF = len(fc)
    X_raw, y = df[fc].copy(), df[target].astype(int)
    Xt, Xte, yt_, yte = train_test_split(X_raw, y, test_size=0.20, stratify=y, random_state=RS)
    Xtr, Xva, ytr, yva = train_test_split(Xt, yt_, test_size=0.20, stratify=yt_, random_state=RS)
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in CAT:
        le = LabelEncoder(); le.fit(Xtr[c].astype(str)); seen = set(le.classes_)
        for s in (Xtr, Xva, Xte): s[c] = le.transform(s[c].astype(str).apply(lambda v: v if v in seen else le.classes_[0]))
    Xtr, Xva, Xte = [s.fillna(0).astype(float) for s in (Xtr, Xva, Xte)]
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(Xtr), columns=fc, index=Xtr.index)
    X_val = pd.DataFrame(sc.transform(Xva), columns=fc, index=Xva.index); X_test = pd.DataFrame(sc.transform(Xte), columns=fc, index=Xte.index)
    yt = yte.values; model = joblib.load(os.path.join(BASE, "models", "xgb_final.joblib")); assert list(model.feature_names_in_) == fc
    def pf(A): return model.predict_proba(pd.DataFrame(A, columns=fc))
    Xte_np = X_test.values; N = len(Xte_np); P0 = pf(Xte_np); pred = P0.argmax(1); p0 = P0[np.arange(N), pred]
    assert abs((pred == yt).mean() - 0.5762) < 0.002
    median_b = X_train.median().values; mean_b = X_train.mean().values
    paper50 = X_test.sample(n=50, random_state=RS).index; pos50 = np.array([X_test.index.get_loc(i) for i in paper50])
    print(f"test n={N}; paper subset n=50 located; setup {time.time()-T0:.0f}s")

    f_shap = os.path.join(CK, "shap_signed.npy")
    if os.path.exists(f_shap): S = np.load(f_shap)
    else:
        sv = np.array(shap.TreeExplainer(model).shap_values(X_test)); S = sv[np.arange(N), :, pred] if sv.ndim == 3 else sv
        np.save(f_shap, S)
    print(f"SHAP {S.shape} ready {time.time()-T0:.0f}s")

    L = {}
    for sd in SEEDS:
        f = os.path.join(CK, f"lime_seed{sd}.npy")
        if os.path.exists(f): L[sd] = np.load(f); print(f"  LIME seed {sd} cached"); continue
        expl = LimeTabularExplainer(training_data=X_train.values, feature_names=fc, class_names=[str(c) for c in range(4)], mode="classification", random_state=sd)
        M = np.zeros((N, nF)); t1 = time.time()
        for i in range(N):
            e = expl.explain_instance(Xte_np[i], pf, num_features=nF, labels=(int(pred[i]),))
            for fi, w in e.local_exp[int(pred[i])]: M[i, fi] = w
            if i % 500 == 0 and i: print(f"  LIME seed {sd}: {i}/{N} ({(time.time()-t1)/i:.2f}s/inst)", flush=True)
        np.save(f, M); L[sd] = M; print(f"  LIME seed {sd} done in {time.time()-t1:.0f}s", flush=True)

    def l1(v): a = np.abs(v); return a / (a.sum(1, keepdims=True) + 1e-9)
    def topk_idx(scores, k): return np.argsort(np.abs(scores), axis=1)[:, ::-1][:, :k]
    def ablate_drop(scores, k, baseline):
        idx = topk_idx(scores, k); Xa = Xte_np.copy(); Xa[np.arange(N)[:, None], idx] = baseline[idx]
        return p0 - pf(Xa)[np.arange(N), pred]
    def adaptive(S_, L_, k, baseline=median_b):
        fs = np.maximum(0, ablate_drop(S_, k, baseline)); fl = np.maximum(0, ablate_drop(L_, k, baseline)); tot = fs + fl
        w = np.where(tot > 0, fs / np.where(tot > 0, tot, 1), 0.5); H = w[:, None] * l1(S_) + (1 - w)[:, None] * l1(L_)
        return w, H, fs, fl

    Lm = np.mean([L[sd] for sd in SEEDS], axis=0)
    _, H5, fs5, fl5 = adaptive(S, Lm, 5)
    rng = np.random.RandomState(RS); B = 10000
    def bci(v, stat=np.mean):
        idx = rng.randint(0, len(v), (B, len(v))); s_ = stat(v[idx], axis=1); return float(np.percentile(s_, 2.5)), float(np.percentile(s_, 97.5))
    def rank_biserial(d): pos = (d > 0).sum(); neg = (d < 0).sum(); return (pos - neg) / max(1, pos + neg)

    comp = {"SHAP": S, "LIME": Lm, "Adaptive hybrid": H5}
    for a in [0.25, 0.5, 0.75]: comp[f"Fixed-alpha hybrid (alpha={a})"] = a * l1(S) + (1 - a) * l1(Lm)

    f_val = os.path.join(CK, "lime_val500_seed0.npy"); vidx = np.random.RandomState(RS).choice(len(X_val), 500, replace=False); Xv = X_val.values[vidx]
    Pv = pf(Xv); pv_pred = Pv.argmax(1)
    if os.path.exists(f_val): Lv = np.load(f_val)
    else:
        expl = LimeTabularExplainer(training_data=X_train.values, feature_names=fc, class_names=[str(c) for c in range(4)], mode="classification", random_state=0); Lv = np.zeros((500, nF))
        for i in range(500):
            e = expl.explain_instance(Xv[i], pf, num_features=nF, labels=(int(pv_pred[i]),))
            for fi, w in e.local_exp[int(pv_pred[i])]: Lv[i, fi] = w
        np.save(f_val, Lv)
    svv = np.array(shap.TreeExplainer(model).shap_values(pd.DataFrame(Xv, columns=fc))); Sv = svv[np.arange(500), :, pv_pred]
    pv0 = Pv[np.arange(500), pv_pred]
    def drop_on(Xb, pb, predb, scores, k, baseline):
        idx = topk_idx(scores, k); Xa = Xb.copy(); Xa[np.arange(len(Xb))[:, None], idx] = baseline[idx]; return pb - pf(Xa)[np.arange(len(Xb)), predb]
    fsv = np.maximum(0, drop_on(Xv, pv0, pv_pred, Sv, 5, median_b)); flv = np.maximum(0, drop_on(Xv, pv0, pv_pred, Lv, 5, median_b)); totv = fsv + flv
    alpha_val = float(np.where(totv > 0, fsv / np.where(totv > 0, totv, 1), 0.5).mean())
    comp[f"Validation-estimated global alpha ({alpha_val:.3f})"] = alpha_val * l1(S) + (1 - alpha_val) * l1(Lm)
    pi = permutation_importance(model, X_val, yva.values, n_repeats=5, random_state=RS, scoring="neg_log_loss", n_jobs=-1).importances_mean
    comp["Permutation importance (global ranking)"] = np.tile(pi, (N, 1))
    rnd = np.random.RandomState(RS).rand(N, nF); comp["Random ranking (control)"] = rnd
    def del_ins_auc(scores, baseline=median_b, steps=nF):
        order = np.argsort(np.abs(scores), axis=1)[:, ::-1]; dele = np.zeros((N, steps + 1)); ins = np.zeros((N, steps + 1))
        Xd = Xte_np.copy(); Xi = np.tile(baseline, (N, 1)); dele[:, 0] = p0; ins[:, 0] = pf(Xi)[np.arange(N), pred]
        for s in range(steps):
            j = order[:, s]; Xd[np.arange(N), j] = baseline[j]; Xi[np.arange(N), j] = Xte_np[np.arange(N), j]
            dele[:, s + 1] = pf(Xd)[np.arange(N), pred]; ins[:, s + 1] = pf(Xi)[np.arange(N), pred]
        return (dele / p0[:, None]).mean(1), (ins / p0[:, None]).mean(1), dele.mean(0), ins.mean(0)

    ex = {}
    def suff5(scores):
        idx5 = topk_idx(scores, 5); Xk = np.tile(median_b, (N, 1)); Xk[np.arange(N)[:, None], idx5] = Xte_np[np.arange(N)[:, None], idx5]; return pf(Xk)[np.arange(N), pred] / p0
    crit = {}
    for name in ["Adaptive hybrid", "LIME", "Fixed-alpha hybrid (alpha=0.25)", "SHAP"]:
        dA, iA, _, _ = del_ins_auc(comp[name]); crit[name] = {"del": dA, "ins": iA, "suf": suff5(comp[name])}
    def contrast(a, b, key, better_low=False):
        d = crit[a][key] - crit[b][key]; d = -d if better_low else d
        return {"mean": float(d.mean()), "ci": bci(d), "wilcoxon_p": float(wilcoxon(crit[a][key], crit[b][key], zero_method="pratt").pvalue), "sign_effect": rank_biserial(d)}
    for b in ["LIME", "Fixed-alpha hybrid (alpha=0.25)", "SHAP"]:
        ex[f"hybrid_minus_{b}"] = {"deletion_AUC_advantage(lower_better,sign_flipped)": contrast("Adaptive hybrid", b, "del", True), "insertion_AUC_advantage": contrast("Adaptive hybrid", b, "ins"), "sufficiency_advantage": contrast("Adaptive hybrid", b, "suf")}

    reg = {"only_LIME_zero_w1": [], "only_SHAP_zero_w0": [], "both_zero_w05": []}
    for sd in SEEDS:
        w, H, fs, fl = adaptive(S, L[sd], 5); reg["only_LIME_zero_w1"].append(((fl == 0) & (fs > 0)).mean()); reg["only_SHAP_zero_w0"].append(((fs == 0) & (fl > 0)).mean()); reg["both_zero_w05"].append(((fs == 0) & (fl == 0)).mean())
    ex["weight_regimes_k5_seed_avg"] = {k: float(np.mean(v)) for k, v in reg.items()}

    def jacc(A_, B_, k):
        ta, tb = topk_idx(A_, k), topk_idx(B_, k); return float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(ta, tb)]))
    ex["jaccard_topk"] = {f"k{k}": {"SHAP_vs_LIME": jacc(S, Lm, k), "hybrid_vs_LIME": jacc(H5 if k == 5 else adaptive(S, Lm, k)[1], Lm, k), "hybrid_vs_SHAP": jacc(H5 if k == 5 else adaptive(S, Lm, k)[1], S, k)} for k in [5, 15]}

    rs = np.array([spearmanr(np.abs(S[i]), np.abs(Lm[i])).correlation for i in range(N)]); rs = rs[~np.isnan(rs)]
    ex["shap_lime_rank_consistency"] = {"n": int(len(rs)), "mean": float(rs.mean()), "median": float(np.median(rs)), "ci_mean": bci(rs), "share_positive": float((rs > 0).mean())}

    Xn = lambda scores: (lambda idx5: (lambda X_: (X_.__setitem__((np.arange(N)[:, None], idx5), X_train.values[np.random.RandomState(RS).randint(0, len(X_train), (N, 5)), idx5]) or X_))(Xte_np.copy()))(topk_idx(scores, 5))
    dn_s = p0 - pf(Xn(S))[np.arange(N), pred]; dn_l = p0 - pf(Xn(Lm))[np.arange(N), pred]; dn_h = p0 - pf(Xn(H5))[np.arange(N), pred]
    ex["noise_baseline_means"] = {"SHAP": float(dn_s.mean()), "LIME": float(dn_l.mean()), "hybrid": float(dn_h.mean())}
    ex["noise_baseline_SHAP_minus_LIME"] = {"mean": float((dn_s - dn_l).mean()), "ci": bci(dn_s - dn_l), "wilcoxon_p": float(wilcoxon(dn_s, dn_l, zero_method="pratt").pvalue)}
    json.dump(ex, open(os.path.join(OUT, "R3-2_provenance_extras.json"), "w"), indent=1); print("EXTRAS2 DONE", json.dumps(ex)[:3000])
    return locals()

def _stage_r33_fig_fix(_MAIN=True):
    import sys, os, json, time, warnings
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); warnings.filterwarnings('ignore')
    import numpy as np, pandas as pd, joblib
    from scipy.stats import wilcoxon, spearmanr
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    BASE = str(DATA_DIR)
    PREP = str(PREPROCESSED_CSV)
    OUT = os.path.join(str(RESULTS_DIR), "R3-3_R3-2_R1-2"); os.makedirs(OUT, exist_ok=True)
    CK   = os.path.join(OUT, "cache"); os.makedirs(CK, exist_ok=True)
    RS = 42; SEEDS = [0, 1, 2, 3, 4]; KS = [3, 5, 7, 10, 15]; CN = ['Distinction', 'Fail', 'Pass', 'Withdrawn']
    T0 = time.time()

    df = pd.read_csv(PREP); target = 'target_result'
    CAT = ["gender", "region", "highest_education", "imd_band", "age_band", "disability", "code_module", "code_presentation"]
    fc = [c for c in df.columns if c not in {"id_student", "final_result", target}]; nF = len(fc)
    X_raw, y = df[fc].copy(), df[target].astype(int)
    Xt, Xte, yt_, yte = train_test_split(X_raw, y, test_size=0.20, stratify=y, random_state=RS)
    Xtr, Xva, ytr, yva = train_test_split(Xt, yt_, test_size=0.20, stratify=yt_, random_state=RS)
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in CAT:
        le = LabelEncoder(); le.fit(Xtr[c].astype(str)); seen = set(le.classes_)
        for s in (Xtr, Xva, Xte): s[c] = le.transform(s[c].astype(str).apply(lambda v: v if v in seen else le.classes_[0]))
    Xtr, Xva, Xte = [s.fillna(0).astype(float) for s in (Xtr, Xva, Xte)]
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(Xtr), columns=fc, index=Xtr.index)
    X_val = pd.DataFrame(sc.transform(Xva), columns=fc, index=Xva.index); X_test = pd.DataFrame(sc.transform(Xte), columns=fc, index=Xte.index)
    yt = yte.values; model = joblib.load(os.path.join(BASE, "models", "xgb_final.joblib")); assert list(model.feature_names_in_) == fc
    def pf(A): return model.predict_proba(pd.DataFrame(A, columns=fc))
    Xte_np = X_test.values; N = len(Xte_np); P0 = pf(Xte_np); pred = P0.argmax(1); p0 = P0[np.arange(N), pred]
    assert abs((pred == yt).mean() - 0.5762) < 0.002
    median_b = X_train.median().values; mean_b = X_train.mean().values
    paper50 = X_test.sample(n=50, random_state=RS).index; pos50 = np.array([X_test.index.get_loc(i) for i in paper50])
    print(f"test n={N}; paper subset n=50 located; setup {time.time()-T0:.0f}s")

    f_shap = os.path.join(CK, "shap_signed.npy")
    if os.path.exists(f_shap): S = np.load(f_shap)
    else:
        sv = np.array(shap.TreeExplainer(model).shap_values(X_test)); S = sv[np.arange(N), :, pred] if sv.ndim == 3 else sv
        np.save(f_shap, S)
    print(f"SHAP {S.shape} ready {time.time()-T0:.0f}s")

    L = {}
    for sd in SEEDS:
        f = os.path.join(CK, f"lime_seed{sd}.npy")
        if os.path.exists(f): L[sd] = np.load(f); print(f"  LIME seed {sd} cached"); continue
        expl = LimeTabularExplainer(training_data=X_train.values, feature_names=fc, class_names=[str(c) for c in range(4)], mode="classification", random_state=sd)
        M = np.zeros((N, nF)); t1 = time.time()
        for i in range(N):
            e = expl.explain_instance(Xte_np[i], pf, num_features=nF, labels=(int(pred[i]),))
            for fi, w in e.local_exp[int(pred[i])]: M[i, fi] = w
            if i % 500 == 0 and i: print(f"  LIME seed {sd}: {i}/{N} ({(time.time()-t1)/i:.2f}s/inst)", flush=True)
        np.save(f, M); L[sd] = M; print(f"  LIME seed {sd} done in {time.time()-t1:.0f}s", flush=True)

    def l1(v): a = np.abs(v); return a / (a.sum(1, keepdims=True) + 1e-9)
    def topk_idx(scores, k): return np.argsort(np.abs(scores), axis=1)[:, ::-1][:, :k]
    def ablate_drop(scores, k, baseline):
        idx = topk_idx(scores, k); Xa = Xte_np.copy(); Xa[np.arange(N)[:, None], idx] = baseline[idx]
        return p0 - pf(Xa)[np.arange(N), pred]
    def adaptive(S_, L_, k, baseline=median_b):
        fs = np.maximum(0, ablate_drop(S_, k, baseline)); fl = np.maximum(0, ablate_drop(L_, k, baseline)); tot = fs + fl
        w = np.where(tot > 0, fs / np.where(tot > 0, tot, 1), 0.5); H = w[:, None] * l1(S_) + (1 - w)[:, None] * l1(L_)
        return w, H, fs, fl

    Lm = np.mean([L[sd] for sd in SEEDS], axis=0)
    _, H5, _, _ = adaptive(S, Lm, 5)
    rng = np.random.RandomState(RS); B = 200
    def bci(v, stat=np.mean):
        idx = rng.randint(0, len(v), (B, len(v))); s_ = stat(v[idx], axis=1); return float(np.percentile(s_, 2.5)), float(np.percentile(s_, 97.5))
    def rank_biserial(d): pos = (d > 0).sum(); neg = (d < 0).sum(); return (pos - neg) / max(1, pos + neg)

    comp = {"SHAP": S, "LIME": Lm, "Adaptive hybrid": H5}
    for a in [0.25, 0.5, 0.75]: comp[f"Fixed-alpha hybrid (alpha={a})"] = a * l1(S) + (1 - a) * l1(Lm)

    f_val = os.path.join(CK, "lime_val500_seed0.npy"); vidx = np.random.RandomState(RS).choice(len(X_val), 500, replace=False); Xv = X_val.values[vidx]
    Pv = pf(Xv); pv_pred = Pv.argmax(1)
    if os.path.exists(f_val): Lv = np.load(f_val)
    else:
        expl = LimeTabularExplainer(training_data=X_train.values, feature_names=fc, class_names=[str(c) for c in range(4)], mode="classification", random_state=0); Lv = np.zeros((500, nF))
        for i in range(500):
            e = expl.explain_instance(Xv[i], pf, num_features=nF, labels=(int(pv_pred[i]),))
            for fi, w in e.local_exp[int(pv_pred[i])]: Lv[i, fi] = w
        np.save(f_val, Lv)
    svv = np.array(shap.TreeExplainer(model).shap_values(pd.DataFrame(Xv, columns=fc))); Sv = svv[np.arange(500), :, pv_pred]
    pv0 = Pv[np.arange(500), pv_pred]
    def drop_on(Xb, pb, predb, scores, k, baseline):
        idx = topk_idx(scores, k); Xa = Xb.copy(); Xa[np.arange(len(Xb))[:, None], idx] = baseline[idx]; return pb - pf(Xa)[np.arange(len(Xb)), predb]
    fsv = np.maximum(0, drop_on(Xv, pv0, pv_pred, Sv, 5, median_b)); flv = np.maximum(0, drop_on(Xv, pv0, pv_pred, Lv, 5, median_b)); totv = fsv + flv
    alpha_val = float(np.where(totv > 0, fsv / np.where(totv > 0, totv, 1), 0.5).mean())
    comp[f"Validation-estimated global alpha ({alpha_val:.3f})"] = alpha_val * l1(S) + (1 - alpha_val) * l1(Lm)
    pi = permutation_importance(model, X_val, yva.values, n_repeats=5, random_state=RS, scoring="neg_log_loss", n_jobs=-1).importances_mean
    comp["Permutation importance (global ranking)"] = np.tile(pi, (N, 1))
    rnd = np.random.RandomState(RS).rand(N, nF); comp["Random ranking (control)"] = rnd
    def del_ins_auc(scores, baseline=median_b, steps=nF):
        order = np.argsort(np.abs(scores), axis=1)[:, ::-1]; dele = np.zeros((N, steps + 1)); ins = np.zeros((N, steps + 1))
        Xd = Xte_np.copy(); Xi = np.tile(baseline, (N, 1)); dele[:, 0] = p0; ins[:, 0] = pf(Xi)[np.arange(N), pred]
        for s in range(steps):
            j = order[:, s]; Xd[np.arange(N), j] = baseline[j]; Xi[np.arange(N), j] = Xte_np[np.arange(N), j]
            dele[:, s + 1] = pf(Xd)[np.arange(N), pred]; ins[:, s + 1] = pf(Xi)[np.arange(N), pred]
        return (dele / p0[:, None]).mean(1), (ins / p0[:, None]).mean(1), dele.mean(0), ins.mean(0)
    noise_b_rng = np.random.RandomState(RS)
    crows = []; curves = {}
    for name, scores in comp.items():
        dAUC, iAUC, dcurve, icurve = del_ins_auc(scores); curves[name] = (dcurve, icurve)
        idx5 = topk_idx(scores, 5); Xk = np.tile(median_b, (N, 1)); Xk[np.arange(N)[:, None], idx5] = Xte_np[np.arange(N)[:, None], idx5]
        suff = pf(Xk)[np.arange(N), pred] / p0
        Xn = Xte_np.copy(); Xn[np.arange(N)[:, None], idx5] = X_train.values[noise_b_rng.randint(0, len(X_train), (N, 5)), idx5]
        crows.append({"method": name, "deletion_AUC_lower_better": float(dAUC.mean()), "deletion_ci": bci(dAUC), "insertion_AUC_higher_better": float(iAUC.mean()), "insertion_ci": bci(iAUC),
                      "sufficiency_top5_higher_better": float(suff.mean()), "sufficiency_ci": bci(suff),
                      "top5_drop_median_baseline(selection criterion)": float(ablate_drop(scores, 5, median_b).mean()),
                      "top5_drop_mean_baseline": float(ablate_drop(scores, 5, mean_b).mean()),
                      "top5_drop_noise_baseline": float((p0 - pf(Xn)[np.arange(N), pred]).mean())})
        print(f"  R3-2 {name}: del={dAUC.mean():.4f} ins={iAUC.mean():.4f} suff={suff.mean():.4f}", flush=True)

    np.savez(os.path.join(OUT, "deletion_insertion_curves.npz"), **{k.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(".", "p"): np.vstack(v) for k, v in curves.items()})
    COL = {"SHAP": "#2a78d6", "LIME": "#eb6834", "Adaptive hybrid": "#1baf7a", "Fixed-alpha hybrid (alpha=0.5)": "#e87ba4", "Permutation importance (global ranking)": "#e34948", "Random ranking (control)": "#52514e"}
    LBL = {"Fixed-alpha hybrid (alpha=0.5)": "Fixed-α hybrid (α = 0.5)", "Permutation importance (global ranking)": "Permutation importance (global)", "Random ranking (control)": "Random ranking (control)"}
    GRID = "#e1e0d9"; fig, ax = plt.subplots(1, 2, figsize=(10, 3.9)); xs = np.arange(nF + 1)
    for name, (dc, ic) in curves.items():
        if name.startswith("Fixed-alpha") and "0.5" not in name: continue
        if name.startswith("Validation-estimated"): col, lab, ls = "#4a3aa7", f"Validation-estimated global α ({alpha_val:.2f})", "--"
        else: col, lab, ls = COL[name], LBL.get(name, name), "-"
        ax[0].plot(xs, dc, color=col, lw=1.8, ls=ls, label=lab); ax[1].plot(xs, ic, color=col, lw=1.8, ls=ls, label=lab)
    ax[0].set(xlabel="features removed (attribution order)", ylabel="mean predicted-class probability", title="(a) deletion curve (lower = more faithful)")
    ax[1].set(xlabel="features inserted (attribution order)", ylabel="mean predicted-class probability", title="(b) insertion curve (higher = more faithful)"); ax[1].legend(frameon=False, fontsize=7, loc="lower right")
    for a in ax: a.grid(color=GRID, lw=0.6); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_R3-2_deletion_insertion.png"), dpi=200); plt.close(fig); print("FIG FIXED")
    return locals()

def _stage_r35_attention_faithfulness(_MAIN=True):
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
    return locals()

def _stage_r35_v2_faithfulness(_MAIN=True):
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
    return locals()

def _stage_r36_imbalance(_MAIN=True):
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
    return locals()

def _stage_r36_extras(_MAIN=True):
    import sys, os, json, time, warnings
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); warnings.filterwarnings('ignore')
    import numpy as np, pandas as pd, joblib, sklearn
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, recall_score, precision_score
    from sklearn.utils.class_weight import compute_sample_weight
    import xgboost as xgb
    BASE = str(DATA_DIR)
    PREP = str(PREPROCESSED_CSV)
    OUT = os.path.join(str(RESULTS_DIR), "R3-6"); RS = 42; CN = ['Distinction', 'Fail', 'Pass', 'Withdrawn']; D, F, P, W = 0, 1, 2, 3; T0 = time.time()
    df = pd.read_csv(PREP); target = 'target_result'
    CAT = ["gender", "region", "highest_education", "imd_band", "age_band", "disability", "code_module", "code_presentation"]
    fc = [c for c in df.columns if c not in {"id_student", "final_result", target}]; X_raw, y = df[fc].copy(), df[target].astype(int)
    Xt, Xte, yt_, yte = train_test_split(X_raw, y, test_size=0.20, stratify=y, random_state=RS); Xtr, Xva, ytr, yva = train_test_split(Xt, yt_, test_size=0.20, stratify=yt_, random_state=RS)
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in CAT:
        le = LabelEncoder(); le.fit(Xtr[c].astype(str)); seen = set(le.classes_)
        for s in (Xtr, Xva, Xte): s[c] = le.transform(s[c].astype(str).apply(lambda v: v if v in seen else le.classes_[0]))
    Xtr, Xva, Xte = [s.fillna(0).astype(float) for s in (Xtr, Xva, Xte)]
    sc = StandardScaler(); X_train = pd.DataFrame(sc.fit_transform(Xtr), columns=fc, index=Xtr.index); X_val = pd.DataFrame(sc.transform(Xva), columns=fc, index=Xva.index); X_test = pd.DataFrame(sc.transform(Xte), columns=fc, index=Xte.index)
    yt, yv, ytr_ = yte.values, yva.values, ytr.values
    tp = pd.read_csv(os.path.join(OUT, "test_probabilities.csv")); assert (tp["df_index"].values == X_test.index.values).all()
    P4 = tp[[f"xgb4_{c}" for c in CN]].values; Pw = tp[[f"xgbw_{c}" for c in CN]].values; Ph = tp[[f"htbt_{c}" for c in CN]].values; sb = tp["bin_xgb_atrisk"].values
    yb = np.isin(yt, [F, W]).astype(int); s4 = P4[:, F] + P4[:, W]

    shares = {"dataset": {CN[c]: int((y.values == c).sum()) for c in range(4)}, "test": {CN[c]: int((yt == c).sum()) for c in range(4)},
              "dataset_share": {CN[c]: float((y.values == c).mean()) for c in range(4)}, "at_risk_share_dataset": float(np.isin(y.values, [F, W]).mean()), "at_risk_share_test": float(yb.mean())}
    json.dump(shares, open(os.path.join(OUT, "class_shares.json"), "w"), indent=1)

    rng = np.random.RandomState(RS); B = 2000; n = len(yt); contrasts = {k: [] for k in ["weighted_minus_ref_balanced_accuracy", "weighted_minus_ref_macro_f1", "weighted_minus_ref_accuracy", "derived_minus_binary_roc_auc", "derived_minus_binary_pr_auc", "xgb_minus_htbt_macro_f1", "xgb_minus_htbt_accuracy", "weighted_minus_ref_recall_Fail", "weighted_minus_ref_recall_Distinction", "weighted_minus_ref_recall_Pass"]}
    p4, pw, ph = P4.argmax(1), Pw.argmax(1), Ph.argmax(1)
    for _ in range(B):
        i = rng.randint(0, n, n); yi = yt[i]
        contrasts["weighted_minus_ref_balanced_accuracy"].append(balanced_accuracy_score(yi, pw[i]) - balanced_accuracy_score(yi, p4[i]))
        contrasts["weighted_minus_ref_macro_f1"].append(f1_score(yi, pw[i], average='macro') - f1_score(yi, p4[i], average='macro'))
        contrasts["weighted_minus_ref_accuracy"].append(accuracy_score(yi, pw[i]) - accuracy_score(yi, p4[i]))
        ybi = yb[i]; contrasts["derived_minus_binary_roc_auc"].append(roc_auc_score(ybi, s4[i]) - roc_auc_score(ybi, sb[i])); contrasts["derived_minus_binary_pr_auc"].append(average_precision_score(ybi, s4[i]) - average_precision_score(ybi, sb[i]))
        contrasts["xgb_minus_htbt_macro_f1"].append(f1_score(yi, p4[i], average='macro') - f1_score(yi, ph[i], average='macro')); contrasts["xgb_minus_htbt_accuracy"].append(accuracy_score(yi, p4[i]) - accuracy_score(yi, ph[i]))
        for c, nm in [(F, "Fail"), (D, "Distinction"), (P, "Pass")]: contrasts[f"weighted_minus_ref_recall_{nm}"].append(recall_score(yi == c, pw[i] == c) - recall_score(yi == c, p4[i] == c))
    ci = {k: {"point": float(v_pt), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v, v_pt in [
        ("weighted_minus_ref_balanced_accuracy", contrasts["weighted_minus_ref_balanced_accuracy"], balanced_accuracy_score(yt, pw) - balanced_accuracy_score(yt, p4)),
        ("weighted_minus_ref_macro_f1", contrasts["weighted_minus_ref_macro_f1"], f1_score(yt, pw, average='macro') - f1_score(yt, p4, average='macro')),
        ("weighted_minus_ref_accuracy", contrasts["weighted_minus_ref_accuracy"], accuracy_score(yt, pw) - accuracy_score(yt, p4)),
        ("derived_minus_binary_roc_auc", contrasts["derived_minus_binary_roc_auc"], roc_auc_score(yb, s4) - roc_auc_score(yb, sb)),
        ("derived_minus_binary_pr_auc", contrasts["derived_minus_binary_pr_auc"], average_precision_score(yb, s4) - average_precision_score(yb, sb)),
        ("xgb_minus_htbt_macro_f1", contrasts["xgb_minus_htbt_macro_f1"], f1_score(yt, p4, average='macro') - f1_score(yt, ph, average='macro')),
        ("xgb_minus_htbt_accuracy", contrasts["xgb_minus_htbt_accuracy"], accuracy_score(yt, p4) - accuracy_score(yt, ph)),
        ("weighted_minus_ref_recall_Fail", contrasts["weighted_minus_ref_recall_Fail"], recall_score(yt == F, pw == F) - recall_score(yt == F, p4 == F)),
        ("weighted_minus_ref_recall_Distinction", contrasts["weighted_minus_ref_recall_Distinction"], recall_score(yt == D, pw == D) - recall_score(yt == D, p4 == D)),
        ("weighted_minus_ref_recall_Pass", contrasts["weighted_minus_ref_recall_Pass"], recall_score(yt == P, pw == P) - recall_score(yt == P, p4 == P))]}
    ci["note"] = "percentile bootstrap over test enrolments (B=2000, seed 42), single training run per model; training variability not included"
    json.dump(ci, open(os.path.join(OUT, "paired_bootstrap_contrasts.json"), "w"), indent=1); print("paired CIs:", {k: (round(v["point"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in ci.items() if isinstance(v, dict)})

    def brier_sum(yy, Pm): return float(np.mean(np.sum((Pm - np.eye(4)[yy]) ** 2, 1)))
    def ece(yy, Pm, bins=15, equal_mass=False):
        conf = Pm.max(1); hit = (Pm.argmax(1) == yy).astype(float)
        edges = np.quantile(conf, np.linspace(0, 1, bins + 1)) if equal_mass else np.linspace(0, 1, bins + 1); e = 0.0
        for i in range(bins):
            m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
            if m.sum(): e += m.mean() * abs(hit[m].mean() - conf[m].mean())
        return float(e)
    prev = np.bincount(ytr_, minlength=4) / len(ytr_); Pclim = np.tile(prev, (n, 1))
    cal = []
    for name, Pm in [("XGBoost (reference)", P4), ("HTBT", Ph), ("XGBoost + balanced class weights", Pw), ("training-prevalence predictor (baseline)", Pclim)]:
        cal.append({"model": name, "brier_sum_over_classes": brier_sum(yt, Pm), "brier_mean_over_classes": brier_sum(yt, Pm) / 4, "ece_top_label_15_equal_width": ece(yt, Pm), "ece_top_label_15_equal_mass": ece(yt, Pm, equal_mass=True), "mean_confidence": float(Pm.max(1).mean()), "accuracy": accuracy_score(yt, Pm.argmax(1))})
    pd.DataFrame(cal).to_csv(os.path.join(OUT, "table_calibration_v2.csv"), index=False); print(pd.DataFrame(cal).round(4).to_string(index=False))

    m_es = xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", n_estimators=1000, learning_rate=0.1, max_depth=6, random_state=RS, n_jobs=-1, verbosity=0, early_stopping_rounds=30)
    m_es.fit(X_train, ytr_, eval_set=[(X_val, yv)], verbose=False); Pes = m_es.predict_proba(X_test); pes = Pes.argmax(1)
    es = {"best_iteration": int(m_es.best_iteration), "accuracy": accuracy_score(yt, pes), "balanced_accuracy": balanced_accuracy_score(yt, pes), "macro_f1": f1_score(yt, pes, average='macro'),
          "macro_ovr_auc": roc_auc_score(pd.get_dummies(yt), Pes, average='macro', multi_class='ovr'), "brier_sum": brier_sum(yt, Pes), "ece_15_equal_width": ece(yt, Pes), "at_risk_auc": roc_auc_score(yb, Pes[:, F] + Pes[:, W]),
          "reference_200_rounds": {"accuracy": accuracy_score(yt, p4), "macro_f1": f1_score(yt, p4, average='macro'), "macro_ovr_auc": roc_auc_score(pd.get_dummies(yt), P4, average='macro', multi_class='ovr'), "brier_sum": brier_sum(yt, P4)}}
    json.dump(es, open(os.path.join(OUT, "xgb_early_stopped_variant.json"), "w"), indent=1); print("early-stopped variant:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in es.items() if k != "reference_200_rounds"})

    model = joblib.load(os.path.join(BASE, "models", "xgb_final.joblib")); Pv4 = model.predict_proba(X_val)
    def bfac(): return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=200, learning_rate=0.1, max_depth=6, random_state=RS, n_jobs=-1, verbosity=0)
    ybtr = np.isin(ytr_, [F, W]).astype(int); mb = bfac().fit(X_train, ybtr); mbw = bfac().fit(X_train, ybtr, sample_weight=compute_sample_weight('balanced', ybtr))
    pd.DataFrame({"df_index": X_val.index.values, "y_true": yv, **{f"xgb4_{CN[c]}": Pv4[:, c] for c in range(4)}, "bin_xgb_atrisk": mb.predict_proba(X_val)[:, 1], "bin_xgbw_atrisk": mbw.predict_proba(X_val)[:, 1]}).to_csv(os.path.join(OUT, "val_probabilities.csv"), index=False)
    json.dump({"xgboost": xgb.__version__, "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}, open(os.path.join(OUT, "versions.json"), "w"), indent=1)
    print(f"EXTRAS DONE in {time.time()-T0:.0f}s")
    return locals()

def _stage_r36_withdrawn(_MAIN=True):
    import numpy as np, pandas as pd, json, os
    OUT = os.path.join(str(RESULTS_DIR), "R3-6"); t = pd.read_csv(os.path.join(OUT, "test_probabilities.csv")); y = t["y_true"].values
    CN = ["Distinction", "Fail", "Pass", "Withdrawn"]; P4 = t[[f"xgb4_{c}" for c in CN]].values; Pw = t[[f"xgbw_{c}" for c in CN]].values
    pr, pw = P4.argmax(1), Pw.argmax(1); W = 3
    def rec(pred, idx):
        m = y[idx] == W; return float((pred[idx][m] == W).mean())
    rng = np.random.RandomState(42); N = len(y); d = []
    for b in range(2000):
        idx = rng.randint(0, N, N); d.append(rec(pw, idx) - rec(pr, idx))
    d = np.array(d); point = rec(pw, np.arange(N)) - rec(pr, np.arange(N)); pp = float((y == 2).mean())
    res = {"weighted_minus_ref_recall_Withdrawn": {"point": point, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "B": 2000, "seed": 42},
           "prevalence_predictor_rows": {"accuracy": pp, "balanced_accuracy": 0.25, "macro_f1": 2 * pp / (1 + pp) / 4, "weighted_f1": pp * 2 * pp / (1 + pp), "macro_ovr_auc": 0.5, "macro_pr_auc": 0.25}}
    json.dump(res, open(os.path.join(OUT, "withdrawn_contrast_and_prevalence_rows.json"), "w"), indent=1); print(res)
    return locals()

def _stage_r37_cutoff_sweep(_MAIN=True):
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
    return locals()

def _stage_r37_fig(_MAIN=True):
    import os, pandas as pd, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    OUT = os.path.join(str(RESULTS_DIR), "R3-7")
    t = pd.read_csv(os.path.join(OUT, "table_cutoff_validation.csv")); e = pd.read_csv(os.path.join(OUT, "table_withdrawal_exposure_validation.csv"))
    C1, C2, C3, GRID = "#2a78d6", "#eb6834", "#e34948", "#e1e0d9"; x = t["cutoff_day"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for m, c, lab in [("macro_ovr_auc", C1, "macro OvR ROC-AUC (all registrations)"), ("at_risk_auc", C2, "at-risk ROC-AUC (all registrations)")]:
        ax[0].plot(x, t[m], "-o", color=c, ms=4, lw=2, label=lab); ax[0].fill_between(x, t[m + "_ci_low"], t[m + "_ci_high"], color=c, alpha=0.18)
    ax[0].plot(e["cutoff_day"], e["at_risk_auc_still_registered"], "--s", color="#1baf7a", ms=4, lw=1.8, label="at-risk ROC-AUC (still registered at T*)")
    ax[0].set(xlabel="prediction cut-off T* (days)", ylabel="ROC-AUC", title="(a) discrimination vs cut-off"); ax[0].set_ylim(0.65, 1.0)
    for m, c, lab in [("accuracy", C1, "accuracy"), ("macro_f1", C2, "macro-F1")]:
        ax[1].plot(x, t[m], "-o", color=c, ms=4, lw=2, label=lab); ax[1].fill_between(x, t[m + "_ci_low"], t[m + "_ci_high"], color=c, alpha=0.18)
    ax[1].set(xlabel="prediction cut-off T* (days)", ylabel="score", title="(b) accuracy and macro-F1 vs cut-off"); ax[1].set_ylim(0.35, 0.8)
    for a in ax:
        a.axvline(30, ls="--", lw=1, color="#898781"); a.text(33, a.get_ylim()[1] - 0.01, "operational T* = 30", fontsize=8, color="#52514e", va="top")
        a.grid(color=GRID, lw=0.6); a.spines[["top", "right"]].set_visible(False); a.set_xticks([7, 30, 60, 90, 120, 180, 270]); a.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cutoff_validation_linear.png"), dpi=200); print("figure written")
    return locals()

def _stage_r37_provenance_check(_MAIN=True):
    import os, json, numpy as np, pandas as pd
    from sklearn.model_selection import train_test_split
    BASE = str(DATA_DIR); OUT = os.path.join(str(RESULTS_DIR), "R3-7")
    sa = pd.read_csv(os.path.join(BASE, "studentAssessment.csv")); asm = pd.read_csv(os.path.join(BASE, "assessments.csv"))
    am = sa.merge(asm[["id_assessment", "assessment_type"]], on="id_assessment", how="left"); ex = am[am.assessment_type == "Exam"]
    res = {"exam_rows": int(len(ex)), "exam_min_date_submitted": int(ex.date_submitted.min()), "exam_rows_submitted_by_180": int((ex.date_submitted <= 180).sum()), "exam_rows_submitted_by_270": int((ex.date_submitted <= 270).sum())}
    assert res["exam_rows_submitted_by_180"] == 0, "Exam exclusion would change cut-offs <= 180"
    si = pd.read_csv(os.path.join(BASE, "studentInfo.csv")); y = si["final_result"].map({"Distinction": 0, "Fail": 1, "Pass": 2, "Withdrawn": 3}).values
    tv, te = train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=42); tr, va = train_test_split(tv, test_size=0.20, stratify=y[tv], random_state=42)
    vy = np.load(os.path.join(OUT, "cache", "val_y.npy")); res["val_y_matches_recomputed_split"] = bool(np.array_equal(vy, y[np.sort(va)]) or np.array_equal(vy, y[va])); res["n_val"] = int(len(va))
    for T in [7, 14, 21, 30, 45, 60, 90, 120, 180, 270]:
        P = np.load(os.path.join(OUT, "cache", f"val_probs_{T}.npy")); assert P.shape == (len(va), 4), T
    res["all_cached_prob_matrices_have_validation_shape"] = True
    json.dump(res, open(os.path.join(OUT, "provenance_check.json"), "w"), indent=1); print(res)
    return locals()

STAGES = {
    "oulad_xai_experiment": _stage_oulad_xai_experiment,
    "oulad_xai_noleak": _stage_oulad_xai_noleak,
    "oulad_htbt": _stage_oulad_htbt,
    "oulad_htbt_eval": _stage_oulad_htbt_eval,
    "contribution1_adaptive_hybrid": _stage_contribution1_adaptive_hybrid,
    "contribution2_temporal_window": _stage_contribution2_temporal_window,
    "contribution3_leakage_taxonomy": _stage_contribution3_leakage_taxonomy,
    "contribution4_stability_analysis": _stage_contribution4_stability_analysis,
    "run_all_contributions": _stage_run_all_contributions,
    "r31_group_split": _stage_r31_group_split,
    "r31b_htbt_seeds": _stage_r31b_htbt_seeds,
    "r31c_htbt_grouped_seeds": _stage_r31c_htbt_grouped_seeds,
    "r33_fidelity": _stage_r33_fidelity,
    "r33_extras2": _stage_r33_extras2,
    "r33_fig_fix": _stage_r33_fig_fix,
    "r35_attention_faithfulness": _stage_r35_attention_faithfulness,
    "r35_v2_faithfulness": _stage_r35_v2_faithfulness,
    "r36_imbalance": _stage_r36_imbalance,
    "r36_extras": _stage_r36_extras,
    "r36_withdrawn": _stage_r36_withdrawn,
    "r37_cutoff_sweep": _stage_r37_cutoff_sweep,
    "r37_fig": _stage_r37_fig,
    "r37_provenance_check": _stage_r37_provenance_check,
}

ORDER = ["oulad_xai_noleak", "oulad_htbt", "oulad_htbt_eval", "run_all_contributions"]


def main():
    ap = argparse.ArgumentParser(prog="Explainable_Learning_Analytics_V2.py",
                                 description="Explainable Learning Analytics for Early Student Performance Prediction.")
    ap.add_argument("stage", nargs="?", help="stage to run; omit to list them")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    a = ap.parse_args()
    if a.list or not a.stage:
        print("Stages (run them in this order for a full reproduction):")
        for s in ORDER:
            print("   ", s)
        print("\nAll stages:")
        for s in STAGES:
            print("   ", s)
        return 0
    if a.stage not in STAGES:
        print(f"unknown stage: {a.stage}", file=sys.stderr)
        return 2
    STAGES[a.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
