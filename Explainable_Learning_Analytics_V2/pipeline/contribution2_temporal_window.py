import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
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

if __name__ == "__main__":
    res = run_temporal_analysis()
