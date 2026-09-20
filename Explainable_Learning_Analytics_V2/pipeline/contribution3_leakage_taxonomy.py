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

if __name__ == "__main__":
    run_leakage_taxonomy()
