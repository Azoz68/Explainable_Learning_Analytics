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

if __name__ == "__main__":
    run_stability_analysis()
