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

if __name__ == "__main__":
    print("This module is imported by the main experiment script.")
    print("Usage: from contribution1_adaptive_hybrid import compute_adaptive_hybrid, evaluate_hybrid_methods")
