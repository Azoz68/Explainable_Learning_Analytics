import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
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
