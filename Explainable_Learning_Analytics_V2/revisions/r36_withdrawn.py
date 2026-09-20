import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
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
