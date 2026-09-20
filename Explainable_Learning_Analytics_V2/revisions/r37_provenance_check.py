import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PREPROCESSED_CSV, RESULTS_DIR, SEED
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
