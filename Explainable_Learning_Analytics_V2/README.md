# Explainable Learning Analytics for Early Student Performance Prediction

Code for the paper *Explainable Learning Analytics for Early Student Performance Prediction Using
Hybrid Interpretability and Temporal Deep Learning*, submitted to *Discover Artificial Intelligence*.

Everything is in one file, `Explainable_Learning_Analytics_V2.py`: the leakage audit, the feature
engineering, the XGBoost and HTBT models, the adaptive SHAP-LIME hybrid explainer, and every
analysis written in response to the reviewers. Each analysis is a named stage, and you run one
stage at a time.

```
python Explainable_Learning_Analytics_V2.py --list
python Explainable_Learning_Analytics_V2.py oulad_xai_noleak
```

## Data

The study uses the Open University Learning Analytics Dataset (OULAD), which is public and fully
anonymised. It is not redistributed here.

1. Download OULAD from <https://analyse.kmi.open.ac.uk/open_dataset> and unpack the seven CSV files
   (`studentInfo.csv`, `studentVle.csv`, `studentAssessment.csv`, `studentRegistration.csv`,
   `assessments.csv`, `courses.csv`, `vle.csv`) into `data/anonymisedData/`.
2. The `oulad_xai_noleak` stage builds the audited feature table and writes
   `oulad_preprocessed_noleak.csv`. The revision stages read that file.

Paths are not hard-coded. They are resolved at the top of the file relative to it, and each one can
be overridden with an environment variable:

| Variable | Default | Holds |
|---|---|---|
| `OULAD_DATA` | `data/anonymisedData` | the OULAD CSV files |
| `OULAD_PREPROCESSED` | `<OULAD_DATA>/oulad_preprocessed_noleak.csv` | the audited feature table |
| `OULAD_MODELS` | `<OULAD_DATA>/models` | saved models, including `xgb_final.joblib` |
| `OULAD_RESULTS` | `results/` | every table, figure and JSON produced |

## Install

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.12. The results in the paper were produced with the versions pinned in
`requirements.txt`; a CPU-only PyTorch build is sufficient.

## Running

Order matters: the feature table and the saved XGBoost model are built first, and the revision
stages read them.

```
python Explainable_Learning_Analytics_V2.py oulad_xai_noleak         # audited pipeline, features, XGBoost, SHAP and LIME
python Explainable_Learning_Analytics_V2.py oulad_htbt               # trains the HTBT, writes htbt_best.pt
python Explainable_Learning_Analytics_V2.py oulad_htbt_eval          # evaluates the saved checkpoint
python Explainable_Learning_Analytics_V2.py run_all_contributions    # the four contribution analyses
```

The `oulad_xai_experiment` stage is the earlier, unaudited version of the pipeline. It is kept so
that the leakage comparison reported in the paper can be reproduced, and it is not the basis of any
reported result.

## What answers which reviewer comment

Every `r3*` stage is named after the comment it answers.

| Stage | Comment | What it does |
|---|---|---|
| `r31_group_split` | R3-1 | student-level group-aware split, leave-one-presentation-out, overlap audit |
| `r31b_htbt_seeds` | R3-1 | HTBT across training seeds on the paper split |
| `r31c_htbt_grouped_seeds` | R3-1 | HTBT across seeds under the grouped split |
| `r33_fidelity` | R3-3, R3-2, R1-2 | fidelity on the full test partition, bootstrap CIs, paired tests, k-sensitivity, independent criteria, fixed-alpha hybrids |
| `r33_extras2` | R3-2 | provenance checks for the independent-criteria run |
| `r33_fig_fix` | R3-3 | deletion and insertion curves, corrected Figures 10 and 11 |
| `r35_attention_faithfulness` | R3-5 | occlusion tests against attention, first pass |
| `r35_v2_faithfulness` | R3-5 | the reported version: tests restricted to active days, per-cluster correlations |
| `r36_imbalance` | R3-6 | class-imbalance metrics, calibration, binary at-risk detection |
| `r36_extras` | R3-6 | paired-bootstrap contrasts, class-weighted and early-stopped variants |
| `r36_withdrawn` | R3-6 | Withdrawn recall contrast and the prevalence-predictor row |
| `r37_cutoff_sweep` | R3-7 | prediction-window sweep scored on validation, single test evaluation at day 30 |
| `r37_fig` | R3-7 | the cut-off figure |
| `r37_provenance_check` | R3-7 | provenance check for the sweep |

The stages `contribution1_adaptive_hybrid` to `contribution4_stability_analysis` hold the four
contribution analyses: the adaptive hybrid, the temporal window, the leakage taxonomy and
explanation stability. `run_all_contributions` runs all four in order.

## How the single file is organised

Each stage is one function, `_stage_<name>`, whose body is that analysis in full. Giving every
analysis its own function scope keeps the variable names of one analysis out of the way of the
others, so the stages cannot interfere with each other. The `STAGES` dictionary at the foot of the
file maps a stage name to its function, and the command line selects one.

The code carries no comments. The methods section of the paper is the commentary.

## Reproducibility

The seed is 42 throughout, the split is 64/16/20 stratified, and the operational cut-off is day 30.
The XGBoost configuration is fixed rather than tuned: `multi:softprob`, 200 rounds, learning rate
0.1, maximum depth 6. Long runs cache intermediate matrices under
`<OULAD_RESULTS>/<experiment>/cache/` and resume from them.

## Prototype

`prototype/` holds the practitioner-facing dashboard described in Section 9, in English and Arabic.
Each file is standalone: open it in a browser, with no server and no network. It reads nothing and
records nothing, and the four student records it shows are anonymised OULAD test cases.

## Citation

If you use this code, please cite the paper.
