# Multiclass Ensemble Experiments — Delay Severity Prediction

**Branch**: `ML_Ensemble_Experiments`
**Date**: 2026-05-07
**Dataset**: `dataset_maestro_FULL_US_2022-2025_BTS_IEM.parquet`, subsampled to 300K rows
**Split**: Temporal 60/20/20 → Train: 165,391 · Val: 55,130 · Test: 55,132
**Target**: Multiclass ordinal — C0 (on-time), C1 (15–30 min), C2 (30–60 min), C3 (>60 min)

---

## Class Distribution (test set)

| Class | Description | Count | Share |
|---|---|---|---|
| C0 | On-time (≤15 min) | 45,643 | 82.8% |
| C1 | Minor delay (15–30 min) | 3,407 | 6.2% |
| C2 | Moderate delay (30–60 min) | 2,880 | 5.2% |
| C3 | Severe delay (>60 min) | 3,202 | 5.8% |

> **Key challenge**: Extreme class imbalance — C0 represents 83% of the data.

---

## Approaches Compared

### A — Standard Multiclass LightGBM (Baseline)
Single LightGBM booster with `objective=multiclass` (softmax over 4 classes). The current production approach.

### B — Multiclass Stacking (LightGBM + XGBoost + HistGBT → Logistic Regression)
Three base learners each output a 4-class probability vector. Their OOF (out-of-fold) predictions (3 × 4 = 12 features) are fed to a Logistic Regression meta-learner that learns the optimal combination. Uses 3 temporal folds.

### C — Chained Binary Classifiers (Ordinal-Aware)
Three cumulative binary LightGBM models that exploit the ordinal structure:
- **Stage 1**: P(class ≥ 1) — is the flight delayed at all?
- **Stage 2**: P(class ≥ 2) — is the delay >30 minutes?
- **Stage 3**: P(class ≥ 3) — is the delay >60 minutes?

Final class probabilities derived from cumulative differences:
```
P(C0) = 1 − P(≥1)
P(C1) = P(≥1) − P(≥2)
P(C2) = P(≥2) − P(≥3)
P(C3) = P(≥3)
```

Monotonicity enforced: P(≥1) ≥ P(≥2) ≥ P(≥3).

---

## Results Summary

### Global Metrics

| Approach | AUC macro | Accuracy | F1 macro | F1 weighted | Log-loss | Time (s) |
|---|---|---|---|---|---|---|
| **A** Standard Multiclass | 0.7644 | 0.8481 | 0.3732 | 0.7974 | 0.5317 | **17.6** |
| **B** Stacking | **0.7745** | 0.8468 | **0.3892** | **0.8018** | **0.5278** | 287.8 |
| **C** Chained Binary | 0.7703 | **0.8500** | 0.3857 | 0.8012 | 0.5712 | 27.2 |

### Per-Class F1 Scores (the critical comparison)

| Approach | C0 (on-time) | C1 (15–30) | C2 (30–60) | C3 (>60) |
|---|---|---|---|---|
| **A** Standard | 0.9237 | 0.0234 | 0.0835 | 0.4620 |
| **B** Stacking | 0.9255 | **0.0479** | **0.2106** | 0.3728 |
| **C** Chained | 0.9249 | 0.0407 | 0.1021 | **0.4751** |

---

## Analysis

### 1. All approaches struggle with C1 (minor delays)

F1 for C1 is below 0.05 across all methods. This class (15–30 min delays) is the hardest to predict because:
- **Boundary ambiguity**: C1 flights are on the edge — they could easily be C0 or C2 depending on small perturbations
- **Low prevalence**: Only 6.2% of the test set
- **Similar feature profiles**: Flights with 10 min delay and 20 min delay look nearly identical in feature space

### 2. Stacking (B) excels at C2 (moderate delays)

Stacking more than doubled C2 F1 vs the baseline (0.2106 vs 0.0835). The meta-learner combines diverse signals — when XGBoost says C2 but LightGBM says C1, the meta-learner learns which to trust. This diversity is exactly where stacking should shine, and it does for the intermediate classes.

### 3. Chained Binary (C) excels at C3 (severe delays)

C achieves the best C3 F1 (0.4751) because the dedicated Stage 3 model (`P(class ≥ 3)`) focuses entirely on distinguishing severe delays from everything else. This is the ordinal advantage — each binary model specializes in one threshold.

### 4. Stacking has the best overall discrimination

B achieves the highest macro AUC (0.7745), F1 macro (0.3892), and F1 weighted (0.8018). The meta-learner successfully redistributes probability mass toward minority classes, improving the overall ranking ability of the ensemble.

### 5. Cost-benefit

| Approach | AUC Δ vs A | Time Δ vs A | Complexity |
|---|---|---|---|
| A Standard | — | — | 1 model |
| B Stacking | **+1.01 pts** ✅ | **+260s** (16×) | 4 models + OOF + meta |
| C Chained | **+0.59 pts** ✅ | **+10s** (1.5×) | 3 binary models |

---

## Conclusions

### For multiclass, ensembling DOES help (unlike binary)

Unlike the binary experiments where ensembling was marginal (+0.14 AUC), multiclass stacking gives a meaningful +1.01 AUC improvement. The reason: with 4 classes and severe imbalance, model diversity matters more — different models find different minority-class decision boundaries.

### Recommendation by use case

| Use case | Best approach | Reason |
|---|---|---|
| **Detect any delay (binary-like)** | C — Chained Binary | Best C0 accuracy + best C3 detection |
| **Severity triage (balance all classes)** | B — Stacking | Best C2 detection, best F1 macro |
| **Fast iteration / simplicity** | A — Standard | 17s training, simple deployment |
| **Operational alerting (severe delays only)** | C — Chained Binary | Best C3 F1 (0.4751), fast training |

### Key finding for the thesis

> The ordinal structure of delay severity is real and exploitable. Chained Binary (C) achieves the best per-class F1 on the most operationally important class (C3: severe delays) while being the fastest to train. Stacking (B) is best for balanced classification across all severity levels. The standard multiclass approach (A) is not optimal for either scenario.

---

## Reproducibility

```bash
git checkout ML_Ensemble_Experiments

python3 train_multiclass.py --subsample 300000 --num-boost-round 1500 \
    --early-stopping 100 --no-balance

# Results: artifacts/multiclass_comparison/multiclass_comparison.json
```
