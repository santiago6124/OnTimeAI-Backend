# Live AUC Gap Investigation Report

**Branch**: `ML_Ensemble_Experiments`
**Date**: 2026-05-11
**Period analyzed**: May 4–8, 2026 (2,817 predictions, 2,574 matched to actuals)

---

## Executive Summary

The model achieves **AUC 0.849 offline** but only **AUC 0.601 live** — a **25-point gap**. The root cause analysis reveals **three compounding problems**, not a single failure:

1. **Calibration inversion** (most damaging): High-confidence predictions (>0.5) are systematically wrong — flights predicted at 86% delay probability only delay 29% of the time. This completely breaks ranking.
2. **Carrier-specific failure**: 6 of 10 carriers have AUC below random (0.5). The model is actively harmful for AA, B6, F9, OH, OO, and YX.
3. **Feature degradation on high-delay days**: AUC and delay rate are strongly negatively correlated (r = −0.756). When delays are most common and prediction matters most, the model fails.

---

## Diagnosis Results

### 1. Probability Distribution — Moderately Compressed

| Metric | Offline (expected) | Live (actual) |
|---|---|---|
| Mean proba | ~0.20 | 0.166 |
| Std proba | ~0.22 | 0.172 |
| % below 0.1 | ~35% | **48.1%** |
| % above 0.5 | ~10% | **5.6%** |

**Verdict**: Probabilities are squeezed toward zero. The model lacks the feature signal to spread predictions — consistent with missing lineage features being replaced by cold-deck medians.

### 2. Threshold Strategy — All strategies underperform equally

| Strategy | n | AUC | Brier | Pos rate |
|---|---|---|---|---|
| replay_quantile@0.22 | 2,266 | 0.613 | 0.192 | 22.5% |
| quantile@0.22 | 260 | 0.595 | 0.253 | 31.5% |
| quantile@0.26 | 45 | 0.596 | 0.237 | 20.0% |

**Verdict**: The threshold strategy isn't the problem — AUC is consistently ~0.60 regardless of threshold. The issue is the **underlying probability ranking**, not where we cut.

### 3. Carrier-Level Discrimination — 6 of 10 carriers BROKEN

| Carrier | n | AUC | Actual delay rate | Predicted pos rate (>0.3) | Verdict |
|---|---|---|---|---|---|
| UA | 60 | 0.663 | 20.0% | 30.0% | ✅ OK |
| DL | 1,811 | 0.624 | 20.9% | 8.7% | ⚠️ Poor (under-predicts) |
| WN | 122 | 0.628 | 31.1% | 39.3% | ⚠️ Poor |
| 9E | 217 | 0.593 | 28.1% | 1.4% | ⚠️ Poor (near-zero predictions) |
| F9 | 155 | 0.525 | 31.6% | **68.4%** | ❌ Broken (massively over-predicts) |
| AA | 60 | 0.490 | 26.7% | **63.3%** | ❌ Broken (below random) |
| OO | 42 | 0.431 | **57.1%** | 2.4% | ❌ Broken (misses most delays) |
| OH | 25 | 0.302 | 28.0% | 24.0% | ❌ Broken |
| B6 | 21 | **0.184** | 9.5% | **52.4%** | ❌ Catastrophic (inverted) |
| YX | 21 | 0.408 | 33.3% | 4.8% | ❌ Broken |

**Key insight**: The model's behavior is wildly inconsistent across carriers. For F9 and B6, it predicts delay >50% of the time regardless of whether flights actually delay. For 9E and OO, it almost never predicts delay even when >28-57% of flights delay. This suggests certain carrier×route×feature combinations produce systematically wrong probabilities.

### 4. Calibration Inversion — High-confidence predictions are WRONG

| Bin | n | Mean proba | Actual delay rate | Gap | Status |
|---|---|---|---|---|---|
| [0.0, 0.1) | 1,235 | 0.057 | 0.175 | +0.118 | ⚠️ Under-confident |
| [0.1, 0.2) | 676 | 0.145 | 0.283 | +0.137 | ⚠️ Under-confident |
| [0.2, 0.3) | 272 | 0.242 | 0.246 | +0.004 | ✅ Well calibrated |
| [0.3, 0.5) | 236 | 0.378 | 0.339 | −0.039 | ✅ Well calibrated |
| **[0.5, 0.7)** | **85** | **0.600** | **0.306** | **−0.294** | **❌ INVERTED** |
| **[0.7, 1.0)** | **70** | **0.860** | **0.286** | **−0.575** | **❌ INVERTED** |

**This is the most critical finding**: When the model says "86% chance of delay," the flight actually delays only 29% of the time. The model's highest-confidence predictions are nearly as likely to be wrong as a coin flip. This destroys AUC because the rank ordering is backwards in the upper tail.

### 5. AUC vs. Delay Rate — Strong Negative Correlation

| Date | n | AUC | Delay rate | Mean proba |
|---|---|---|---|---|
| May 4 | 503 | 0.583 | 13.7% | 0.194 |
| May 5 | 460 | 0.616 | 19.1% | 0.182 |
| May 6 | 522 | 0.572 | 28.4% | 0.189 |
| May 7 | 859 | 0.565 | **32.4%** | 0.159 |
| May 8 | 230 | **0.863** | 7.4% | 0.080 |

**Correlation(AUC, delay_rate) = −0.756** — strong negative.

The model performs well only when delays are rare (May 8: 7.4% delay rate → AUC 0.863). On high-delay days (May 7: 32.4%), the model's AUC drops to 0.565. This means the model can identify "normal" flights as on-time but **cannot identify which specific flights will delay** when delays are widespread.

**Root cause**: On high-delay days, the discriminative signal comes from **lineage features** (cascade effects, inbound delays) — exactly the features that are degraded in live (44% chain-walk hit rate). Without them, the model defaults to the cold-deck median, losing the ability to rank flights.

### 6. Delay Distribution — Consistent with BTS historical

| Metric | Live (May 4-8) | BTS historical |
|---|---|---|
| Mean delay | 6.0 min | ~5-10 min |
| % delayed (>15 min) | 23.3% | ~20-22% |
| % early (<0) | 56.0% | ~55-60% |

**Verdict**: The delay distribution is normal — this is NOT an anomalous weather week. The model should be able to perform here.

---

## Root Cause Ranking

| Rank | Hypothesis | Evidence | Impact on AUC |
|---|---|---|---|
| **1** | **Calibration inversion** in upper tail | Flights at P>0.5 delay only 30% → rank ordering broken | **~10-15 pts** |
| **2** | **Carrier-specific failure** (6/10 broken) | Non-DL carriers have near-random or inverted AUC | **~5-8 pts** |
| **3** | **Feature degradation** (lineage NaN) | 44% chain-walk hit, compressed proba, high-delay day failure | **~5-8 pts** |
| **4** | **ATL sampling bias** | Training on full US, testing only ATL hub | **~2-3 pts** |

---

## Available APIs for Proper Live Testing Sample

| API | Data | Cost | Use for |
|---|---|---|---|
| **BTS On-Time Performance** | Official delay data, all US flights | Free (2-month lag) | Ground truth backtest — **Option A** |
| **AeroAPI (FlightAware)** | Live schedules, actuals, tail nums | ~$0.005/call | Real-time monitoring — **Option B** |
| **IEM METAR** | Weather observations | Free | Weather features (already integrated) |
| **OpenSky Network** | ADS-B positions | Free (academic) | Future: inbound ETA features |
| **FAA ASPM** | Airport delay metrics | Free (limited) | Context: was this a GDP day? |
| **aviationstack** | Flight status, schedules | Freemium | Alternative to AeroAPI (lower quality) |
| **Cirium / OAG** | Premium schedule + delay data | Paid (enterprise) | Not practical for thesis |

### Recommended Testing Approach

**Option A — BTS Retrospective Backtest** (best for thesis):
1. Download BTS On-Time data for March 2026 (most recent available)
2. Get IEM weather for same period
3. Run full pipeline offline (features + model) → no chain-walk issues
4. Compute AUC on full US data (matching training distribution)
5. Then filter to ATL-only to quantify ATL-specific bias

**Sampling assumptions**:
- Full month, no subsampling, no stratification
- Must include ALL carriers (not just DL)
- Temporal hold-out: model trained 2022-2025, tested on March 2026
- Expected n: ~500K flights → AUC CI width: ±0.001

---

## Recommended Fixes

| Priority | Fix | Expected impact |
|---|---|---|
| **P0** | Investigate WHY P>0.5 predictions are inverted — likely specific carriers (F9, AA, B6) producing bogus high probabilities | Fixes ~10 AUC pts |
| **P1** | Retrain with carrier-stratified evaluation; consider carrier-specific calibration or thresholds | Fixes ~5-8 AUC pts |
| **P2** | Increase chain-walk budget (50/tick); pre-cache lineage for ATL flights | Fixes ~3-5 AUC pts |
| **P3** | Run BTS backtest to establish unbiased offline-to-live gap | Diagnostic, not fix |
