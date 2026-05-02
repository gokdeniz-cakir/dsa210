# Machine Learning Methods Report

**Score-Based Popularity Prediction in Classical Music**
DSA 210 — Spring 2025–2026

---

## 1. Overview

This report documents the application of machine learning methods to predict classical music popularity from score-derived features. We test whether the signal identified in our EDA (34/65 significant feature correlations, all modest at |ρ| ≤ 0.31) translates into meaningful predictive power when features are combined.

The analysis proceeds through three stages: continuous prediction (regression), composer identity disentanglement, and discrete prediction (classification). Each stage revealed something the previous one could not.

**Dataset:** 961 pieces, 18 composers, 65 features (25 static + 40 temporal), Spotify popularity labels.

---

## 2. Approach 1: Popularity Similarity Index (PSI)

### Method

We defined a custom baseline metric: the cosine similarity between a piece's standardized feature vector and the centroid of the top 25% most popular pieces, rescaled to [0, 1].

### Results

| Feature Set | Spearman r | Variance Explained |
|---|---|---|
| 14 baseline features (8 tonal + 6 temporal) | 0.28 | ~7.8% |
| 25 features (+ 11 harmonic/structural) | 0.29 | ~8.5% |

The baseline used 8 tonal features and 6 temporal-descriptive features. Adding 11 harmonic/structural features (chord change rate, cadence rate, key changes, etc.) produced only marginal improvement despite nearly doubling complexity. We also tested blending global and composer-specific PSI profiles and multi-attractor models using k-means clustering; neither approach broke through ~9%.

---

## 3. Approach 2: Linear Regression

### Method

OLS and Ridge regression (α = 1, α = 10) with standardized features, evaluated with 5-fold cross-validation.

### Results

| Model | R² Mean | R² Std |
|---|---|---|
| OLS | 0.074 | 0.034 |
| Ridge (α=1) | 0.074 | 0.034 |
| Ridge (α=10) | 0.078 | 0.036 |

Linear models converge to ~7–8% R², confirming the PSI ceiling. Two independent methods hitting the same ceiling suggested it might be real — for linear methods.

---

## 4. Approach 3: Non-Linear Tree Models

### Method

Random Forest (100 trees, max depth 10) and Gradient Boosting (100 trees, max depth 5), evaluated with 5-fold cross-validation.

### Results

| Feature Set | RF R² | GBR R² |
|---|---|---|
| Static only (25 features) | 17.2% | 11.0% |
| Static + Temporal (65 features) | 17.9% | 12.9% |

Tree models broke through the linear ceiling to ~17%, but this raised the question: were they capturing genuine non-linear musical patterns, or implicitly identifying composers?

---

## 5. Disentangling Composer Identity

### 5.1 Composer Classification from Features

A Random Forest classifier trained on 25 static features to predict composer identity achieved **54.1% accuracy** (chance: ~5.5%). The features encode substantial composer-specific stylistic information, meaning regression models could exploit this as a shortcut.

### 5.2 Residualization

We removed the per-composer mean feature vector from each piece, preserving only within-composer variation:

$$\vec{x}'_i = \vec{x}_i - \bar{\vec{x}}_c$$

We then retrained all models on residualized features.

| Config | Model | Raw R² | Residualized R² |
|---|---|---|---|
| Static (25) | OLS | 7.6% | −0.5% |
| Static (25) | Ridge (α=10) | 7.8% | −0.3% |
| Static (25) | Random Forest | 19.4% | 20.2% |
| Static (25) | Gradient Boosting | 10.0% | 21.9% |
| Static+Temporal (65) | OLS | 11.7% | −0.7% |
| Static+Temporal (65) | Ridge (α=10) | 12.2% | −0.2% |
| Static+Temporal (65) | Random Forest | 21.9% | 23.4% |
| Static+Temporal (65) | Gradient Boosting | 21.5% | **24.6%** |

### 5.3 Key Finding

**Linear models collapsed.** Their ~7–8% signal was almost entirely composer-identity leakage. The actual linear relationship between score features and popularity, controlling for composer, is approximately 0%.

**Tree models improved.** Removing composer identity *boosted* accuracy. We suggest that composer identity was consuming model capacity on a shortcut that wasn't strongly predictive. Once blocked, tree models found genuine within-composer patterns worth ~20–25% of variance.

**Temporal features add ~4 percentage points.** Comparing static-only vs static+temporal on residualized GBR (19.3% → 23.1%) confirms that the dynamic arc of a piece carries additional signal.

### 5.4 Robustness

GBR residualized (65 features) across 5 random seeds: **23.7% ± 1.4%**. The result is stable.

### 5.5 The Composer Effect

| Input | R² |
|---|---|
| Score features only (GBR, residualized) | 24.6% |
| Composer name only (OLS) | 35.0% |
| Combined | ~39.0% |

Composer identity explains 1.4× more variance than score features. This is a substantially smaller gap than linear analysis suggested (4.7×). When properly modeled, the notes explain at least two-thirds as much as the name on the cover.

---

## 6. Tier Prediction: Classification

### 6.1 Motivation

The ~24% regression ceiling suggested real but limited continuous signal. However, much of the noise in continuous popularity scores may be irrelevant. The difference between popularity 43 and 48 is likely small and hard to capture, but the difference between "popular" and "not popular" may be a more viable target. If score features define a *region* where popularity is possible rather than a gradient, classification should outperform regression.

### 6.2 Results

All classifiers use residualized features (65 features), evaluated with 5-fold stratified cross-validation.

| Task | Model | AUC | Accuracy |
|---|---|---|---|
| Top 10% vs Rest | RF | 0.796 | 86.5% |
| Top 25% vs Rest | RF | **0.843** | 83.5% |
| Top 25% vs Rest | GBR | 0.842 | — |
| Top 50% vs Rest | GBR | 0.776 | 69.4% |
| Top 25% vs Bottom 25% | GBR | **0.887** | 77.0% |

The top-25% vs rest result (AUC 0.84) is our primary metric, as it uses the full dataset without discarding middle cases. The extremes comparison (AUC 0.89) provides supporting evidence that the signal strengthens when ambiguous cases are removed.

---

## 7. Verification

### 7.1 Leakage Check

| Features | AUC |
|---|---|
| Raw | 0.838 |
| Residualized | 0.842 |

The classification signal survives residualization intact (Raw AUC ≈ Residualized AUC), confirming it is **not** a composer-identity artifact.

### 7.2 Precision and Recall

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Miss (not popular) | 0.83 | 0.97 | 0.90 |
| Hit (popular) | 0.86 | 0.45 | 0.59 |

The precision-recall profile is asymmetric: 86% precision means that when the model predicts a piece is popular, it is correct 86% of the time. 45% recall means roughly half of actually popular pieces are identified. This suggests that approximately half of popular pieces are popular for reasons visible in the score, while the other half are popular for reasons the score cannot capture (performance quality, cultural context, algorithmic amplification).

### 7.3 Calibration

| Predicted Probability | Observed Frequency |
|---|---|
| 0.06 | 0.07 |
| 0.15 | 0.09 |
| 0.25 | 0.17 |
| 0.35 | 0.32 |
| 0.44 | 0.57 |
| 0.55 | 0.64 |
| 0.65 | 0.81 |
| 0.75 | 0.97 |
| 0.84 | 0.96 |
| 0.92 | 1.00 |

The model is consistently **underconfident**: when it assigns a 75% probability of being in the top quartile, the actual frequency is 97%. Pieces that score highly on the model's internal ranking are almost certainly popular.

---

## 8. Summary

1. **Linear methods hit an ~8% ceiling** that was almost entirely composer-identity leakage (drops to ~0% after residualization).
2. **Non-linear tree models break through to ~24%** R² and *improve* after residualization, confirming genuine within-composer signal.
3. **Temporal features add ~4 percentage points**, confirming that the dynamic arc of a piece carries predictive information beyond static averages.
4. **Classification outperforms regression** (AUC 0.84 vs R² 0.24), suggesting score features define a region where popularity is *possible* rather than a gradient along which it increases.
5. **The classification signal is not a composer artifact** — it survives residualization intact.
6. **The model is underconfident** — pieces it flags as likely popular are almost certainly popular (97% observed at 75% predicted).
7. **Score features explain ~25% of variance, composer identity explains ~35%**, and the combined model reaches ~39%, leaving ~61% unexplained by either notes or name.
