# Exploratory Data Analysis and Hypothesis Testing

**Score-Based Popularity Prediction in Classical Music**
DSA 210 — Spring 2025–2026

---

## 1. Data Collection Summary

### 1.1 Source Corpus

MIDI files were sourced from the `drengskapur/midi-classical-music` dataset on HuggingFace (~4,500 files). After manual curation for encoding consistency and verified composer attribution, I retained **961 pieces** across **18 composers** and **5 musical eras** (Baroque, Classical, Romantic, Late Romantic, Impressionist).

### 1.2 Popularity Ground Truth

Popularity was measured via the Spotify Web API. For each piece, I searched recordings by title and composer, collected popularity scores (0–100) for all matching tracks, and took the **maximum** popularity across all recordings. The maximum was chosen to isolate score-level signal by absorbing variation from performer fame and algorithmic amplification into the "unexplained" portion.

### 1.3 Corpus Balancing

To prevent models from learning "composer X = popular," each composer was capped at 100 pieces and pieces were selected to span each composer's full popularity range when possible.

| Composer | Era | Count |
|---|---|---|
| Mozart | Classical | 100 |
| Beethoven | Classical | 100 |
| Schubert | Romantic | 100 |
| Bach | Baroque | 100 |
| Vivaldi | Baroque | 100 |
| Chopin | Romantic | 99 |
| Haydn | Classical | 91 |
| Schumann | Romantic | 61 |
| Brahms | Romantic | 60 |
| Rachmaninov | Late Romantic | 34 |
| Liszt | Romantic | 28 |
| Debussy | Impressionist | 17 |
| Tchaikovsky | Romantic | 16 |
| Dvorak | Romantic | 15 |
| Grieg | Romantic | 15 |
| Ravel | Impressionist | 14 |
| Mendelssohn | Romantic | 10 |
| Handel | Baroque | 1 |
| **Total** | | **961** |

### 1.4 Feature Extraction

I extracted **65 features** from each MIDI file using the `music21` library, organized as:

- **25 static features**: 8 tonal (pitch range, mean, std, entropy, interval mean, stepwise/leap ratios, contour change rate), 6 temporal (total notes, duration, note density, unique durations, duration std, repetition ratio), and 11 harmonic/structural (chord change rate/std, chord size, vertical range, chord pitch class entropy, tonic-dominant ratio, cadence rate, key change rate, modulation mean, sectional repetition/similarity).
- **40 temporal evolution features**: Each piece was divided into 4 quartiles. 8 base features (note count, pitch mean/std/range, note density, interval mean, stepwise ratio, unique pitches) were computed per quartile (32 values) plus the linear slope across quartiles (8 values), capturing dynamic trajectories like "increasing density" or "narrowing pitch range."

### 1.5 Data Quality

The merged dataset has 961 rows × 69 columns (4 metadata + 25 static + 40 temporal). Two static features (`duration` and `note_density`) have 55 missing values (5.7% of rows) due to MIDI files where duration could not be reliably parsed. No temporal features have missing values.

---

## 2. Exploratory Data Analysis

### 2.1 Popularity Distribution

| Statistic | Value |
|---|---|
| Mean | 54.04 |
| Median | 54.00 |
| Std Dev | 11.73 |
| Min / Max | 5 / 88 |
| Q1 / Q3 | 45.0 / 65.0 |
| IQR | 20.0 |
| Skewness | −0.377 |
| Kurtosis | 0.335 |

The distribution is approximately symmetric with a slight left skew, meaning there are somewhat more low-popularity outliers than high-popularity ones. The range spans nearly the full 0–100 Spotify scale.

### 2.2 Popularity by Composer

| Composer | Era | n | Mean | Median | Std |
|---|---|---|---|---|---|
| Tchaikovsky | Romantic | 16 | 67.69 | 72.0 | 8.06 |
| Bach | Baroque | 100 | 65.67 | 65.0 | 2.58 |
| Mendelssohn | Romantic | 10 | 64.20 | 66.0 | 9.51 |
| Mozart | Classical | 100 | 61.82 | 61.0 | 4.73 |
| Beethoven | Classical | 100 | 57.67 | 54.0 | 7.59 |
| Grieg | Romantic | 15 | 57.27 | 55.0 | 13.69 |
| Chopin | Romantic | 99 | 55.40 | 55.0 | 7.10 |
| Debussy | Impressionist | 17 | 53.94 | 57.0 | 11.25 |
| Schumann | Romantic | 61 | 53.90 | 55.0 | 13.95 |
| Ravel | Impressionist | 14 | 52.79 | 50.0 | 9.79 |
| Dvorak | Romantic | 15 | 52.20 | 52.0 | 13.40 |
| Rachmaninov | Late Romantic | 34 | 51.09 | 52.0 | 7.71 |
| Liszt | Romantic | 28 | 50.71 | 51.0 | 13.98 |
| Brahms | Romantic | 60 | 49.32 | 53.0 | 13.32 |
| Schubert | Romantic | 100 | 49.26 | 43.0 | 9.75 |
| Vivaldi | Baroque | 100 | 48.43 | 48.0 | 6.65 |
| Haydn | Classical | 91 | 40.62 | 43.0 | 13.30 |

Composer mean popularity ranges from 40.62 (Haydn) to 67.69 (Tchaikovsky), a span of 27 points. Notably, within-composer variance differs dramatically: Bach's std is only 2.58 (almost all pieces share similar popularity), while Haydn's is 13.30 (wide internal spread). This suggests that for some composers, fame attaches to the name; for others, individual pieces are differentiated.

### 2.3 Popularity by Era

| Era | n | Mean | Median | Std |
|---|---|---|---|---|
| Baroque | 201 | 56.98 | 65.0 | 10.03 |
| Classical | 291 | 53.76 | 57.0 | 12.81 |
| Impressionist | 31 | 53.42 | 50.0 | 10.46 |
| Late Romantic | 34 | 51.09 | 52.0 | 7.71 |
| Romantic | 404 | 53.08 | 53.0 | 11.86 |

Era-level differences are modest compared to composer-level differences. The Baroque era has the highest mean (56.98), driven primarily by Bach. The Impressionist and Late Romantic eras have small sample sizes (31 and 34), limiting the reliability of era-level conclusions for those groups.

### 2.4 Feature–Popularity Correlations (Top 15 by Magnitude)

| Feature | Spearman ρ | p (Bonferroni) | Sig. |
|---|---|---|---|
| pitch_range_q1 | −0.309 | 5.95 × 10⁻²¹ | *** |
| unique_pitches_q1 | −0.300 | 1.49 × 10⁻¹⁹ | *** |
| note_count_q1 | −0.297 | 3.11 × 10⁻¹⁹ | *** |
| note_count_q4 | −0.284 | 1.67 × 10⁻¹⁷ | *** |
| note_count_q2 | −0.283 | 2.48 × 10⁻¹⁷ | *** |
| note_count_q3 | −0.275 | 2.40 × 10⁻¹⁶ | *** |
| pitch_std_q1 | −0.275 | 2.70 × 10⁻¹⁶ | *** |
| pitch_range_q2 | −0.268 | 1.68 × 10⁻¹⁵ | *** |
| duration | −0.266 | 2.41 × 10⁻¹⁴ | *** |
| unique_pitches_q2 | −0.264 | 5.16 × 10⁻¹⁵ | *** |
| chord_change_std | −0.256 | 5.11 × 10⁻¹⁴ | *** |
| chord_pc_entropy | +0.250 | 2.50 × 10⁻¹³ | *** |
| unique_pitches_q3 | −0.246 | 7.37 × 10⁻¹³ | *** |
| pitch_range_q3 | −0.246 | 7.43 × 10⁻¹³ | *** |
| unique_pitches_q4 | −0.237 | 6.29 × 10⁻¹² | *** |

After Bonferroni correction (α = 0.05 / 65 = 0.000769), **34 of 65 features** are significantly correlated with popularity. All top correlations are negative except `chord_pc_entropy` (+0.250), suggesting that popular pieces tend to be shorter, use a narrower pitch range, and have fewer total notes, but with richer harmonic vocabulary. The temporal features (quartile-specific) dominate the top of the list, suggesting that how a piece starts matters more than its average properties.

### 2.5 Key Observation: Composer Clustering

83% of tracks share a popularity score with at least two other tracks by the same composer. This creates visible horizontal "bands" in the composer-vs-popularity plot, confirming that Spotify popularity scores cluster strongly by composer identity. This observation motivated the residualization analysis in the modeling phase.

---

## 3. Hypothesis Tests

All hypothesis tests use non-parametric methods, justified by the non-normality of the popularity distribution (see Test 1). Significance level α = 0.05 throughout, with Bonferroni correction applied for multiple comparisons.

### Test 1: Normality of Popularity Distribution

**H₀:** Popularity scores follow a normal distribution.
**H₁:** Popularity scores do not follow a normal distribution.
**Method:** Shapiro-Wilk test.

| Statistic | Value |
|---|---|
| W | 0.9755 |
| p-value | 1.18 × 10⁻¹¹ |

**Result:** Reject H₀ (p < 0.001). The popularity distribution departs significantly from normality, with slight left skew (−0.377) and mild excess kurtosis (0.335). This justifies the use of non-parametric tests for all subsequent analyses.

### Test 2: Popularity Differs Across Composers

**H₀:** Popularity distributions are identical across composers.
**H₁:** At least one composer's popularity distribution differs from the others.
**Method:** Kruskal-Wallis H test (17 composers with ≥ 10 pieces each; Handel excluded due to n = 1).

| Statistic | Value |
|---|---|
| H | 377.20 |
| p-value | 2.17 × 10⁻⁷⁰ |
| η² (effect size) | 0.383 |

**Result:** Reject H₀ (p < 0.001). Composer identity explains approximately 38.3% of the rank-variance in popularity, a large effect. This confirms that composer fame is the dominant confound in any score-based prediction model.

### Test 3: Popularity Differs Across Eras

**H₀:** Popularity distributions are identical across musical eras.
**H₁:** At least one era's popularity distribution differs from the others.
**Method:** Kruskal-Wallis H test (5 era groups).

| Statistic | Value |
|---|---|
| H | 18.17 |
| p-value | 1.14 × 10⁻³ |
| η² (effect size) | 0.015 |

**Result:** Reject H₀ (p = 0.001), but the effect is very small (η² = 0.015). Post-hoc pairwise Mann-Whitney tests with Bonferroni correction reveal that only two pairs are significantly different:

| Pair | U | p (adjusted) | Sig. |
|---|---|---|---|
| Baroque vs Late Romantic | 4,734 | 0.0022 | ** |
| Baroque vs Romantic | 47,449 | 0.0070 | ** |

All other era pairs are non-significant after correction. Era alone carries minimal predictive information compared to composer identity.

### Test 4: Feature–Popularity Correlations

**H₀ (per feature):** There is no monotonic association between the feature and popularity.
**H₁ (per feature):** A monotonic association exists.
**Method:** Spearman rank correlation with Bonferroni correction (65 simultaneous tests, adjusted α = 7.69 × 10⁻⁴).

**Result:** 34 of 65 features show significant correlations after Bonferroni correction. All significant correlations are modest in magnitude (|ρ| ≤ 0.31), meaning no single feature is a strong predictor on its own. The strongest correlate is `pitch_range_q1` (ρ = −0.309): the pitch range in the opening quartile of a piece is negatively associated with popularity.

The five strongest static features:

| Feature | ρ | Interpretation |
|---|---|---|
| chord_change_std | −0.256 | Consistent harmonic rhythm → more popular |
| duration | −0.254 | Shorter pieces → more popular |
| chord_pc_entropy | +0.250 | Richer chord vocabulary → more popular |
| pitch_std | −0.231 | Narrower pitch focus → more popular |
| pitch_range | −0.223 | Narrower range → more popular |

### Test 5: Feature Differences Between Popular and Unpopular Pieces

**H₀ (per feature):** The feature distribution is identical for top-quartile and non-top-quartile pieces.
**H₁ (per feature):** The distributions differ.
**Method:** Mann-Whitney U test with Bonferroni correction (25 static features, adjusted α = 0.002). Top quartile defined as popularity ≥ 65 (n = 253 pieces vs. n = 708).

**Result:** 12 of 25 static features are significantly different between popular and non-popular pieces. The largest effect sizes (rank-biserial correlation):

| Feature | U | r (rank-biserial) | p (adjusted) | Sig. |
|---|---|---|---|---|
| pitch_range | 57,813 | +0.354 | 1.29 × 10⁻¹⁵ | *** |
| pitch_std | 58,442 | +0.347 | 5.42 × 10⁻¹⁵ | *** |
| duration | 52,677 | +0.349 | 1.52 × 10⁻¹⁴ | *** |
| chord_pc_entropy | 118,594 | −0.324 | 4.60 × 10⁻¹³ | *** |
| chord_change_std | 60,906 | +0.320 | 9.94 × 10⁻¹³ | *** |
| contour_change_rate | 117,970 | −0.317 | 1.64 × 10⁻¹² | *** |

The rank-biserial correlations are moderate (0.18–0.35), consistent with the Spearman correlations. No single feature separates the groups cleanly, but the multivariate combination is potentially discriminating — a hypothesis tested formally in the ML phase.

### Test 6: Feature–Popularity Relationships Vary Across Eras

**H₀:** The Spearman correlation between a feature and popularity is the same across eras.
**H₁:** The correlation differs across eras.
**Method:** Within-era Spearman correlations, with Fisher z-test comparing the two largest groups (Baroque, n = 201, vs. Romantic, n = 404).

| Feature | ρ (Baroque) | ρ (Romantic) | z | p |
|---|---|---|---|---|
| pitch_std | −0.518*** | −0.134** | −5.05 | < 0.0001 |
| pitch_range | −0.436*** | −0.158** | −3.54 | 0.0004 |
| cadence_rate | −0.347*** | +0.016 | −4.36 | < 0.0001 |

**Result:** For three features, the correlation with popularity is significantly different between Baroque and Romantic eras. Most strikingly, `pitch_std` has a strong negative correlation in Baroque (ρ = −0.518) but only a weak one in Romantic (ρ = −0.134), and `cadence_rate` flips from negative (Baroque) to near-zero (Romantic). This confirms that "what makes a piece popular" varies by musical era — a finding with implications for any pooled prediction model.

---

## 4. Summary of Findings

1. **Popularity is non-normal** (Shapiro-Wilk p < 10⁻¹¹), justifying non-parametric methods.
2. **Composer identity dominates** (Kruskal-Wallis η² = 0.38), while era has minimal direct effect (η² = 0.015).
3. **34 of 65 features are significantly correlated with popularity** after Bonferroni correction, but all individual correlations are modest (|ρ| ≤ 0.31).
4. **12 of 25 static features significantly differ** between top-quartile and non-top-quartile pieces (Mann-Whitney, Bonferroni-corrected).
5. **Feature-popularity relationships are era-dependent**: the same feature can have opposite implications for popularity in different eras.
6. The modest individual correlations combined with multiple significant features suggest that **multivariate modeling (explored in the next phase) is required** to assess whether these features jointly carry meaningful predictive power.
