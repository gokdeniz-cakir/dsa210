# DSA 210

Can the popularity of a classical music piece be predicted from computational analysis of its musical score alone? This project investigates that question using a corpus of 961 MIDI files across 18 composers, 65 score-derived features, and Spotify popularity labels.

## Key Results

- Score-based features explain **~24% of popularity variance** (Gradient Boosting, residualized) — but only with non-linear models. Linear models capture ~8%, almost all of which is composer-identity leakage.
- When reframed as tier classification (top 25% vs rest), score features achieve **AUC 0.84** with **86% precision**.
- The classification signal **survives residualization** (Raw AUC ≈ Residualized AUC), confirming it is not a composer-identity artifact.

## Repository Structure

```
├── data/
│   ├── features and corpus/
│   │   ├── balanced_corpus_v2.csv          # 961 pieces, 25 static features + metadata
│   │   └── temporal_features.csv           # 40 temporal evolution features
│   └── scripts/
│       ├── build_balanced_corpus.py        # Pipeline: download MIDIs, query Spotify, extract features
│       ├── extractors.py                   # Core feature extraction (25 static features from a Score)
│       ├── corpus.py                       # Corpus loading and batch processing utilities
│       └── temporal_features.py            # Temporal evolution feature extraction (40 features)
├── notebooks/
│   ├── eda_hypothesis_tests.ipynb          # EDA and hypothesis testing
│   └── ml_methods.ipynb                    # ML methods and classification
├── reports/
│   ├── eda_report.md                       # EDA written report
│   ├── ml_methods_report.md                # ML methods written report
│   └── final_report.pdf                    # Final report
├── proposal.pdf                            # Project proposal
├── requirements.txt                        # Python dependencies
└── README.md
```

## Reproducing the Analysis

### 1. Setup

```bash
git clone https://github.com/gokdeniz-cakir/dsa210.git
cd dsa210
pip install -r requirements.txt
```

### 2. EDA and Hypothesis Tests

Open and run `notebooks/eda_hypothesis_tests.ipynb`. This notebook loads the data from `data/features and corpus/`, performs exploratory analysis (popularity distributions, composer/era breakdowns, feature correlations), and runs six formal hypothesis tests:

1. Shapiro-Wilk (normality of popularity)
2. Kruskal-Wallis (popularity across composers)
3. Kruskal-Wallis (popularity across eras) with post-hoc Mann-Whitney
4. Spearman correlations with Bonferroni correction (65 features)
5. Mann-Whitney U (top quartile vs rest, 25 static features)
6. Fisher z-test (era-specific correlation differences)

### 3. ML Methods

Open and run `notebooks/ml_methods.ipynb`. This notebook runs the full modeling pipeline:

- Popularity Similarity Index (PSI) baseline
- Linear regression (OLS, Ridge)
- Non-linear tree models (Random Forest, Gradient Boosting)
- Composer-identity residualization analysis
- Tier classification (Top 25% vs Rest, Top 25% vs Bottom 25%)
- Verification (leakage check, precision/recall, calibration)

### 4. Final Report

The final report is available at `reports/final_report.pdf`.

## Data

The dataset consists of two CSV files in `data/features and corpus/`:

- **balanced_corpus_v2.csv** — 961 rows × 29 columns (piece_id, composer, title, popularity + 25 static features)
- **temporal_features.csv** — 961 rows × 43 columns (piece_id, composer, popularity + 40 temporal features)

These CSVs are included in the repo, so the notebooks can be run directly without repeating the extraction pipeline.

### How the Data Was Built

The full pipeline from raw MIDI to analysis-ready CSVs involved three stages. The scripts for each stage are included in `data/scripts/`.

1. **MIDI sourcing and curation.** Raw MIDI files were downloaded from the `drengskapur/midi-classical-music` dataset on HuggingFace (~4,500 files). These were manually filtered for encoding consistency and verified composer attribution, then balanced (max 100 pieces per composer, spanning each composer's popularity range) to produce the 961-piece corpus.

2. **Popularity labeling.** For each piece, the Spotify Web API was queried by title and composer. Popularity scores (0–100) were collected for all matching recordings, and the maximum was taken as the ground-truth label.

3. **Feature extraction.** Each MIDI file was parsed with the `music21` library to extract 25 static features (tonal, temporal, harmonic/structural). The core extraction logic is in `data/scripts/extractors.py`, which computes all 25 features from a single `music21` Score object. Corpus-level loading and batch processing are handled by `data/scripts/corpus.py`. The 40 temporal evolution features (quartile-level values + slopes) were computed separately by `data/scripts/temporal_features.py`. The overall pipeline — downloading MIDIs from HuggingFace, querying Spotify for popularity labels, and calling the extraction functions — was orchestrated by `data/scripts/build_balanced_corpus.py`.

**Note:** Running the data pipeline from scratch requires Spotify API credentials and downloads ~4,500 MIDI files from HuggingFace. The pre-extracted CSVs are provided so the analysis notebooks can be run without this step.

## AI Tool Usage

AI tools (Claude) were used to assist with editing and structuring of the reports (EDA report, ML report and final report).
