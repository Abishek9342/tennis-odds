# ATP Tennis Match Predictor

A full end-to-end machine learning pipeline that predicts ATP tennis match outcomes, calculates betting edges using the Kelly Criterion, and tracks real-world prediction accuracy.

Built with LightGBM, trained on 26 years of ATP data (2000–2026), served via a Streamlit dashboard and a daily CLI prediction script.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quick Start](#quick-start)
3. [End-to-End Pipeline](#end-to-end-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Architecture](#model-architecture)
6. [Live Prediction Workflow](#live-prediction-workflow)
7. [Betting Decision Engine](#betting-decision-engine)
8. [Model Performance](#model-performance)
9. [Cold-Start: New Players](#cold-start-new-players)
10. [Daily Workflow](#daily-workflow)
11. [Known Limitations](#known-limitations)
12. [Technologies](#technologies)

---

## Project Structure

```
tennis-odds/
├── main.py                  # CLI — scrape / features / train / evaluate / predict / ensemble
├── app.py                   # Streamlit dashboard
├── predict_upcoming.py      # Fetch upcoming matches + predictions + Kelly stakes
├── backtest.py              # Historical backtest on any year range
├── check_results.py         # Auto-verify predictions against actual results
│
├── src/
│   ├── scraper.py           # Stage 1: download + normalise raw ATP Excel files
│   ├── features.py          # Stage 2: feature engineering (Elo, form, H2H, odds, rank)
│   ├── model.py             # Stage 3: train, evaluate, predict, ensemble, SHAP
│   └── atp_scraper.py       # Live data: rankings (Sofascore), fixtures, odds API
│
├── data/
│   ├── raw/                 # 27 Excel files, 2000–2026 (~20 MB, tennis-data.co.uk)
│   └── processed/
│       ├── raw.parquet              # 70,362 normalised matches
│       ├── features.parquet         # 69,451 rows × 114 feature columns
│       └── models/
│           ├── model.lgb                    # With-odds model  (AUC 0.763 CV / 0.753 test)
│           ├── model_no_odds.lgb            # No-odds model    (AUC 0.740 CV)
│           ├── feature_cols.json            # 94 features used by with-odds model
│           ├── feature_cols_no_odds.json    # 86 features used by no-odds model
│           └── rank_elo_coeffs.npy          # Rank→Elo calibration for cold-start
│
├── docs/
│   ├── data_pipeline.md
│   ├── feature_engineering.md
│   ├── model_evaluation.md
│   └── lessons_learned.md
│
├── predictions_log.csv      # Running log of all live predictions + actual results
└── pyproject.toml
```

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.11+, uv required)
uv sync

# 2. Full pipeline rebuild (only needed if starting from scratch)
python main.py scrape       # Download all ATP files (2000–2026)
python main.py features     # Build 114-column feature matrix
python main.py train        # Train both LightGBM models

# 3. Streamlit dashboard
uv run streamlit run app.py

# 4. Today's predictions + Kelly stakes
uv run python predict_upcoming.py --date 2026-05-08 --excel

# 5. After matches finish — check results + update accuracy log
uv run python check_results.py
```

---

## End-to-End Pipeline

```
STAGE 1 — DATA COLLECTION                 scraper.py
  Source: tennis-data.co.uk
  27 Excel files, 2000–2026
  ~70,150 ATP matches with bookmaker odds
        │
        ▼ load_and_normalise()
  raw.parquet  (70,150 rows × 22 columns)
        │
        ▼
STAGE 2 — FEATURE ENGINEERING             features.py
  Single chronological pass via _PlayerTracker:
    ├── Elo ratings (global + per surface)
    ├── Rolling win rates (10d / 30d / 90d)
    ├── Streaks & momentum
    ├── Surface affinity
    ├── Set dominance
    ├── Head-to-head history
    ├── Strength of schedule
    ├── Bookmaker odds (Bet365, Pinnacle, Max)
    ├── ATP rank features
    └── Tournament context
  One row per match — deterministic hash assigns P1/P2
        │
        ▼ build_features()
  features.parquet  (69,451 rows × 114 columns)
        │
        ▼
STAGE 3 — MODEL TRAINING                  model.py
  LightGBM binary classification
  Temporal splits (date-based, no shuffle):
    Train  : 2000-01-01 → 2023-12-31
    Val    : 2024-01-01 → 2025-06-30  (early-stopping)
    Test   : 2025-07-01 → 2026-04-30  (never seen)
  Two models saved:
    model.lgb          → with bookmaker odds  (94 features, AUC 0.763 CV / 0.753 test)
    model_no_odds.lgb  → structural signals only (86 features, AUC 0.740 CV)
        │
        ▼
STAGE 4 — LIVE PREDICTION                 predict_upcoming.py / app.py
  Fetch upcoming matches     ← Sofascore unofficial API
  Fetch live ATP rankings    ← Sofascore rankings API
  Fetch live odds            ← The Odds API (Pinnacle + Bet365)
  Run no-odds model          ← always
  Run with-odds model        ← when Pinnacle odds available
  Ensemble blend             ← 0.65 × with-odds + 0.35 × no-odds
  Kelly Criterion            ← size bet from vig-free edge
        │
        ▼
STAGE 5 — RESULTS TRACKING                check_results.py
  Auto-fill actual winners from ATP scores
  Running accuracy + calibration by confidence bucket
  predictions_log.csv updated
```

---

## Feature Engineering

All features are built in a **single chronological pass** (`_PlayerTracker`) with strict query-before-update ordering — no future data can leak into any match's feature vector.

### Player Features (per player, per match)

| Group | Features | Count |
|---|---|---|
| Elo (global) | `w_elo`, `l_elo`, `elo_diff` | 3 |
| Elo (surface) | `w_elo_surf`, `l_elo_surf`, `elo_surf_diff` | 3 |
| Rolling win rates | `win_rate_10d/30d/90d`, surface-specific variants | 12 |
| Match counts | `matches_10d/30d/90d`, surface-specific | 8 |
| Momentum | `momentum` (10d/90d ratio), `win_rate_trend` (30d−90d) | 2 |
| Streaks | `current_streak` (signed), `streak_5`, `last_match_won` | 3 |
| Consistency | `consistency_30d` (std dev of recent results) | 1 |
| Surface affinity | `surface_affinity_30d/90d` (surf rate − overall rate) | 2 |
| Strength of schedule | `avg_opp_elo_30d/90d` | 2 |
| Days since last match | `days_since_last` | 1 |
| Set dominance | `avg_sets_won/lost_30d/90d`, `set_ratio_30d/90d`, `straight_sets_rate_30d/90d` | 8 |

Each feature above exists for both winner (w_) and loser (l_) perspectives, plus a diff (_diff). **Total player features: ~80**.

### Head-to-Head Features

| Feature | Description |
|---|---|
| `h2h_win_rate_w` | P1's all-time H2H win rate |
| `h2h_surface_w` | P1's H2H win rate on current surface |
| `h2h_recent_2y_w` | P1's H2H win rate in last 2 years |
| `h2h_n_matches` | Total H2H meetings |

H2H features default to 0.5 when no prior meetings exist (neutral prior).

### Bookmaker Odds Features

| Feature | Description |
|---|---|
| `implied_prob_w/l` | Bet365 raw implied probability (1/odds) |
| `overround` | Bet365 bookmaker margin |
| `pin_prob_w/l` | Pinnacle vig-free probability |
| `b365_vs_pin` | Soft-vs-sharp divergence (B365 norm − Pinnacle) |
| `max_prob_w/l` | Best available odds, margin-free |

### Rank Features

| Feature | Description |
|---|---|
| `w_rank`, `l_rank` | ATP ranking |
| `rank_diff` | l_rank − w_rank (positive = p1 better ranked) |
| `log_rank_w/l`, `log_rank_diff` | Log-scale rank (stabilises large rank gaps) |

### Tournament Context

`is_grand_slam`, `is_best_of_5`, `round_num`, `surface_code` — all excluded from model (zero-gain, see ZERO_GAIN_COLS).

### Label Assignment

Each match gets a deterministic P1/P2 assignment via `hash(winner + loser + date) % 2`. When label=0 (loser is P1), `_flip_row()` negates all `_diff` features, swaps all `w_*`/`l_*` pairs, and inverts H2H features. This ensures ~50% label balance without any random shuffle that would break temporal ordering.

---

## Model Architecture

### LightGBM Configuration

```python
{
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
}
```

### Training Protocol (Two-Stage)

**Stage 1**: Early-stopping on core training data (≤2023) with validation set (2024–mid-2025) as reference → finds `best_iteration`.

**Stage 2**: Retrain on core + validation combined for exactly `best_iteration` rounds (no early stopping). This is the final saved model.

Cross-validation (5-fold `TimeSeriesSplit`) is run separately for architecture validation.

### Zero-Gain Features (excluded from both models)

9 features confirmed to contribute zero gain in permutation + split analysis:
`round_num`, `is_grand_slam`, `is_best_of_5`, `surface_code`, `w_streak_5`, `l_streak_5`, `rank_pct_w`, `rank_pct_l`, `h2h_recent_2y_w`, `h2h_surface_w`

### Two-Model Ensemble

```
final_prob = 0.65 × prob_with_odds + 0.35 × prob_no_odds
```

Weights reflect relative AUC (0.763 vs 0.740). When odds are unavailable, only the no-odds model is used. The ensemble is used for **all** Kelly/bet decisions.

---

## Live Prediction Workflow

### Name Resolution

Sofascore returns player names in full (`"Jannik Sinner"`). The dataset uses `"Sinner J."` format. `match_to_internal()` handles:
- Simple surnames (Sinner → Sinner J.)
- Compound surnames (Van Assche, Carreno Busta, Budkov Kjaer)
- Suffixes (Jr, Sr, III)
- Accent-stripping (Davidovich Fokina → same as dataset)

### Live Rankings

Rankings are fetched from Sofascore's ATP rankings API (ranking ID 5). Falls back to last historical rank if unavailable.

### Live Odds

The Odds API (free tier: 500 req/month) provides Pinnacle + Bet365 odds for active ATP events. Odds are matched to match fixtures via last-name fuzzy matching.

### Odds Feature Construction

At inference, the same `_compute_odds_features()` function used during training builds the bookmaker feature group from live Pinnacle/Bet365 decimals. These are fed only to the with-odds model.

---

## Betting Decision Engine

### Edge Calculation (Vig-Free)

The edge is calculated against the **vig-free** (margin-removed) market probability, not the raw implied:

```python
raw_w = 1 / pinnacle_odds_p1
raw_l = 1 / pinnacle_odds_p2
vig_free_p1 = raw_w / (raw_w + raw_l)   # removes ~2.3% Pinnacle margin
edge = ensemble_prob - vig_free_p1
```

Using raw implied `1/odds` understates the true edge by the bookmaker margin (~2.3pp for Pinnacle). We compare against the vig-free price since that's what the market actually believes.

### Kelly Criterion

```
Kelly = (b × p − q) / b
    b = decimal_odds − 1
    p = ensemble win probability
    q = 1 − p
```

**Recommended: Quarter Kelly** (25% of full Kelly). Full Kelly maximises long-run growth but variance is very high.

### Qualifying Thresholds

| Threshold | Value | Rationale |
|---|---|---|
| Min edge (vig-free) | ≥ 3% | Covers model uncertainty + transaction costs |
| Min confidence | ≥ 70% | Avoid low-signal matches |
| Min odds | ≥ 1.20 | Very short prices have no Kelly value |

A bet qualifies only when **all three** thresholds are met.

---

## Model Performance

**Date-based splits** (never shuffled):
- Train: 2000-01-01 → 2023-12-31
- Val: 2024-01-01 → 2025-06-30 (early-stopping reference only)
- Test: 2025-07-01 → 2026-04-30 (true holdout — 2,121 matches, completely excluded from training)

| Model | CV AUC | Test AUC | Baseline |
|---|---|---|---|
| LightGBM — with odds (ensemble) | **0.763** | **0.753** | — |
| LightGBM — no odds | **0.740** | — | — |
| Pinnacle closing odds | — | ~0.749 | ← sharp market |
| Elo only | 0.703 | — | — |
| Random | 0.500 | — | ← floor |

The with-odds model beats Pinnacle closing odds on CV (0.763 vs 0.749) because it combines structural signals (Elo, form, streaks) with market signals. On the test holdout it scores 0.753.

### Calibration (no-odds model, 2023–2024)

| Model says | Actual win rate | Gap |
|---|---|---|
| 11% | 11% | +0.5pp |
| 29% | 32% | −3.6pp |
| 55% | 56% | −1.2pp |
| 72% | 72% | +0.3pp |
| 90% | 91% | −1.3pp |

Max calibration gap: 3.6pp. Probabilities are reliable enough to use directly for Kelly staking.

---

## Cold-Start: New Players

When a player has no historical data (wildcard, qualifier), the model would default to Elo=1500, which overestimates players ranked below ~400.

**Solution**:
1. During training, `build_rank_elo_table()` fits `Elo ≈ a × log1p(rank) + b` on all 70k historical (rank, Elo) pairs and saves coefficients to `rank_elo_coeffs.npy`.
2. At inference, if a player is absent from the dataset:
   - Their live ATP rank is fetched from Sofascore
   - `rank_to_elo(rank, poly)` estimates their Elo (e.g. rank 100 → 1572, rank 400 → 1529)
   - `fetch_sofascore_player_recent()` fetches their last 15 results to seed form features

This prevents the model from over- or under-estimating an opponent's strength when facing a player it has never seen.

> **Note**: Cold-start applies only via `main.py ensemble` / `model.predict()`. The daily `predict_upcoming.py` requires both players to be in the historical dataset.

---

## Daily Workflow

### Morning — get predictions

```bash
# All upcoming matches (next 2 days), with Pinnacle odds auto-loaded
uv run python predict_upcoming.py --excel

# Specific date
uv run python predict_upcoming.py --date 2026-05-08 --excel

# Custom bankroll
uv run python predict_upcoming.py --date 2026-05-08 --bankroll 500 --excel
```

Output columns:
- `P1/P2 Win % (no odds)` — structural model
- `P1/P2 Win % (odds)` — with-odds model
- `P1/P2 Win % (ensemble)` — **the probability used for bet sizing**
- `Pinnacle P1/P2 Odds`, `Bet365 P1/P2 Odds`
- `Kelly P1/P2 (fraction)` — quarter Kelly stake as fraction
- `$ Stake` — dollar amount from your bankroll
- `Favourite` — based on ensemble probability
- `Bet Recommendation` — BET or SKIP with reason

### Evening — check results

```bash
uv run python check_results.py
# → auto-fills Actual Winner, prints running accuracy by confidence bucket
```

### Single match — ensemble prediction

```bash
# No odds (no-odds model only)
python main.py ensemble --p1 "Sinner J." --p2 "Zverev A." --surface Clay

# With Pinnacle odds (ensemble: 65% with-odds + 35% no-odds)
python main.py ensemble --p1 "Sinner J." --p2 "Zverev A." --surface Clay \
    --psw 1.18 --psl 4.50
```

### Historical backtest

```bash
uv run python backtest.py --excel
# → AUC, accuracy, calibration, ROI simulation on test holdout (2025-07-01→2026-04-30)
```

---

## Known Limitations

| Limitation | Impact | Potential fix |
|---|---|---|
| Stale rankings during inference | ATP Tour rankings page requires JS rendering; live rank fetch uses Sofascore which may lag | JS-rendered scraping service |
| No serve/return stats | Ace rate, first-serve %, break-point conversion are high-value missing signals | Add Match Charting Project data |
| No injury signals | Pre-match fitness and travel fatigue not captured | Tennis injury tracking APIs |
| No cold-start in `predict_upcoming.py` | New-to-dataset players are skipped entirely | Extend `predict_upcoming.py` with same cold-start logic as `model.predict()` |
| Closing odds only | Dataset has closing prices, not opening — line-movement signal unavailable | Real-time odds feed |
| Bet365 geo-restriction | Bet365 odds unavailable via The Odds API from some regions | Use Pinnacle as primary book (already default) |

---

## Technologies

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Package manager | uv |
| ML model | LightGBM 4.3+ |
| Data | pandas, numpy, pyarrow |
| Dashboard | Streamlit + Plotly |
| Scraping | requests, BeautifulSoup4, lxml |
| Excel I/O | openpyxl, xlrd |
| Live odds | The Odds API (Pinnacle + Bet365) |
| Live fixtures + rankings | Sofascore unofficial JSON API |
| Explainability | LightGBM built-in SHAP (pred_contrib) |
