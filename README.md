# ATP Tennis Match Predictor

A full end-to-end machine learning pipeline that predicts ATP tennis match outcomes, calculates betting edges using the Kelly Criterion, and tracks real-world prediction accuracy.

Built with LightGBM, trained on 26 years of ATP data (2000–2026), and served via a Streamlit dashboard and CLI.

---

## Project Structure

```
tennis-odds/
├── main.py                  # CLI — scrape / features / train / evaluate / predict / ensemble
├── app.py                   # Streamlit dashboard (UI)
├── predict_upcoming.py      # Fetch today's matches + run predictions + Kelly criterion
├── backtest.py              # Backtest model on historical years
├── check_results.py         # Auto-verify predictions against actual match results
│
├── src/
│   ├── scraper.py           # Stage 1: download + normalise raw ATP data
│   ├── features.py          # Stage 2: feature engineering (Elo, form, H2H, odds, rank)
│   ├── model.py             # Stage 3: train, evaluate, predict, ensemble, SHAP
│   └── atp_scraper.py       # Live data: ATP rankings, scores, upcoming matches, odds API
│
├── data/
│   ├── raw/                 # 27 Excel files, 2000–2026 (~20 MB)
│   └── processed/
│       ├── raw.parquet              # 70,362 normalised matches
│       ├── features.parquet         # 69,451 rows × 114 feature columns
│       └── models/
│           ├── model.lgb                    # With-odds model  (94 features, AUC 0.763 CV / 0.753 test)
│           ├── model_no_odds.lgb            # No-odds model    (86 features, AUC 0.740 CV)
│           ├── feature_cols.json
│           ├── feature_cols_no_odds.json
│           └── rank_elo_coeffs.npy          # Rank→Elo calibration for cold-start inference
│
├── docs/
│   ├── data_pipeline.md         # Scraping + normalisation details
│   ├── feature_engineering.md   # All 114 features + 3 leakage bugs fixed
│   ├── model_evaluation.md      # AUC, baselines, betting ROI, CV results
│   └── lessons_learned.md       # Critical lessons from development
│
├── predictions_log.csv      # Running log of all live predictions + actual results
└── pyproject.toml           # Python dependencies
```

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.14+, uv required)
uv sync

# 2. Full pipeline (only needed if rebuilding from scratch)
python main.py scrape       # Download all ATP files (2000–2026)
python main.py features     # Build 114-column feature matrix
python main.py train        # Train LightGBM with temporal CV

# 3. Dashboard
uv run streamlit run app.py

# 4. Today's predictions
uv run python predict_upcoming.py --date 2026-05-01

# 5. After matches finish — check results
uv run python check_results.py
```

---

## End-to-End Pipeline

```
STAGE 1 — DATA COLLECTION          scraper.py
  tennis-data.co.uk
  27 Excel files (2000–2026)
  70,150 ATP matches
        │
        ▼ load_and_normalise()
  raw.parquet  (70,150 × 22 cols)
        │
        ▼
STAGE 2 — FEATURE ENGINEERING      features.py
  Single chronological pass via _PlayerTracker:
    ├── Elo ratings (global + surface)
    ├── Rolling win rates (10d / 30d / 90d)
    ├── Streaks & momentum
    ├── Surface affinity
    ├── Set dominance
    ├── Head-to-head history
    ├── Strength of schedule
    ├── Bookmaker odds features
    ├── ATP rank features
    └── Tournament context
  One row per match — deterministic hash assigns P1/P2
        │
        ▼ build_features()
  features.parquet  (69,244 × 114 cols)
        │
        ▼
STAGE 3 — MODEL TRAINING            model.py
  LightGBM binary classification
  TimeSeriesSplit(5) — temporal CV, no shuffle
  Two models saved:
    model.lgb          → with bookmaker odds  (AUC 0.781)
    model_no_odds.lgb  → structural signals only (AUC 0.720)
        │
        ▼
STAGE 4 — LIVE PREDICTION           predict_upcoming.py / app.py
  Fetch upcoming matches  ← atp_scraper.py (Sofascore API)
  Fetch live ATP rankings ← atp_scraper.py (atptour.com)
  Fetch live odds         ← The Odds API (Pinnacle / Bet365)
  Run ensemble prediction ← blend no-odds + with-odds
  Kelly Criterion         ← size bet from edge
        │
        ▼
STAGE 5 — RESULTS TRACKING          check_results.py
  Auto-fill actual winners from ATP scores
  Running accuracy + calibration by confidence bucket
  predictions_log.csv updated
```

---

## Model Performance

Date-based splits: Train ≤ 2023-12-31 · Val 2024-01-01–2025-06-30 · Test 2025-07-01–2026-04-30 (never seen during training).

| Model | CV AUC | Test AUC | Log-loss | Brier |
|---|---|---|---|---|
| LightGBM — with odds | **0.763** | **0.753** | 0.5857 | 0.2015 |
| LightGBM — no odds | **0.740** | — | — | — |
| Pinnacle closing odds | ~0.749 | — | — | — |
| Elo only | 0.703 | — | — | — |
| Random | 0.500 | — | — | — |

Test holdout: 2,121 matches (2025-07-01–2026-04-30), completely excluded from all training. CV AUC measured on temporal folds from 2000–2025-06-30.

### Calibration (no-odds model, 2023–2024)

| Model says | Actual win rate | Gap |
|---|---|---|
| 11% | 11% | +0.5pp |
| 29% | 32% | -3.6pp |
| 55% | 56% | -1.2pp |
| 72% | 72% | +0.3pp |
| 90% | 91% | -1.3pp |

Max calibration gap: 3.6pp. The model probabilities are reliable enough to use directly for Kelly staking.

---

## Key Features (114 total)

### Player Features (via `_PlayerTracker` — single chronological pass)
| Group | Count | Top feature |
|---|---|---|
| Elo (global + surface) | 6 | `elo_diff` — 38% of model gain |
| Rolling win rates (10/30/90d) | 22 | `win_rate_30d` |
| Streaks & lag | 7 | `current_streak` |
| Momentum & consistency | 11 | `momentum` |
| Surface affinity | 9 | `surface_affinity_90d` |
| Set dominance | 18 | `set_ratio_90d_diff` |
| Strength of schedule | 6 | `avg_opp_elo_90d` |
| Head-to-head | 4 | `h2h_win_rate_w` |

### Bookmaker Odds Features
| Feature | Description |
|---|---|
| `implied_prob_w/l` | Bet365 raw implied probability |
| `pin_prob_w/l` | Pinnacle margin-free probability |
| `b365_vs_pin` | Soft vs sharp bookmaker divergence |
| `max_prob_w/l` | Best available odds, normalised |
| `overround` | Bookmaker margin |

### Rank + Tournament Context
`log_rank_diff`, `rank_pct_w/l`, `is_grand_slam`, `is_best_of_5`, `round_num`, `surface_code`

---

## Daily Workflow

### Morning — get predictions
```bash
# Fetch tomorrow's matches with predictions
uv run python predict_upcoming.py --date 2026-05-02 --excel

# With live Pinnacle odds (requires ODDS_API_KEY env variable)
export ODDS_API_KEY="your_key_here"
uv run python predict_upcoming.py --date 2026-05-02 --excel
```

### Evening — check results
```bash
uv run python check_results.py
# → auto-fills Actual Winner, prints running accuracy
```

### Single match — ensemble prediction
```bash
# No odds (no-odds model only)
python main.py ensemble --p1 "Sinner J." --p2 "Zverev A." --surface Clay

# With Pinnacle odds (blended: 65% with-odds + 35% no-odds)
python main.py ensemble --p1 "Sinner J." --p2 "Zverev A." --surface Clay \
    --psw 1.18 --psl 4.50
```

### Backtest on test period
```bash
uv run python backtest.py --excel
# → AUC, accuracy, calibration table, ROI simulation on 2025-07-01→2026-04-30
```

---

## Live Odds Integration

The Odds API (free tier: 500 req/month) provides Pinnacle odds. Bet365 is geo-restricted so Pinnacle is used as the primary sharp-book signal.

```bash
# Get API key from: https://the-odds-api.com
export ODDS_API_KEY="your_key_here"
uv run python predict_upcoming.py --date 2026-05-02
```

When odds are available, the script:
1. Runs the with-odds model
2. Blends with no-odds model (ensemble)
3. Computes Kelly fraction for each player
4. Shows bet recommendation

---

## Ensemble Model

The ensemble blends both models weighted by their relative AUC:

```
final_prob = 0.65 × prob_with_odds + 0.35 × prob_no_odds
```

When odds are unavailable, only the no-odds model is used.

---

## Kelly Criterion

The Kelly fraction tells you what percentage of your bankroll to bet:

```
Kelly = (b × p − q) / b
    b = decimal_odds − 1
    p = model win probability
    q = 1 − p
```

| Kelly result | Meaning | Action |
|---|---|---|
| Positive | Model sees edge over market | Bet Kelly% of bankroll |
| Negative or zero | Market is ahead of model | Skip the bet |

**Recommended:** use Half or Quarter Kelly in practice. Full Kelly maximises long-run growth but is highly volatile.

---

## Evaluating Prediction Quality

Three layers of evaluation:

**1. Historical backtest** — model accuracy on held-out years
```bash
uv run python backtest.py --years 2023 2024 --excel
```

**2. Calibration** — does "70% confident" actually win 70% of the time?
The calibration table in `backtest.py` shows predicted vs actual win rate per bucket. Max gap should be < 5pp for reliable Kelly staking.

**3. Live tracking** — real-world accuracy as predictions accumulate
```bash
uv run python check_results.py
```
After 50+ predictions, check:
- Overall accuracy (target: 65%+ for no-odds model)
- Accuracy by confidence bucket (should increase with confidence)
- Whether Kelly-positive bets actually returned profit

---

## Cold-Start: New Players

When a player has no historical data (e.g., a wildcard or qualifier making their tournament debut), the model would otherwise default to Elo=1500 — an overestimate for anyone ranked below ~400.

**Solution:** During training, `build_rank_elo_table()` fits a log-linear regression `Elo = a × log1p(rank) + b` on all 70k historical (rank, Elo) pairs. The coefficients are saved to `rank_elo_coeffs.npy`.

At inference, if a player is absent from the historical dataset:
1. Their ATP rank is fetched live from atptour.com
2. `rank_to_elo(rank, poly)` estimates their Elo — e.g. rank 100 → Elo 1572, rank 400 → Elo 1529
3. `fetch_sofascore_player_recent()` fetches their last 15 match results to seed form features (win rate, streak, momentum)

This prevents the model from underrating experienced-but-new-dataset players and overcorrecting their opponent's edge.

---

## Known Limitations

- **Stale rankings during inference**: ATP Tour's rankings page requires JS rendering; live rank fetch falls back to last known historical rank. Fix: integrate ATP rankings via a scraping service that handles JS.
- **No serve/return stats**: ace rate, first-serve %, break-point conversion — the highest expected-value missing signal.
- **No injury signals**: sub-injury fatigue and travel schedule not captured.
- **Bet365 geo-restriction**: Bet365 odds unavailable via The Odds API from some regions; use Pinnacle as primary.
- **Closing odds only**: dataset has closing prices, not opening or intraday — line movement signal unavailable.

---

## Technologies

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| Package manager | uv |
| ML model | LightGBM 4.3+ |
| Data | pandas, numpy, pyarrow |
| Dashboard | Streamlit + Plotly |
| Scraping | requests, BeautifulSoup4, lxml |
| Excel I/O | openpyxl, xlrd |
| Live odds | The Odds API |
| Live scores/fixtures | Sofascore (unofficial JSON API) |
| Live rankings | atptour.com (HTML scraping) |
