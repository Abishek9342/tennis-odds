"""FastAPI service exposing predictions, ensemble probabilities, Kelly stakes,
tournament simulation, and CLV summaries.

Run:
    uv run uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health
    POST /predict              {p1, p2, surface, [odds]}      → ensemble probability + Kelly
    POST /simulate-tournament  {bracket, surface, n_sims}      → Monte-Carlo win probs
    POST /futures-edge         {bracket, surface, market_odds} → edge vs outright odds
    GET  /clv/summary                                          → CLV running stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "processed"
MODEL_DIR = DATA / "models"
DEFAULT_W = 0.65

app = FastAPI(title="tennis-odds prediction API", version="1.0.0")

# Loaded once at startup
_state: dict = {}


@app.on_event("startup")
def _load_artifacts():
    import features as feat_mod
    import model as model_mod

    raw = pd.read_parquet(DATA / "raw.parquet").sort_values("date").reset_index(drop=True)
    tracker = feat_mod.build_tracker(raw)
    known = feat_mod.known_players(raw)
    max_rank = float(pd.concat([raw["w_rank"], raw["l_rank"]]).dropna().max())

    booster_no   = lgb.Booster(model_file=str(MODEL_DIR / "model_no_odds.lgb"))
    cols_no      = json.loads((MODEL_DIR / "feature_cols_no_odds.json").read_text())
    booster_odds = None
    cols_odds: list = []
    if (MODEL_DIR / "model.lgb").exists():
        booster_odds = lgb.Booster(model_file=str(MODEL_DIR / "model.lgb"))
        cols_odds    = json.loads((MODEL_DIR / "feature_cols.json").read_text())

    w = DEFAULT_W
    if (MODEL_DIR / "ensemble_weight.json").exists():
        w = float(json.loads((MODEL_DIR / "ensemble_weight.json").read_text()).get("weight_odds", w))

    calibrator = None
    if (MODEL_DIR / "calibrator.json").exists():
        from calibration import load_calibrator
        calibrator = load_calibrator(MODEL_DIR / "calibrator.json")

    _state.update({
        "raw":         raw,
        "tracker":     tracker,
        "known":       known,
        "max_rank":    max_rank,
        "feat_mod":    feat_mod,
        "model_mod":   model_mod,
        "booster_no":  booster_no,
        "cols_no":     cols_no,
        "booster_odds": booster_odds,
        "cols_odds":   cols_odds,
        "weight_odds": w,
        "calibrator":  calibrator,
    })


# ── Schemas ─────────────────────────────────────────────────────────────────

class OddsIn(BaseModel):
    b365w: Optional[float] = None
    b365l: Optional[float] = None
    psw:   Optional[float] = None
    psl:   Optional[float] = None


class PredictIn(BaseModel):
    p1: str
    p2: str
    surface: str = Field(..., description="Hard / Clay / Grass / Carpet")
    tournament: Optional[str] = None
    round_label: Optional[str] = None
    odds: Optional[OddsIn] = None


class BracketIn(BaseModel):
    bracket: list[str] = Field(..., min_items=2, description="Power-of-2 length")
    surface: str
    n_sims: int = 5000


class FuturesIn(BracketIn):
    market_odds: dict[str, float]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _predict_pair(p1: str, p2: str, surface: str,
                  odds: Optional[OddsIn] = None,
                  tournament: Optional[str] = None,
                  round_label: Optional[str] = None) -> dict:
    s = _state
    feat_mod  = s["feat_mod"]
    model_mod = s["model_mod"]
    tracker   = s["tracker"]

    feat = feat_mod.get_live_features(tracker, p1, p2, surface)
    feat["surface_code"]  = float(feat_mod.SURFACE_MAP.get(surface, float("nan")))
    is_gs = 1.0 if (tournament and any(gs.lower() in tournament.lower()
                                       for gs in feat_mod.GRAND_SLAMS)) else 0.0
    feat["is_grand_slam"] = is_gs
    feat["is_best_of_5"]  = is_gs
    feat["round_num"]     = float(feat_mod.ROUND_MAP.get(str(round_label or "").strip(), float("nan")))

    p1_rank = model_mod._get_last_rank(s["raw"], p1)
    p2_rank = model_mod._get_last_rank(s["raw"], p2)
    feat.update(model_mod._compute_rank_features(p1_rank, p2_rank, s["max_rank"]))

    X_no = pd.DataFrame([{c: feat.get(c, np.nan) for c in s["cols_no"]}])
    p_no = float(s["booster_no"].predict(X_no)[0])

    if odds and s["booster_odds"]:
        feat.update(model_mod._compute_odds_features(odds.model_dump()))
        X = pd.DataFrame([{c: feat.get(c, np.nan) for c in s["cols_odds"]}])
        p_full = float(s["booster_odds"].predict(X)[0])
        w = s["weight_odds"]
        p = w * p_full + (1.0 - w) * p_no
    else:
        p_full = None
        p = p_no

    if s["calibrator"] is not None:
        p = float(s["calibrator"].transform([p])[0])

    return {
        "p1": p1, "p2": p2, "surface": surface,
        "p1_win_prob":    round(p, 4),
        "p2_win_prob":    round(1 - p, 4),
        "prob_no_odds":   round(p_no, 4),
        "prob_with_odds": round(p_full, 4) if p_full is not None else None,
        "weight_odds":    s["weight_odds"],
        "calibrated":     s["calibrator"] is not None,
    }


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok" if _state else "loading",
        "models":       {"with_odds": _state.get("booster_odds") is not None,
                         "no_odds":   _state.get("booster_no") is not None},
        "weight_odds":  _state.get("weight_odds"),
        "calibrated":   _state.get("calibrator") is not None,
        "known_players": len(_state.get("known", [])),
    }


@app.post("/predict")
def predict(req: PredictIn):
    try:
        return _predict_pair(req.p1, req.p2, req.surface, req.odds,
                              req.tournament, req.round_label)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/simulate-tournament")
def simulate_tournament(req: BracketIn):
    from tournament_sim import simulate

    surface = req.surface
    def _prob(a: str, b: str) -> float:
        return _predict_pair(a, b, surface)["p1_win_prob"]

    df = simulate(req.bracket, prob_fn=_prob, n_sims=req.n_sims, seed=0)
    return {"results": df.reset_index().to_dict(orient="records"),
            "n_sims":  req.n_sims}


@app.post("/futures-edge")
def futures_edge(req: FuturesIn):
    from tournament_sim import simulate, futures_edge as _fe

    surface = req.surface
    def _prob(a: str, b: str) -> float:
        return _predict_pair(a, b, surface)["p1_win_prob"]

    sim_df = simulate(req.bracket, prob_fn=_prob, n_sims=req.n_sims, seed=0)
    edge_df = _fe(sim_df, req.market_odds)
    return {"edges": edge_df.to_dict(orient="records")}


@app.get("/clv/summary")
def clv_summary():
    import clv
    return clv.summary() or {"n": 0, "message": "No closed bets yet."}
