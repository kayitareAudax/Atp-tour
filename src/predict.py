#!/usr/bin/env python3
"""
Wimbledon Match Predictor — Inference
Load the trained XGBoost model and predict outcome probability for a given match-up.
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def load_model():
    """Load the trained model and its artifacts."""
    path = MODELS_DIR / "wimbledon_xgb_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `src/wimbledon_predictor.py` first."
        )
    with open(path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts["model"], artifacts["encoders"], artifacts["feature_cols"], artifacts["metrics"]


def predict_match(player_a: dict, player_b: dict, round_name: str = "R128") -> float:
    """
    Predict probability that player_a beats player_b at Wimbledon.

    Parameters
    ----------
    player_a : dict with keys:
        rank, rank_points, age, ht, hand ('R'/'L'), ioc, seed, serve_win_pct, bp_conv
    player_b : dict with same keys
    round_name : str, e.g. 'R128', 'QF', 'SF', 'F'

    Returns
    -------
    prob_a_wins : float (0–1)
    """
    model, encoders, feature_cols, metrics = load_model()

    round_order = {
        "R128": 1, "R64": 2, "R32": 3, "R16": 4,
        "QF": 5, "SF": 6, "F": 7
    }
    round_num = round_order.get(round_name, 1)

    # Build feature row
    row = {
        "surface_enc": encoders["surface"].transform(["Grass"])[0],
        "round_num": round_num,
        "a_rank": player_a.get("rank", 999),
        "a_rank_points": player_a.get("rank_points", 0),
        "a_age": player_a.get("age", 27),
        "a_ht": player_a.get("ht", 185),
        "a_hand_enc": encoders["hand"].transform([player_a.get("hand", "R")])[0],
        "a_ioc_enc": encoders["ioc_a"].transform([player_a.get("ioc", "UNK")])[0],
        "a_seed": player_a.get("seed", 0),
        "a_serve_win_pct": player_a.get("serve_win_pct", 0.65),
        "a_bp_conv": player_a.get("bp_conv", 0.4),
        "b_rank": player_b.get("rank", 999),
        "b_rank_points": player_b.get("rank_points", 0),
        "b_age": player_b.get("age", 27),
        "b_ht": player_b.get("ht", 185),
        "b_hand_enc": encoders["hand"].transform([player_b.get("hand", "R")])[0],
        "b_ioc_enc": encoders["ioc_b"].transform([player_b.get("ioc", "UNK")])[0],
        "b_seed": player_b.get("seed", 0),
        "b_serve_win_pct": player_b.get("serve_win_pct", 0.65),
        "b_bp_conv": player_b.get("bp_conv", 0.4),
        "rank_diff": player_b.get("rank", 999) - player_a.get("rank", 999),
        "rank_points_diff": player_a.get("rank_points", 0) - player_b.get("rank_points", 0),
        "age_diff": player_a.get("age", 27) - player_b.get("age", 27),
        "ht_diff": player_a.get("ht", 185) - player_b.get("ht", 185),
        "same_hand": 1 if player_a.get("hand", "R") == player_b.get("hand", "R") else 0,
        "seed_diff": player_a.get("seed", 0) - player_b.get("seed", 0),
        "rank_ratio": (
            player_b.get("rank", 999) / (player_a.get("rank", 999) + player_b.get("rank", 999))
            if (player_a.get("rank", 999) + player_b.get("rank", 999)) > 0 else 0.5
        ),
        "serve_win_diff": player_a.get("serve_win_pct", 0.65) - player_b.get("serve_win_pct", 0.65),
    }

    # Fill match-stat columns (unknown pre-match → zeros)
    stat_cols = [
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced", "minutes",
    ]
    for col in stat_cols:
        row[col] = 0

    X = pd.DataFrame([row])[feature_cols]
    proba = model.predict_proba(X)[:, 1][0]
    return float(proba)


def batch_predict(matchups: list[dict]) -> list[dict]:
    """Predict multiple match-ups. Each entry in matchups has player_a, player_b, round."""
    results = []
    for m in matchups:
        prob = predict_match(m["player_a"], m["player_b"], m.get("round", "R128"))
        results.append({
            "player_a": m["player_a"].get("name", "A"),
            "player_b": m["player_b"].get("name", "B"),
            "prob_a_wins": round(prob, 4),
            "predicted_winner": m["player_a"]["name"] if prob >= 0.5 else m["player_b"]["name"],
            "confidence": round(abs(prob - 0.5) * 2, 4),
        })
    return results


# ── CLI Demo ───────────────────────────────────────────────────────────────
def main():
    model, encoders, feature_cols, metrics = load_model()
    print("═" * 55)
    print("  🎾 Wimbledon XGBoost — Inference Demo")
    print("═" * 55)
    print(f"  Model Metrics: Accuracy={metrics['accuracy']:.3f}, "
          f"ROC-AUC={metrics['roc_auc']:.3f}, F1={metrics['f1']:.3f}")
    print("─" * 55)

    # Example: Djokovic vs Alcaraz (hypothetical 2024 SF)
    djokovic = {
        "name": "Novak Djokovic",
        "rank": 2, "rank_points": 8460, "age": 37.1, "ht": 188,
        "hand": "R", "ioc": "SRB", "seed": 2,
        "serve_win_pct": 0.72, "bp_conv": 0.42,
    }
    alcaraz = {
        "name": "Carlos Alcaraz",
        "rank": 3, "rank_points": 8130, "age": 21.1, "ht": 183,
        "hand": "R", "ioc": "ESP", "seed": 3,
        "serve_win_pct": 0.69, "bp_conv": 0.44,
    }

    prob = predict_match(djokovic, alcaraz, "SF")
    winner = djokovic["name"] if prob >= 0.5 else alcaraz["name"]
    print(f"  Match: {djokovic['name']} vs {alcaraz['name']} (SF)")
    print(f"  Probability {djokovic['name']} wins: {prob:.4f}")
    print(f"  Predicted winner: {winner}")
    print(f"  Confidence: {abs(prob - 0.5) * 2:.4f}")

    # Batch example
    print("\n─ Batch Demo ─")
    matchups = [
        {
            "player_a": {
                "name": "Jannik Sinner", "rank": 1, "rank_points": 9570,
                "age": 22.8, "ht": 188, "hand": "R", "ioc": "ITA",
                "seed": 1, "serve_win_pct": 0.71, "bp_conv": 0.43,
            },
            "player_b": {
                "name": "Daniil Medvedev", "rank": 5, "rank_points": 6485,
                "age": 28.3, "ht": 198, "hand": "R", "ioc": "RUS",
                "seed": 5, "serve_win_pct": 0.68, "bp_conv": 0.38,
            },
            "round": "QF",
        },
        {
            "player_a": {
                "name": "Alexander Zverev", "rank": 4, "rank_points": 7300,
                "age": 27.2, "ht": 198, "hand": "R", "ioc": "GER",
                "seed": 4, "serve_win_pct": 0.70, "bp_conv": 0.35,
            },
            "player_b": {
                "name": "Alex de Minaur", "rank": 9, "rank_points": 3630,
                "age": 25.3, "ht": 183, "hand": "R", "ioc": "AUS",
                "seed": 9, "serve_win_pct": 0.66, "bp_conv": 0.41,
            },
            "round": "R16",
        },
    ]
    results = batch_predict(matchups)
    for r in results:
        print(f"  {r['player_a']} vs {r['player_b']} → "
              f"{r['predicted_winner']} ({r['prob_a_wins']:.3f}, conf={r['confidence']:.3f})")

    print("\n" + "═" * 55)
    print("  ✅ Inference ready. Import `predict_match` to use programmatically.")
    print("═" * 55)


if __name__ == "__main__":
    main()