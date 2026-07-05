#!/usr/bin/env python3
"""
Wimbledon Match Predictor using XGBoost
Uses Jeff Sackmann's ATP tennis dataset to predict Wimbledon match outcomes.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
TOURNEY_FILTER = "Wimbledon"
GRASS_SURFACE_ONLY = True  # Wimbledon is always grass, but filter anyway
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ── Load Data ──────────────────────────────────────────────────────────────
def load_data(years: list[int]) -> pd.DataFrame:
    """Load and concatenate ATP match data for given years."""
    dfs = []
    for year in years:
        fp = DATA_DIR / f"atp_matches_{year}.csv"
        if fp.exists():
            df = pd.read_csv(fp, low_memory=False)
            dfs.append(df)
            print(f"  Loaded {year}: {len(df):,} matches")
        else:
            print(f"  ⚠  Missing: {fp}")
    if not dfs:
        raise FileNotFoundError("No ATP match data found.")
    return pd.concat(dfs, ignore_index=True)


def filter_wimbledon(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Wimbledon (grass) main-draw singles matches."""
    mask = df["tourney_name"].str.contains(TOURNEY_FILTER, case=False, na=False)
    wim = df[mask].copy()
    # Remove qualifying rounds & doubles (best_of == 5 for men's singles)
    wim = wim[wim["best_of"] == 5]
    wim = wim[wim["round"].notna()]
    print(f"  Wimbledon matches after filtering: {len(wim):,}")
    return wim


# ── Feature Engineering ────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rich feature set for each match.  
    Target: winner = 1 (player A), loser = 0 (player B).  
    We create one row per match from winner's perspective (always label=1)
    and then duplicate rows by flipping to create balanced data.
    """
    feats = pd.DataFrame()

    # Basic identifiers
    feats["match_id"] = df["tourney_id"].astype(str) + "_" + df["match_num"].astype(str)
    feats["surface"] = df["surface"]  # always Grass for Wimbledon
    feats["round"] = df["round"]
    feats["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")

    # Encode round as ordinal
    round_order = {
        "R128": 1, "R64": 2, "R32": 3, "R16": 4,
        "QF": 5, "SF": 6, "F": 7
    }
    feats["round_num"] = feats["round"].map(round_order).fillna(0).astype(int)

    # Winner features (player A)
    feats["a_rank"] = df["winner_rank"].fillna(999)
    feats["a_rank_points"] = df["winner_rank_points"].fillna(0)
    feats["a_age"] = df["winner_age"].fillna(27)
    feats["a_ht"] = df["winner_ht"].fillna(185)
    feats["a_hand"] = df["winner_hand"].fillna("R")
    feats["a_ioc"] = df["winner_ioc"].fillna("UNK")
    feats["a_seed"] = df["winner_seed"].fillna(0).astype(int)

    # Loser features (player B)
    feats["b_rank"] = df["loser_rank"].fillna(999)
    feats["b_rank_points"] = df["loser_rank_points"].fillna(0)
    feats["b_age"] = df["loser_age"].fillna(27)
    feats["b_ht"] = df["loser_ht"].fillna(185)
    feats["b_hand"] = df["loser_hand"].fillna("R")
    feats["b_ioc"] = df["loser_ioc"].fillna("UNK")
    feats["b_seed"] = df["loser_seed"].fillna(0).astype(int)

    # Derived comparative features
    feats["rank_diff"] = feats["b_rank"] - feats["a_rank"]  # positive = B is worse
    feats["rank_points_diff"] = feats["a_rank_points"] - feats["b_rank_points"]
    feats["age_diff"] = feats["a_age"] - feats["b_age"]
    feats["ht_diff"] = feats["a_ht"] - feats["b_ht"]
    feats["same_hand"] = (feats["a_hand"] == feats["b_hand"]).astype(int)
    feats["seed_diff"] = feats["a_seed"] - feats["b_seed"]

    # Win-probability calculator: Elo-style rank ratio
    feats["rank_ratio"] = np.where(
        (feats["a_rank"] + feats["b_rank"]) > 0,
        feats["b_rank"] / (feats["a_rank"] + feats["b_rank"]),
        0.5
    )

    # Match statistics (from winner perspective)
    stats_cols = [
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
    ]
    for col in stats_cols:
        feats[col] = df[col].fillna(0)

    # Serve quality metrics
    feats["a_serve_win_pct"] = np.where(
        feats["w_svpt"] > 0,
        (feats["w_1stWon"] + feats["w_2ndWon"]) / feats["w_svpt"],
        0
    )
    feats["b_serve_win_pct"] = np.where(
        feats["l_svpt"] > 0,
        (feats["l_1stWon"] + feats["l_2ndWon"]) / feats["l_svpt"],
        0
    )
    feats["serve_win_diff"] = feats["a_serve_win_pct"] - feats["b_serve_win_pct"]

    # Break point conversion
    feats["a_bp_conv"] = np.where(
        feats["l_bpFaced"] > 0,
        (feats["l_bpFaced"] - feats["l_bpSaved"]) / feats["l_bpFaced"],
        0
    )
    feats["b_bp_conv"] = np.where(
        feats["w_bpFaced"] > 0,
        (feats["w_bpFaced"] - feats["w_bpSaved"]) / feats["w_bpFaced"],
        0
    )

    feats["minutes"] = df["minutes"].fillna(0)

    # Target: did player A win? (always True in original data)
    feats["target"] = 1  # original rows: winner = player A

    return feats


def flip_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Create flipped copies (player A ↔ B) so we have balanced labels."""
    flipped = feats.copy()

    # Swap A ↔ B
    swap_pairs = [
        ("a_rank", "b_rank"), ("a_rank_points", "b_rank_points"),
        ("a_age", "b_age"), ("a_ht", "b_ht"),
        ("a_hand", "b_hand"), ("a_ioc", "b_ioc"),
        ("a_seed", "b_seed"), ("a_serve_win_pct", "b_serve_win_pct"),
        ("a_bp_conv", "b_bp_conv"),
    ]
    for col_a, col_b in swap_pairs:
        tmp = flipped[col_a].copy()
        flipped[col_a] = flipped[col_b]
        flipped[col_b] = tmp

    # Swap match stats
    stat_pairs = [
        ("w_ace", "l_ace"), ("w_df", "l_df"),
        ("w_svpt", "l_svpt"), ("w_1stIn", "l_1stIn"),
        ("w_1stWon", "l_1stWon"), ("w_2ndWon", "l_2ndWon"),
        ("w_SvGms", "l_SvGms"), ("w_bpSaved", "l_bpSaved"),
        ("w_bpFaced", "l_bpFaced"),
    ]
    for col_w, col_l in stat_pairs:
        tmp = flipped[col_w].copy()
        flipped[col_w] = flipped[col_l]
        flipped[col_l] = tmp

    # Recompute derived features
    flipped["rank_diff"] = flipped["b_rank"] - flipped["a_rank"]
    flipped["rank_points_diff"] = flipped["a_rank_points"] - flipped["b_rank_points"]
    flipped["age_diff"] = flipped["a_age"] - flipped["b_age"]
    flipped["ht_diff"] = flipped["a_ht"] - flipped["b_ht"]
    flipped["same_hand"] = (flipped["a_hand"] == flipped["b_hand"]).astype(int)
    flipped["seed_diff"] = flipped["a_seed"] - flipped["b_seed"]
    flipped["rank_ratio"] = np.where(
        (flipped["a_rank"] + flipped["b_rank"]) > 0,
        flipped["b_rank"] / (flipped["a_rank"] + flipped["b_rank"]),
        0.5
    )
    flipped["serve_win_diff"] = flipped["a_serve_win_pct"] - flipped["b_serve_win_pct"]

    flipped["target"] = 0  # flipped: player A loses

    return flipped


def prepare_training_data(df: pd.DataFrame) -> tuple:
    """Prepare feature matrix X and target y, plus encoders for inference."""
    # Encode categoricals
    le_hand = LabelEncoder()
    le_ioc_a = LabelEncoder()
    le_ioc_b = LabelEncoder()
    le_surface = LabelEncoder()

    all_hands = pd.concat([df["a_hand"], df["b_hand"]]).unique()
    le_hand.fit(all_hands)
    df["a_hand_enc"] = le_hand.transform(df["a_hand"])
    df["b_hand_enc"] = le_hand.transform(df["b_hand"])

    all_iocs = pd.concat([df["a_ioc"], df["b_ioc"]]).unique()
    le_ioc_a.fit(all_iocs)
    le_ioc_b.fit(all_iocs)
    df["a_ioc_enc"] = le_ioc_a.transform(df["a_ioc"])
    df["b_ioc_enc"] = le_ioc_b.transform(df["b_ioc"])

    df["surface_enc"] = le_surface.fit_transform(df["surface"].astype(str))

    # Feature columns
    feature_cols = [
        "surface_enc", "round_num",
        "a_rank", "a_rank_points", "a_age", "a_ht", "a_hand_enc", "a_ioc_enc",
        "a_seed", "a_serve_win_pct", "a_bp_conv",
        "b_rank", "b_rank_points", "b_age", "b_ht", "b_hand_enc", "b_ioc_enc",
        "b_seed", "b_serve_win_pct", "b_bp_conv",
        "rank_diff", "rank_points_diff", "age_diff", "ht_diff",
        "same_hand", "seed_diff", "rank_ratio", "serve_win_diff",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
        "minutes",
    ]

    X = df[feature_cols].fillna(0)
    y = df["target"].values

    encoders = {
        "hand": le_hand,
        "ioc_a": le_ioc_a,
        "ioc_b": le_ioc_b,
        "surface": le_surface,
    }

    return X, y, encoders, feature_cols


# ── Train Model ────────────────────────────────────────────────────────────
def train_model(X: pd.DataFrame, y: np.ndarray, encoders: dict, feature_cols: list) -> tuple:
    """Split, train XGBoost, and evaluate."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Training samples: {len(X_train):,}")
    print(f"  Test samples:     {len(X_test):,}")
    print(f"  Positive ratio:   {y.mean():.3f}")

    # Scale pos_weight for class balance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluation ─────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "═" * 55)
    print("  📊 Model Evaluation on Test Set")
    print("═" * 55)
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Precision:      {prec:.4f}")
    print(f"  Recall:         {rec:.4f}")
    print(f"  F1-Score:       {f1:.4f}")
    print(f"  ROC-AUC:        {auc:.4f}")
    print("─" * 55)
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print("═" * 55)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        model, X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    print(f"\n  5-Fold CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\n  Top-15 Feature Importances:")
    for _, row in importance.head(15).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"    {row['feature']:<30s} {row['importance']:.4f} {bar}")

    return model, encoders, feature_cols, {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": auc, "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std()
    }


# ── Save Artifacts ─────────────────────────────────────────────────────────
def save_artifacts(model, encoders, feature_cols, metrics):
    """Save model, encoders, and feature list for later inference."""
    artifacts = {
        "model": model,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "metrics": metrics,
    }
    path = MODELS_DIR / "wimbledon_xgb_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n  ✅ Model saved → {path} ({os.path.getsize(path) / 1024:.1f} KB)")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("═" * 55)
    print("  🎾 Wimbledon XGBoost Match Predictor")
    print("═" * 55)

    # 1. Load
    print("\n[1/5] Loading ATP match data...")
    raw = load_data(YEARS)
    print(f"  Total matches loaded: {len(raw):,}")

    # 2. Filter
    print("\n[2/5] Filtering Wimbledon matches...")
    wimbledon = filter_wimbledon(raw)

    if len(wimbledon) == 0:
        print("  ❌ No Wimbledon matches found. Check tourney_name column.")
        # Debug: show available tournament names
        print("  Available tournaments:", raw["tourney_name"].dropna().unique()[:20])
        sys.exit(1)

    # 3. Feature engineering
    print("\n[3/5] Engineering features...")
    feats = build_features(wimbledon)
    flipped = flip_features(feats)
    all_data = pd.concat([feats, flipped], ignore_index=True)
    print(f"  Total after flipping: {len(all_data):,} rows")

    # 4. Prepare & Train
    print("\n[4/5] Training XGBoost model...")
    X, y, encoders, feature_cols = prepare_training_data(all_data)
    model, encoders, feature_cols, metrics = train_model(X, y, encoders, feature_cols)

    # 5. Save
    print("\n[5/5] Saving model artifacts...")
    save_artifacts(model, encoders, feature_cols, metrics)

    print("\n" + "═" * 55)
    print("  ✅ Done! Run `src/predict.py` to make predictions.")
    print("═" * 55)


if __name__ == "__main__":
    main()