#!/usr/bin/env python3
"""Market-value trajectory — SHAP feature-attribution analysis.

Answers "how does the model actually predict, and which variables are real
signal vs. noise?" for the best-performing method in
mv_trajectory_model_comparison.py (LightGBM on pinball loss — see that
script's README results). Uses shap.TreeExplainer, which is exact for tree
ensembles (not an approximation) and runs fast even on tens of thousands of
rows.

Usage:
    python3 scripts/mv_trajectory_shap_analysis.py --player-id 581678 --age 23 \
        --value 130000000 --sub-position "Attacking Midfield" --foot right --height 186 \
        --peak-value 180000000 --value-1yr-ago 160000000 --prior-season-minutes 1608 \
        --prior-season-goals 5 --prior-season-assists 2 --days-since-last-transfer 1200 \
        --horizon 365
"""

import argparse

import numpy as np
import pandas as pd
import shap
from google.cloud import bigquery
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
except ImportError as e:
    raise SystemExit("pip install lightgbm") from e

from mv_trajectory_model_comparison import CAT_FEATS, FEATS, NUM_FEATS, PROJECT_ID, PULL_SQL


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player-id", type=int, required=True)
    ap.add_argument("--age", type=int, required=True)
    ap.add_argument("--value", type=float, required=True)
    ap.add_argument("--position", default="Attack")
    ap.add_argument("--sub-position", default="Right Winger")
    ap.add_argument("--foot", default="right")
    ap.add_argument("--height", type=float, default=180.0)
    ap.add_argument("--peak-value", type=float, default=None)
    ap.add_argument("--value-1yr-ago", type=float, default=None)
    ap.add_argument("--prior-season-minutes", type=float, default=0.0)
    ap.add_argument("--team-minutes-available", type=float, default=None)
    ap.add_argument("--prior-season-goals", type=float, default=0.0)
    ap.add_argument("--prior-season-assists", type=float, default=0.0)
    ap.add_argument("--had-paid-transfer-365d", action="store_true")
    ap.add_argument("--had-free-or-loan-365d", action="store_true")
    ap.add_argument("--days-since-last-transfer", type=float, default=99999)
    ap.add_argument("--horizon", type=int, default=365, choices=[365, 730, 1095])
    ap.add_argument("--sample-size", type=int, default=30000, help="holdout rows to run SHAP on")
    args = ap.parse_args()

    client = bigquery.Client(project=PROJECT_ID)
    print("Pulling training data (shared with mv_trajectory_model_comparison.py)...")
    df = client.query(PULL_SQL).to_dataframe()
    for c in df.columns:
        if pd.api.types.is_extension_array_dtype(df[c]) and df[c].dtype.kind in "iuf":
            df[c] = df[c].astype("float64")
    df = df.dropna(subset=["age", "anchor_value", "future_value"]).copy()
    df["ratio"] = df["future_value"] / df["anchor_value"]
    df = df[(df["ratio"] > 0.02) & (df["ratio"] < 50)]
    df["log_ratio"] = np.log(df["ratio"])
    df["log_value"] = np.log(df["anchor_value"])
    df["comp_log_value_avg"] = df["comp_log_value_avg"].fillna(df["comp_log_value_avg"].median())
    df["value_drawdown"] = df["value_drawdown"].fillna(0.0)
    df["momentum_missing"] = df["momentum_1yr"].isna().astype(int)
    df["momentum_1yr"] = df["momentum_1yr"].fillna(df["momentum_1yr"].median())
    df["prior_season_minutes"] = df["prior_season_minutes"].fillna(0)
    df["prior_season_goals"] = df["prior_season_goals"].fillna(0)
    df["prior_season_assists"] = df["prior_season_assists"].fillna(0)
    df["prior_season_ga_per90"] = np.where(
        df["prior_season_minutes"] > 0,
        (df["prior_season_goals"] + df["prior_season_assists"]) / (df["prior_season_minutes"] / 90), 0.0)
    df["minutes_share_missing"] = df["team_minutes_available"].isna().astype(int)
    df["minutes_share"] = np.where(
        df["team_minutes_available"] > 0, df["prior_season_minutes"] / df["team_minutes_available"], 0.0
    ).clip(0, 1.2)
    df["days_since_last_transfer"] = df["days_since_last_transfer"].fillna(99999)
    df["had_paid_transfer_365d"] = df["had_paid_transfer_365d"].fillna(0).astype(int)
    df["had_free_or_loan_365d"] = df["had_free_or_loan_365d"].fillna(0).astype(int)
    for c in CAT_FEATS:
        df[c] = df[c].fillna("unknown").astype("category")

    sub = df[df["horizon_days"] == args.horizon].copy()
    train, test = train_test_split(sub, test_size=0.2, random_state=42)
    Xtr, ytr = train[FEATS], train["log_ratio"]
    Xte = test[FEATS]

    model = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=400, max_depth=6,
                               learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbosity=-1)
    model.fit(Xtr, ytr, categorical_feature=CAT_FEATS)

    sample = Xte.sample(min(args.sample_size, len(Xte)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample)

    mean_abs = pd.Series(np.abs(shap_values.values).mean(axis=0), index=FEATS).sort_values(ascending=False)
    print(f"\n=== Global SHAP importance (median model, {args.horizon}d, n={len(sample):,}) ===")
    for feat, val in mean_abs.items():
        print(f"  {feat:<28} {val:.4f}")

    print("\n=== Correlation flags among numeric features (|r| > 0.4) ===")
    corr = train[NUM_FEATS].corr()
    for i, f1 in enumerate(NUM_FEATS):
        for f2 in NUM_FEATS[i + 1:]:
            r = corr.loc[f1, f2]
            if abs(r) > 0.4:
                print(f"  {f1} <-> {f2}: r={r:.3f}")

    target = {
        "age": args.age, "log_value": np.log(args.value), "position": args.position,
        "sub_position": args.sub_position,
        "foot": args.foot, "height_in_cm": args.height,
        "comp_log_value_avg": df["comp_log_value_avg"].median(),
        "value_drawdown": np.log(args.value / args.peak_value) if args.peak_value else 0.0,
        "momentum_1yr": np.log(args.value / args.value_1yr_ago) if args.value_1yr_ago else df["momentum_1yr"].median(),
        "momentum_missing": 0 if args.value_1yr_ago else 1,
        "prior_season_minutes": args.prior_season_minutes,
        "minutes_share": (args.prior_season_minutes / args.team_minutes_available
                           if args.team_minutes_available else 0.0),
        "minutes_share_missing": 0 if args.team_minutes_available else 1,
        "prior_season_ga_per90": ((args.prior_season_goals + args.prior_season_assists) / (args.prior_season_minutes / 90)
                                   if args.prior_season_minutes > 0 else 0.0),
        "had_paid_transfer_365d": int(args.had_paid_transfer_365d),
        "had_free_or_loan_365d": int(args.had_free_or_loan_365d),
        "days_since_last_transfer": args.days_since_last_transfer,
    }
    row = pd.DataFrame([target])
    for c in CAT_FEATS:
        row[c] = row[c].astype(pd.CategoricalDtype(categories=df[c].cat.categories))
    row = row[FEATS]

    player_shap = explainer(row)
    base_ratio = np.exp(player_shap.base_values[0])
    pred_ratio = np.exp(model.predict(row)[0])
    print(f"\n=== Player-specific SHAP breakdown, {args.horizon}d ===")
    print(f"Population average prediction: {base_ratio:.3f}x")
    print(f"This player's prediction: {pred_ratio:.3f}x")
    contribs = pd.Series(player_shap.values[0], index=FEATS).sort_values(key=abs, ascending=False)
    for feat, val in contribs.items():
        direction = "pushes UP  " if val > 0 else "pushes DOWN"
        print(f"  {feat:<28} value={str(row[feat].iloc[0]):<14} SHAP={val:+.4f}  ({direction})")


if __name__ == "__main__":
    main()
