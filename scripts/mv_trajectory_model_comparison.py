#!/usr/bin/env python3
"""Market-value trajectory — five-method comparison (research/validation tool).

Compares the deployed nonparametric method (mv_trajectory_projection.py's
empirical exact-age cohort) against four fitted alternatives, on the same
underlying data, for a given player:

  A. Empirical exact-age bucket (deployed method) — nonparametric, no shape
     assumption, but ignores every feature except age.
  B. Quantile regression (statsmodels QuantReg) — log(future/anchor) ~ a
     linear/quadratic-in-age formula with the features below, fit
     separately at q=.25/.5/.75. Pools across all ages via a smooth curve
     instead of one exact bucket, but assumes additive/linear effects.
  C. XGBoost, reg:quantileerror objective, same features as B.
  D. LightGBM, objective="quantile" — leaf-wise tree growth vs. XGBoost's
     level-wise. Consistently the best performer on pinball loss below.
  E. sklearn's HistGradientBoostingRegressor, native loss="quantile" — a
     third, independent boosting implementation.

Not meant to run per-prediction — pulls ~1.15M rows from BigQuery
(valuations + transfers + performance stats) and fits ~40 models (5
methods x 3 horizons, several quantiles each), taking a few minutes. Use it
to validate/recalibrate the deployed empirical method against real
held-out accuracy, not as a per-player lookup step. For feature-attribution
analysis, see the companion script mv_trajectory_shap_analysis.py.

Feature set, and why: age, log(anchor_value), a competition-strength proxy
(mean log-value of players in the same domestic league), sub_position/
foot/height, value_drawdown (log-distance below the player's own trailing-
peak value, from a point-in-time-safe windowed MAX over player_valuations
only), momentum_1yr (log-change vs. the player's own valuation ~1yr
earlier), transfer-type flags/recency, and prior-calendar-year minutes/
goals/assists from a performance table, normalized into minutes_share
(minutes played / that club's total available minutes that year — a
point-in-time-safe rotation-share signal).

Two players-table fields are deliberately excluded throughout: a player's
current contract-expiration date and all-time-peak value are TODAY's-
snapshot fields, not point-in-time historical ones — using them on a
historical training row would leak information not knowable as of that
row's actual anchor date (a future contract renewal, or a later peak).

Results, holdout-backtested (80/20 split), see README for a worked example
and the actual numbers from a real run.

Usage:
    python3 scripts/mv_trajectory_model_comparison.py --player-id 581678 \
        --age 23 --value 130000000 --sub-position "Attacking Midfield" \
        --foot right --height 186 --peak-value 180000000 --value-1yr-ago 160000000 \
        --prior-season-minutes 1608 --prior-season-goals 5 --prior-season-assists 2 \
        --days-since-last-transfer 1200
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xgboost as xgb
from google.cloud import bigquery
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

PROJECT_ID = "real-zaragoza-500608"
DATASET = "raw_transfermarkt_datasets"
QUANTILES = [0.25, 0.5, 0.75]
HORIZONS = [(365, 270, 455, "+1 season"), (730, 635, 820, "+2 seasons"), (1095, 1000, 1185, "+3 seasons")]

PULL_SQL = f"""
WITH comp_strength AS (
  SELECT player_club_domestic_competition_id AS competition_id,
         AVG(LOG(market_value_in_eur)) AS comp_log_value_avg
  FROM `{PROJECT_ID}.{DATASET}.player_valuations`
  WHERE market_value_in_eur > 0 AND player_club_domestic_competition_id IS NOT NULL
  GROUP BY competition_id HAVING COUNT(*) >= 200
),
val_hist AS (
  SELECT player_id, date, market_value_in_eur,
         MAX(market_value_in_eur) OVER (
           PARTITION BY player_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
         ) AS trailing_peak_value,
         LAG(market_value_in_eur, 1) OVER (PARTITION BY player_id ORDER BY date) AS prev_value,
         LAG(date, 1) OVER (PARTITION BY player_id ORDER BY date) AS prev_date
  FROM `{PROJECT_ID}.{DATASET}.player_valuations` WHERE market_value_in_eur > 0
),
apps_by_club_year AS (
  SELECT player_id, EXTRACT(YEAR FROM date) AS yr, player_club_id,
         SUM(minutes_played) AS mp, SUM(goals) AS g, SUM(assists) AS a, COUNT(*) AS apps
  FROM `{PROJECT_ID}.{DATASET}.appearances` GROUP BY player_id, yr, player_club_id
),
primary_club_year AS (
  SELECT player_id, yr, player_club_id, mp, g, a,
         ROW_NUMBER() OVER (PARTITION BY player_id, yr ORDER BY apps DESC) AS rn
  FROM apps_by_club_year
),
team_games_by_year AS (
  SELECT club_id, yr, COUNT(*) AS n_games FROM (
    SELECT home_club_id AS club_id, EXTRACT(YEAR FROM date) AS yr FROM `{PROJECT_ID}.{DATASET}.games`
    UNION ALL
    SELECT away_club_id AS club_id, EXTRACT(YEAR FROM date) AS yr FROM `{PROJECT_ID}.{DATASET}.games`
  )
  GROUP BY club_id, yr
),
appearances_yearly AS (
  SELECT p.player_id, p.yr, p.mp, p.g, p.a, t.n_games * 90 AS team_minutes_available
  FROM primary_club_year p
  LEFT JOIN team_games_by_year t ON t.club_id = p.player_club_id AND t.yr = p.yr
  WHERE p.rn = 1
),
anchors AS (
  SELECT v.player_id, v.date AS anchor_date, v.market_value_in_eur AS anchor_value,
         DATE_DIFF(v.date, p.date_of_birth, YEAR) AS age,
         p.position, p.sub_position, p.foot, p.height_in_cm,
         v.player_club_domestic_competition_id AS competition_id,
         h.trailing_peak_value, h.prev_value, h.prev_date
  FROM `{PROJECT_ID}.{DATASET}.player_valuations` v
  JOIN `{PROJECT_ID}.{DATASET}.players` p USING(player_id)
  JOIN val_hist h ON h.player_id = v.player_id AND h.date = v.date AND h.market_value_in_eur = v.market_value_in_eur
  WHERE v.market_value_in_eur > 0 AND p.position != "Goalkeeper" AND p.date_of_birth IS NOT NULL
    AND DATE_DIFF(v.date, p.date_of_birth, YEAR) BETWEEN 15 AND 38
),
anchors_enriched AS (
  SELECT a.*,
         SAFE.LOG(a.anchor_value / a.trailing_peak_value) AS value_drawdown,
         CASE WHEN a.prev_date IS NOT NULL AND DATE_DIFF(a.anchor_date, a.prev_date, DAY) BETWEEN 180 AND 550
              THEN SAFE.LOG(a.anchor_value / a.prev_value) END AS momentum_1yr,
         ay.mp AS prior_season_minutes, ay.g AS prior_season_goals,
         ay.a AS prior_season_assists, ay.team_minutes_available
  FROM anchors a
  LEFT JOIN appearances_yearly ay ON ay.player_id = a.player_id AND ay.yr = EXTRACT(YEAR FROM a.anchor_date) - 1
),
transfer_flags AS (
  SELECT ae.player_id, ae.anchor_date,
         MAX(CASE WHEN DATE_DIFF(ae.anchor_date, t.transfer_date, DAY) BETWEEN 0 AND 365
                   AND t.transfer_fee > 0 THEN 1 ELSE 0 END) AS had_paid_transfer_365d,
         MAX(CASE WHEN DATE_DIFF(ae.anchor_date, t.transfer_date, DAY) BETWEEN 0 AND 365
                   AND (t.transfer_fee = 0 OR t.transfer_fee IS NULL) THEN 1 ELSE 0 END) AS had_free_or_loan_365d,
         MIN(CASE WHEN t.transfer_date <= ae.anchor_date
                   THEN DATE_DIFF(ae.anchor_date, t.transfer_date, DAY) END) AS days_since_last_transfer
  FROM anchors_enriched ae
  LEFT JOIN `{PROJECT_ID}.{DATASET}.transfers` t ON t.player_id = ae.player_id
  GROUP BY ae.player_id, ae.anchor_date
),
horizons AS (
  SELECT 365 AS horizon_days, 270 AS win_lo, 455 AS win_hi UNION ALL
  SELECT 730, 635, 820 UNION ALL SELECT 1095, 1000, 1185
),
future_vals AS (
  SELECT ae.*, tf.had_paid_transfer_365d, tf.had_free_or_loan_365d, tf.days_since_last_transfer,
         h.horizon_days, v2.market_value_in_eur AS future_value,
         ROW_NUMBER() OVER (PARTITION BY ae.player_id, ae.anchor_date, h.horizon_days
           ORDER BY ABS(DATE_DIFF(v2.date, ae.anchor_date, DAY) - h.horizon_days)) AS rn
  FROM anchors_enriched ae
  JOIN transfer_flags tf ON tf.player_id = ae.player_id AND tf.anchor_date = ae.anchor_date
  JOIN horizons h ON TRUE
  JOIN `{PROJECT_ID}.{DATASET}.player_valuations` v2 ON v2.player_id = ae.player_id
   AND DATE_DIFF(v2.date, ae.anchor_date, DAY) BETWEEN h.win_lo AND h.win_hi AND v2.market_value_in_eur > 0
)
SELECT f.player_id, f.anchor_date, f.anchor_value, f.age, f.position, f.sub_position, f.foot,
       f.height_in_cm, f.competition_id, f.horizon_days, f.future_value, f.value_drawdown,
       f.momentum_1yr, f.prior_season_minutes, f.prior_season_goals, f.prior_season_assists,
       f.team_minutes_available, f.had_paid_transfer_365d, f.had_free_or_loan_365d,
       f.days_since_last_transfer, cs.comp_log_value_avg
FROM future_vals f
LEFT JOIN comp_strength cs ON cs.competition_id = f.competition_id
WHERE f.rn = 1
"""

NUM_FEATS = ["age", "log_value", "comp_log_value_avg", "height_in_cm", "value_drawdown",
             "momentum_1yr", "momentum_missing", "prior_season_minutes", "minutes_share",
             "minutes_share_missing", "prior_season_ga_per90", "had_paid_transfer_365d",
             "had_free_or_loan_365d", "days_since_last_transfer"]
CAT_FEATS = ["position", "sub_position", "foot"]
FEATS = NUM_FEATS + CAT_FEATS


def pinball_loss(y_true, y_pred, q):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player-id", type=int, required=True)
    ap.add_argument("--age", type=int, required=True)
    ap.add_argument("--value", type=float, required=True, help="current market value in EUR")
    ap.add_argument("--position", default="Attack")
    ap.add_argument("--sub-position", default="Right Winger")
    ap.add_argument("--foot", default="right")
    ap.add_argument("--height", type=float, default=180.0)
    ap.add_argument("--peak-value", type=float, default=None, help="player's own trailing-peak TM value")
    ap.add_argument("--value-1yr-ago", type=float, default=None, help="player's own value ~1yr before now")
    ap.add_argument("--prior-season-minutes", type=float, default=0.0)
    ap.add_argument("--team-minutes-available", type=float, default=None,
                     help="that club's total games (home+away) that year x 90; omit if unknown")
    ap.add_argument("--prior-season-goals", type=float, default=0.0)
    ap.add_argument("--prior-season-assists", type=float, default=0.0)
    ap.add_argument("--had-paid-transfer-365d", action="store_true")
    ap.add_argument("--had-free-or-loan-365d", action="store_true")
    ap.add_argument("--days-since-last-transfer", type=float, default=99999)
    args = ap.parse_args()

    client = bigquery.Client(project=PROJECT_ID)
    print("Pulling training data (~1.15M rows -- valuations + transfers + appearances + games, a few minutes)...")
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
    print(f"{len(df):,} rows after cleaning")

    target = {
        "age": args.age, "log_value": np.log(args.value), "position": args.position,
        "sub_position": args.sub_position, "foot": args.foot, "height_in_cm": args.height,
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

    for horizon, _, _, label in HORIZONS:
        sub = df[df["horizon_days"] == horizon].copy()
        print(f"\n{'='*76}\n{label} ({horizon}d, n={len(sub):,})")
        train, test = train_test_split(sub, test_size=0.2, random_state=42)

        row = pd.DataFrame([target])
        for c in CAT_FEATS:
            row[c] = row[c].astype(pd.CategoricalDtype(categories=df[c].cat.categories))
        row = row[FEATS]

        age_bucket = train[train["age"] == args.age]["ratio"]
        empA = {q: float(age_bucket.quantile(q)) for q in QUANTILES}

        qr_sample = train.sample(min(150_000, len(train)), random_state=42)
        formula = ("log_ratio ~ age + I(age**2) + log_value + comp_log_value_avg + height_in_cm + "
                   "value_drawdown + momentum_1yr + prior_season_minutes + minutes_share + "
                   "prior_season_ga_per90 + had_paid_transfer_365d + had_free_or_loan_365d + "
                   "days_since_last_transfer")
        predB = {}
        for q in QUANTILES:
            fit = smf.quantreg(formula, qr_sample).fit(q=q, max_iter=2000)
            predB[q] = float(np.exp(fit.predict(row).iloc[0]))

        Xtr, ytr = train[FEATS], train["log_ratio"]
        Xte, yte = test[FEATS], test["log_ratio"]
        predC, xgb_test = {}, {}
        for q in QUANTILES:
            m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, n_estimators=300,
                                  max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                  enable_categorical=True, random_state=42)
            m.fit(Xtr, ytr)
            predC[q] = float(np.exp(m.predict(row)[0]))
            xgb_test[q] = m.predict(Xte)

        results = {"A": empA, "B": predB, "C": predC}
        scores = {"C": (float(((yte.values >= xgb_test[0.25]) & (yte.values <= xgb_test[0.75])).mean()),
                        pinball_loss(yte.values, xgb_test[0.25], 0.25), pinball_loss(yte.values, xgb_test[0.75], 0.75))}

        if HAS_LGB:
            predD, lgb_test = {}, {}
            for q in QUANTILES:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=400, max_depth=6,
                                       learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                       random_state=42, verbosity=-1)
                m.fit(Xtr, ytr, categorical_feature=CAT_FEATS)
                predD[q] = float(np.exp(m.predict(row)[0]))
                lgb_test[q] = m.predict(Xte)
            results["D"] = predD
            scores["D"] = (float(((yte.values >= lgb_test[0.25]) & (yte.values <= lgb_test[0.75])).mean()),
                           pinball_loss(yte.values, lgb_test[0.25], 0.25), pinball_loss(yte.values, lgb_test[0.75], 0.75))

        Xtr_hgb = pd.get_dummies(Xtr, columns=CAT_FEATS)
        Xte_hgb = pd.get_dummies(Xte, columns=CAT_FEATS).reindex(columns=Xtr_hgb.columns, fill_value=0)
        row_hgb = pd.get_dummies(row, columns=CAT_FEATS).reindex(columns=Xtr_hgb.columns, fill_value=0)
        predE, hgb_test = {}, {}
        for q in QUANTILES:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, max_depth=6,
                                               learning_rate=0.05, random_state=42)
            m.fit(Xtr_hgb, ytr)
            predE[q] = float(np.exp(m.predict(row_hgb)[0]))
            hgb_test[q] = m.predict(Xte_hgb)
        results["E"] = predE
        scores["E"] = (float(((yte.values >= hgb_test[0.25]) & (yte.values <= hgb_test[0.75])).mean()),
                       pinball_loss(yte.values, hgb_test[0.25], 0.25), pinball_loss(yte.values, hgb_test[0.75], 0.75))

        names = {"A": "A empirical bucket", "B": "B quantile reg", "C": "C XGBoost",
                 "D": "D LightGBM", "E": "E HistGB (sklearn)"}
        print(f"  {'Method':<24}{'low':>8}{'base':>8}{'high':>8}{'cov':>8}{'pin25':>9}{'pin75':>9}")
        for key in ["A", "B", "C", "D", "E"]:
            if key not in results:
                continue
            p = results[key]
            cov, p25, p75 = scores.get(key, (None, None, None))
            covs = f"{cov:.1%}" if cov is not None else "   n/a"
            p25s = f"{p25:.4f}" if p25 is not None else "    n/a"
            p75s = f"{p75:.4f}" if p75 is not None else "    n/a"
            print(f"  {names[key]:<24}{p[0.25]:>8.3f}{p[0.5]:>8.3f}{p[0.75]:>8.3f}{covs:>8}{p25s:>9}{p75s:>9}")


if __name__ == "__main__":
    main()
