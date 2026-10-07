#!/usr/bin/env python3
"""Market-value trajectory projection — nonparametric peer-cohort method.

For a given player, projects their market value 1/2/3 seasons forward using
a matched-cohort ("comparables") approach rather than a fitted model:

1. Anchor: the player's own most recent valuation (value, age) — or, with
   --as-of-date, their latest valuation on or before that date, which lets
   the same query be used as a backtest ("what would this have said back
   then?") against value history that's since played out.
2. Cohort: every other (player, valuation-date) row at the SAME AGE (exact
   year, no value-band matching) — ages tens of thousands of distinct
   players per horizon, because a ratio-based comparison doesn't need a
   value band to be fair: a 21-year-old's % change is comparable whether
   they're worth EUR200K or EUR20M.
3. Forward ratio: for each cohort row, its own value ~1/2/3 years later
   (nearest match inside a tolerance window), expressed as future/anchor.
4. Band: 25th/50th/75th percentile of that ratio distribution per horizon,
   applied to the target's current value — reported as % change first
   (the primary number), EUR band second (the derived one).

No ML, no regression, nothing fitted — a nonparametric peer-percentile
method, queried fresh from BigQuery every run. Its reliability rests
entirely on the cohort being genuinely comparable and large enough; every
printed n/n_players exists so that risk is visible, not hidden. Position
is deliberately excluded by default: an age-matched cohort with and
without a position filter produces a similar-shaped ratio distribution,
so the position filter was found to cost 6x the sample for little
discriminating power.

Real limitations:
- Doesn't condition on the player's own performance trend — cohort matches
  on age only.
- Survivorship bias: a cohort row only counts at a horizon if that player
  still has a later valuation at all — players who drop out of tracked
  football are silently excluded, biasing the band upward.
- Valuations are a periodic snapshot (~2x/year per player), so "N years
  forward" is a nearest match within a tolerance window, not an exact
  anniversary.
- n_players below ~30 should be read as directional only — shown, not
  hidden.

Data: public Transfermarkt valuation history (via Kaggle/community mirrors
of Transfermarkt's own published data).

Usage:
    python3 scripts/mv_trajectory_projection.py --player-id 581678
    python3 scripts/mv_trajectory_projection.py --player-id 581678 --as-of-date 2021-10-07
"""

import argparse
import datetime

from google.cloud import bigquery

PROJECT_ID = "real-zaragoza-500608"
DATASET = "raw_transfermarkt_datasets"

HORIZONS = [
    (365, 270, 455, "+1 season"),
    (730, 635, 820, "+2 seasons"),
    (1095, 1000, 1185, "+3 seasons"),
]

POSITION_GROUPS = {
    "Centre-Forward": "attack", "Second Striker": "attack",
    "Right Winger": "attack", "Left Winger": "attack",
    "Attacking Midfield": "midfield", "Central Midfield": "midfield",
    "Defensive Midfield": "midfield", "Right Midfield": "midfield", "Left Midfield": "midfield",
    "Centre-Back": "defence", "Left-Back": "defence", "Right-Back": "defence",
    "Goalkeeper": "goalkeeper",
}


def position_filter_sql(mode, sub_position):
    if mode == "off":
        return 'AND p.position != "Goalkeeper"', {}
    if mode == "exact":
        return "AND p.sub_position = @sub_position", {"sub_position": sub_position}
    if mode == "group":
        group = POSITION_GROUPS.get(sub_position)
        members = [k for k, v in POSITION_GROUPS.items() if v == group] if group else [sub_position]
        return "AND p.sub_position IN UNNEST(@position_group)", {"position_group": members}
    raise ValueError(f"unknown position mode: {mode}")


def cohort_sql(extra_filter_sql=""):
    return f"""
WITH target AS (
  SELECT p.player_id, p.sub_position, p.name, v.date AS anchor_date,
         v.market_value_in_eur AS anchor_value,
         DATE_DIFF(v.date, p.date_of_birth, YEAR) AS anchor_age
  FROM `{PROJECT_ID}.{DATASET}.players` p
  JOIN `{PROJECT_ID}.{DATASET}.player_valuations` v USING(player_id)
  WHERE p.player_id = @player_id AND v.date <= @as_of_date
  ORDER BY v.date DESC
  LIMIT 1
),
anchors AS (
  SELECT v.player_id, v.date AS anchor_date, v.market_value_in_eur AS anchor_value
  FROM `{PROJECT_ID}.{DATASET}.player_valuations` v
  JOIN `{PROJECT_ID}.{DATASET}.players` p USING(player_id)
  CROSS JOIN target t
  WHERE v.player_id != t.player_id
    AND DATE_DIFF(v.date, p.date_of_birth, YEAR) = t.anchor_age
    AND v.market_value_in_eur > 0
    {extra_filter_sql}
),
horizons AS (
  SELECT * FROM UNNEST([
    STRUCT(365 AS horizon_days, 270 AS win_lo, 455 AS win_hi),
    STRUCT(730, 635, 820),
    STRUCT(1095, 1000, 1185)
  ])
),
future_vals AS (
  SELECT a.player_id, a.anchor_date, h.horizon_days,
         v2.market_value_in_eur / a.anchor_value AS ratio,
         ROW_NUMBER() OVER (
           PARTITION BY a.player_id, a.anchor_date, h.horizon_days
           ORDER BY ABS(DATE_DIFF(v2.date, a.anchor_date, DAY) - h.horizon_days)
         ) AS rn
  FROM anchors a
  JOIN horizons h ON TRUE
  JOIN `{PROJECT_ID}.{DATASET}.player_valuations` v2
    ON v2.player_id = a.player_id
   AND DATE_DIFF(v2.date, a.anchor_date, DAY) BETWEEN h.win_lo AND h.win_hi
   AND v2.market_value_in_eur > 0
)
SELECT horizon_days, COUNT(*) AS n, COUNT(DISTINCT player_id) AS n_players,
       APPROX_QUANTILES(ratio, 100)[OFFSET(25)] AS p25,
       APPROX_QUANTILES(ratio, 100)[OFFSET(50)] AS p50,
       APPROX_QUANTILES(ratio, 100)[OFFSET(75)] AS p75
FROM future_vals
WHERE rn = 1
GROUP BY horizon_days
ORDER BY horizon_days
"""


FEE_OVERLAY_SQL = f"""
-- Splits the 1-year-forward ratio by whether the player's own next move
-- involves a real fee, a free/loan move, or no transfer at all.
WITH target AS (
  SELECT p.player_id, v.date AS anchor_date, v.market_value_in_eur AS anchor_value,
         DATE_DIFF(v.date, p.date_of_birth, YEAR) AS anchor_age
  FROM `{PROJECT_ID}.{DATASET}.players` p
  JOIN `{PROJECT_ID}.{DATASET}.player_valuations` v USING(player_id)
  WHERE p.player_id = @player_id AND v.date <= @as_of_date
  ORDER BY v.date DESC
  LIMIT 1
),
anchors AS (
  SELECT v.player_id, v.date AS anchor_date, v.market_value_in_eur AS anchor_value
  FROM `{PROJECT_ID}.{DATASET}.player_valuations` v
  JOIN `{PROJECT_ID}.{DATASET}.players` p USING(player_id)
  CROSS JOIN target t
  WHERE v.player_id != t.player_id
    AND p.position != "Goalkeeper"
    AND DATE_DIFF(v.date, p.date_of_birth, YEAR) = t.anchor_age
    AND v.market_value_in_eur > 0
),
future_1yr AS (
  SELECT a.player_id, a.anchor_date, v2.market_value_in_eur / a.anchor_value AS ratio,
         ROW_NUMBER() OVER (PARTITION BY a.player_id, a.anchor_date
           ORDER BY ABS(DATE_DIFF(v2.date, a.anchor_date, DAY) - 365)) AS rn
  FROM anchors a
  JOIN `{PROJECT_ID}.{DATASET}.player_valuations` v2
    ON v2.player_id = a.player_id
   AND DATE_DIFF(v2.date, a.anchor_date, DAY) BETWEEN 270 AND 455
   AND v2.market_value_in_eur > 0
),
categorized AS (
  SELECT f.player_id, f.anchor_date, f.ratio,
         MAX(CASE WHEN t.transfer_fee > 0 THEN 1 ELSE 0 END) AS had_paid_fee,
         MAX(CASE WHEN t.transfer_fee = 0 THEN 1 ELSE 0 END) AS had_free_move
  FROM future_1yr f
  LEFT JOIN `{PROJECT_ID}.{DATASET}.transfers` t
    ON t.player_id = f.player_id
   AND t.transfer_date BETWEEN f.anchor_date AND DATE_ADD(f.anchor_date, INTERVAL 455 DAY)
  WHERE f.rn = 1
  GROUP BY f.player_id, f.anchor_date, f.ratio
)
SELECT
  CASE WHEN had_paid_fee = 1 THEN "paid_fee"
       WHEN had_free_move = 1 THEN "free_or_loan"
       ELSE "no_transfer" END AS category,
  COUNT(*) AS n, COUNT(DISTINCT player_id) AS n_players,
  APPROX_QUANTILES(ratio, 100)[OFFSET(25)] AS p25,
  APPROX_QUANTILES(ratio, 100)[OFFSET(50)] AS p50,
  APPROX_QUANTILES(ratio, 100)[OFFSET(75)] AS p75
FROM categorized
GROUP BY category
"""


def eur(v):
    return f"€{v:,.0f}"


def pct(ratio):
    change = (ratio - 1) * 100
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.0f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player-id", type=int, required=True, help="Transfermarkt player_id")
    ap.add_argument("--as-of-date", type=str, default=None,
                     help="YYYY-MM-DD — anchor on the latest valuation on/before this date "
                          "instead of the player's current one (for backtesting)")
    ap.add_argument("--position-mode", choices=["off", "group", "exact"], default="off")
    ap.add_argument("--min-n", type=int, default=30, help="warn below this distinct-player count")
    args = ap.parse_args()

    as_of = (datetime.date.fromisoformat(args.as_of_date) if args.as_of_date
             else datetime.date.today())

    client = bigquery.Client(project=PROJECT_ID)

    anchor_row = list(client.query(
        f"""SELECT p.name, p.sub_position, v.date, v.market_value_in_eur,
                   DATE_DIFF(v.date, p.date_of_birth, YEAR) AS age
            FROM `{PROJECT_ID}.{DATASET}.players` p
            JOIN `{PROJECT_ID}.{DATASET}.player_valuations` v USING(player_id)
            WHERE p.player_id = {args.player_id} AND v.date <= '{as_of.isoformat()}'
            ORDER BY v.date DESC LIMIT 1"""
    ).result())
    if not anchor_row:
        raise SystemExit(f"No valuation history found for player_id={args.player_id} as of {as_of}")
    a = anchor_row[0]
    v0 = a.market_value_in_eur
    print(f"\n{a.name} — {a.sub_position}, age {a.age}, TM value {eur(v0)} as of {a.date.date()}"
          f"{' (backtest anchor)' if args.as_of_date else ''}\n")

    pos_extra, pos_params = position_filter_sql(args.position_mode, a.sub_position)
    param_list = [
        bigquery.ScalarQueryParameter("player_id", "INT64", args.player_id),
        bigquery.ScalarQueryParameter("as_of_date", "DATE", as_of.isoformat()),
    ]
    for k, val in pos_params.items():
        if isinstance(val, list):
            param_list.append(bigquery.ArrayQueryParameter(k, "STRING", val))
        else:
            param_list.append(bigquery.ScalarQueryParameter(k, "STRING", val))
    job_config = bigquery.QueryJobConfig(query_parameters=param_list)

    general = {r.horizon_days: r for r in client.query(cohort_sql(pos_extra), job_config=job_config).result()}
    print(f"Peer cohort — exact age {a.age}, position-mode={args.position_mode}:")
    print(f"{'Season':<10}{'n_players':>10}{'Low':>10}{'Base':>10}{'High':>10}   {'Low €':>12}{'Base €':>12}{'High €':>12}")
    for days, _, _, label in HORIZONS:
        r = general.get(days)
        if not r:
            print(f"{label:<10}{'--':>10}   no data at this horizon")
            continue
        flag = "  (thin)" if r.n_players < args.min_n else ""
        print(f"{label:<10}{r.n_players:>10}{pct(r.p25):>10}{pct(r.p50):>10}{pct(r.p75):>10}   "
              f"{eur(v0*r.p25):>12}{eur(v0*r.p50):>12}{eur(v0*r.p75):>12}{flag}")

    fee_rows = {r.category: r for r in client.query(FEE_OVERLAY_SQL, job_config=job_config).result()}
    print(f"\nTransfer-fee overlay (+1 season, exact-age cohort — does the player's own next move "
          f"involve a real fee, a free/loan move, or no transfer at all?):")
    for key, label in [("paid_fee", "Paid fee    "), ("free_or_loan", "Free/loan   "), ("no_transfer", "No transfer ")]:
        r = fee_rows.get(key)
        if not r:
            continue
        flag = "  (thin)" if r.n_players < args.min_n else ""
        print(f"  {label} n_players={r.n_players:<5} {pct(r.p25)}/{pct(r.p50)}/{pct(r.p75)}"
              f"  ({eur(v0*r.p25)}/{eur(v0*r.p50)}/{eur(v0*r.p75)}){flag}")
    print()


if __name__ == "__main__":
    main()
