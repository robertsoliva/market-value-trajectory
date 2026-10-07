#!/usr/bin/env python3
"""Plots a player's real valuation history against the cohort method's
backtest bands anchored at a past date, for the README's backtest figure."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date

# Jude Bellingham, player_id 581678 — full public Transfermarkt valuation history.
HISTORY = [
    ("2019-10-17", 2_500_000), ("2020-02-27", 12_000_000), ("2020-04-08", 11_000_000),
    ("2020-09-16", 27_000_000), ("2021-02-10", 35_000_000), ("2021-05-28", 55_000_000),
    ("2021-10-07", 70_000_000), ("2021-12-22", 75_000_000), ("2022-06-09", 80_000_000),
    ("2022-09-21", 90_000_000), ("2022-11-09", 100_000_000), ("2022-12-23", 110_000_000),
    ("2023-03-29", 120_000_000), ("2023-06-22", 120_000_000), ("2023-10-13", 150_000_000),
    ("2023-12-22", 180_000_000), ("2024-06-07", 180_000_000), ("2024-12-27", 180_000_000),
    ("2025-06-09", 180_000_000), ("2025-12-12", 160_000_000), ("2026-03-16", 140_000_000),
    ("2026-06-05", 130_000_000),
]

ANCHOR_DATE = date(2021, 10, 7)
ANCHOR_VALUE = 70_000_000
# (low, base, high) ratio bands from output/backtest_2021-10-07.txt
BANDS = {
    365: (1.0, 2.0, 4.0),
    730: (1.5, 3.0, 7.0),
    1095: (1.8, 4.0, 10.0),
}

dates = [date.fromisoformat(d) for d, _ in HISTORY]
values = [v / 1e6 for _, v in HISTORY]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates, values, color="#1f77b4", marker="o", markersize=3, label="Actual TM value")
ax.axvline(ANCHOR_DATE, color="grey", linestyle=":", linewidth=1)
ax.annotate("backtest anchor", (ANCHOR_DATE, ANCHOR_VALUE / 1e6), textcoords="offset points",
            xytext=(6, -14), fontsize=8, color="grey")

for horizon_days, (lo, base, hi) in BANDS.items():
    target_date = date.fromordinal(ANCHOR_DATE.toordinal() + horizon_days)
    ax.plot([target_date], [ANCHOR_VALUE * base / 1e6], "D", color="#d62728", markersize=5)
    ax.vlines(target_date, ANCHOR_VALUE * lo / 1e6, ANCHOR_VALUE * hi / 1e6, color="#d62728", alpha=0.5, linewidth=6)

ax.plot([], [], "D", color="#d62728", label="Cohort median prediction")
ax.plot([], [], color="#d62728", alpha=0.5, linewidth=6, label="Cohort 25th-75th band")

ax.set_ylabel("Market value (EUR millions)")
ax.set_title("Jude Bellingham — actual value vs. cohort-method backtest\n(anchored at 2021-10-07, age 18)")
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("output/backtest_chart.png", dpi=150)
print("Wrote output/backtest_chart.png")
