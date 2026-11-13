"""Which sites record winnings?  Prints, per site and stake, the number of hands and the
share whose `winnings` field is empty or all zero.  This is why the project uses PartyPoker.
Usage: python -I code/survey_sites.py data/raw"""
import glob
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

rows = []
for f in sorted(glob.glob(sys.argv[1] + "/*.parquet")):
    t = pq.read_table(f, columns=["venue", "min_bet", "winnings"]).to_pandas()
    t["missing"] = t.winnings.map(lambda w: float(np.sum(w)) == 0)
    t["venue"] = t.venue.fillna("(no venue: Pluribus and others)")
    rows.append(t.groupby(["venue", "min_bet"], dropna=False).missing.agg(["size", "sum"]).reset_index())
a = pd.concat(rows).groupby(["venue", "min_bet"], dropna=False)[["size", "sum"]].sum().reset_index()
a["share_missing"] = (a["sum"] / a["size"]).round(3)
print(a[a["size"] > 2000].rename(columns={"size": "hands"}).drop(columns="sum").to_string(index=False))
v = a.groupby("venue")[["size", "sum"]].sum()
print((v["sum"] / v["size"]).round(3).rename("share of hands with no winnings").to_string())
