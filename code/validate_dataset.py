"""Step 2 (checks only): validate the parsed table and print the numbers quoted in the report."""
import sys
import numpy as np, pandas as pd
df = pd.read_parquet(sys.argv[1])
print("rows", len(df), "hands", df.hand.nunique(), "players", df.player.nunique(), "tables", df.groupby(["stake", "table"], observed=True).ngroups)
print("hands per stake:", df.drop_duplicates("hand").stake.value_counts().to_dict())
print("players dealt in:", df.drop_duplicates("hand").n.value_counts(normalize=True).sort_index().round(3).to_dict())
print("duplicate (hand, player) rows:", int(df.duplicated(["hand", "player"]).sum()))

# 1. zero-sum check: net results of a hand sum to minus the rake
s = df.groupby("hand").net_bb.sum()
print("sum of nets per hand (bb): mean %.3f, share in [-4, 0.01]: %.4f" % (s.mean(), ((s <= 0.01) & (s >= -4)).mean()))

# 2. stack continuity: next starting stack at the same table = stack + net
d = df.sort_values(["stake", "table", "player", "tsec", "hand"])
same = (d.player.values[1:] == d.player.values[:-1]) & (d.table.values[1:] == d.table.values[:-1]) & (d.stake.values[1:] == d.stake.values[:-1])
gap = d.tsec.values[1:] - d.tsec.values[:-1]
pred = d.stack_bb.values[:-1] + d.net_bb.values[:-1]
nxt = d.stack_bb.values[1:]
m = same & (gap < 300)
err = nxt[m] - pred[m]
print("consecutive same-table pairs: %d; |next stack - (stack+net)| < 0.02bb: %.4f; next > pred (top-up/rebuy): %.4f; next < pred: %.4f"
      % (m.sum(), (np.abs(err) < 0.02).mean(), (err >= 0.02).mean(), (err <= -0.02).mean()))

# 3. decisions
print("decision type share:", df.d_type.value_counts(normalize=True).sort_index().round(4).to_dict())
print("posgrp share:", df.posgrp.value_counts(normalize=True).sort_index().round(4).to_dict())
print("VPIP %.3f PFR %.3f" % (df.vpip.mean(), df.pfr.mean()))
print("stack_bb quantiles", df.stack_bb.quantile([.05, .25, .5, .75, .95]).round(1).to_dict())
for thr in (30, 50, 100):
    print(f"hands with loss >= {thr}bb: {(df.net_bb <= -thr).mean():.5f}   win >= {thr}bb: {(df.net_bb >= thr).mean():.5f}")

# 4. players and sessions (per-player timeline over all tables, split at 30-minute gaps)
d = df.sort_values(["player", "tsec", "hand"])
newp = np.r_[True, d.player.values[1:] != d.player.values[:-1]]
gap = np.r_[0, np.diff(d.tsec.values)]
news = newp | (gap > 1800)
sid = np.cumsum(news)
sl = pd.Series(sid).value_counts()
pc = d.player.value_counts()
print("players: >=200 hands %d, >=500 %d, >=1000 %d, >=5000 %d" % ((pc >= 200).sum(), (pc >= 500).sum(), (pc >= 1000).sum(), (pc >= 5000).sum()))
print("sessions %d; length quantiles %s; sessions >=100 hands: %d holding %.3f of rows; >=200: %d holding %.3f"
      % (len(sl), sl.quantile([.5, .9, .99]).to_dict(), (sl >= 100).sum(), sl[sl >= 100].sum() / len(d), (sl >= 200).sum(), sl[sl >= 200].sum() / len(d)))
# multi-tabling: distinct tables per session
nt = pd.DataFrame({"sid": sid, "table": d.table.cat.codes.values}).groupby("sid").table.nunique()
big = sl.index[sl >= 100]
print("tables per session (sessions >=100 hands): quantiles", nt.loc[big].quantile([.25, .5, .75, .9, .99]).to_dict(), "share single-table: %.3f" % (nt.loc[big] == 1).mean())
dur = pd.DataFrame({"sid": sid, "t": d.tsec.values}).groupby("sid").t.agg(lambda x: x.max() - x.min())
print("session duration minutes (>=100 hands): median %.0f; hands per hour median %.0f" % (dur.loc[big].median() / 60, (sl.loc[big] / (dur.loc[big] / 3600)).median()))
