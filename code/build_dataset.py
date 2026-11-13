"""Step 1: parse raw PHH hands into one row per (player, hand).

Usage:  python -I code/build_dataset.py data/raw data/player_hands.parquet

We keep PartyPoker no-limit hold'em hands played at 6-max tables.  PartyPoker is the
venue whose `winnings` field is complete, so each player's net result per hand can
be computed exactly as  winnings - chips put in  (checked against stack continuity
in validate_dataset.py).

For every player in every hand we record
  * the context of their first pre-flop decision (position, raises / limpers before),
  * that decision (fold / check / call / raise) and its size relative to the pot,
  * summary flags used only for validation (VPIP, PFR, post-flop bets and calls),
  * stack, chips contributed and net result, all in big blinds.
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VENUE = "PartyPoker"
MIN_TABLE_HANDS = 200          # a table must have this many hands for its size to be trusted

# position groups
EP, CO, BTN, SB, BB, NA = 0, 1, 2, 3, 4, -1
# decision types
D_NONE, D_FOLD, D_CHECK, D_CALL, D_RAISE = -1, 0, 1, 2, 3


def position_group(i, n):
    """i = index in PHH order (0 = small blind, 1 = big blind, n-1 = button)."""
    if n < 3:
        return NA
    if i == 0:
        return SB
    if i == 1:
        return BB
    behind = n - 1 - i            # players still to act before the blinds
    return BTN if behind == 0 else CO if behind == 1 else EP


def parse_hand(actions, blinds, stacks, winnings):
    """Replay the betting of one hand.  Returns per-player tuples."""
    n = len(stacks)
    commit = [abs(b) for b in blinds]          # chips in on the current street
    total = [0.0] * n                          # chips in on earlier streets
    cur = max(commit)
    pot = sum(commit)
    nraise = nlimp = 0
    street = 0
    d_type = [D_NONE] * n
    d_nraise = [0] * n
    d_nlimp = [0] * n
    d_callr = [np.nan] * n
    d_raiser = [np.nan] * n
    vpip = [False] * n
    pfr = [False] * n
    folded = [False] * n
    saw_flop = [False] * n
    post_aggr = [0] * n
    post_call = [0] * n
    showdown = [False] * n
    for a in actions:
        p = a.split()
        if p[0] == "d":
            if p[1] == "db":
                if street == 0:
                    for i in range(n):
                        saw_flop[i] = not folded[i]
                street += 1
                for i in range(n):
                    total[i] += commit[i]
                    commit[i] = 0.0
                cur = 0.0
            continue
        i = int(p[0][1:]) - 1
        kind = p[1]
        if kind == "sm":
            showdown[i] = True
            continue
        left = stacks[i] - total[i] - commit[i]
        to_call = min(max(cur - commit[i], 0.0), left)
        first = street == 0 and d_type[i] == D_NONE
        if first:
            d_nraise[i], d_nlimp[i] = nraise, nlimp
        if kind == "f":
            folded[i] = True
            if first:
                d_type[i] = D_FOLD
        elif kind == "cc":
            if to_call > 1e-9:
                if first:
                    d_type[i] = D_CALL
                    d_callr[i] = to_call / (pot + to_call)
                if street == 0:
                    vpip[i] = True
                    if nraise == 0:
                        nlimp += 1
                else:
                    post_call[i] += 1
                commit[i] += to_call
                pot += to_call
            elif first:
                d_type[i] = D_CHECK
        elif kind == "cbr":
            amt = float(p[2])
            if first:
                d_type[i] = D_RAISE
                d_raiser[i] = (amt - cur) / (pot + to_call) if pot + to_call > 0 else np.nan
            if street == 0:
                vpip[i] = True
                pfr[i] = True
                nraise += 1
            else:
                post_aggr[i] += 1
            pot += amt - commit[i]
            commit[i] = amt
            cur = max(cur, amt)
    contrib = [total[i] + commit[i] for i in range(n)]
    return d_type, d_nraise, d_nlimp, d_callr, d_raiser, vpip, pfr, saw_flop, post_aggr, post_call, showdown, contrib, pot


def table_sizes(path):
    t = pq.read_table(path, columns=["venue", "source_file", "table", "players"]).to_pandas()
    t = t[t.venue == VENUE]
    if len(t) == 0:
        return None
    t["stake"] = t.source_file.map(lambda s: s.split("/")[1])
    t["n"] = t.players.map(len)
    return t.groupby(["stake", "table"]).n.agg(["max", "size"]).reset_index()


def parse_shard(args):
    path, keep = args
    cols = ["venue", "source_file", "table", "time", "day", "month", "hand", "actions", "players",
            "starting_stacks", "winnings", "blinds_or_straddles"]
    t = pq.read_table(path, columns=cols).to_pandas()
    t = t[t.venue == VENUE]
    out = {k: [] for k in ["stake", "table", "player", "tsec", "hand", "bb", "n", "pos", "posgrp", "d_type",
                            "d_nraise", "d_nlimp", "d_callr", "d_raiser", "vpip", "pfr", "saw_flop", "post_aggr",
                            "post_call", "showdown", "stack_bb", "contrib_bb", "net_bb", "pot_bb"]}
    bad = 0
    for r in t.itertuples(index=False):
        stake = r.source_file.split("/")[1]
        if (stake, r.table) not in keep:
            continue
        players = list(r.players)
        n = len(players)
        blinds = [float(b) for b in r.blinds_or_straddles]
        stacks = [float(s) for s in r.starting_stacks]
        win = [float(w) for w in r.winnings]
        if n < 2 or len(blinds) != n or len(win) != n or sum(win) <= 0 or r.day != r.day:
            bad += 1
            continue
        bb = max(abs(b) for b in blinds)
        try:
            res = parse_hand(list(r.actions), blinds, stacks, win)
        except Exception:
            bad += 1
            continue
        d_type, d_nraise, d_nlimp, d_callr, d_raiser, vpip, pfr, saw_flop, post_aggr, post_call, showdown, contrib, pot = res
        h, m, s = r.time.split(":")
        tsec = int(((r.month - 7) * 31 + r.day) * 86400 + int(h) * 3600 + int(m) * 60 + int(s))
        # players who post a blind out of turn get no position label (their decision is treated as missing)
        odd = [i >= 2 and abs(blinds[i]) > 0 for i in range(n)]
        for i in range(n):
            out["stake"].append(stake); out["table"].append(r.table); out["player"].append(players[i])
            out["tsec"].append(tsec); out["hand"].append(int(r.hand)); out["bb"].append(bb); out["n"].append(n)
            out["pos"].append(i); out["posgrp"].append(NA if odd[i] else position_group(i, n))
            out["d_type"].append(d_type[i]); out["d_nraise"].append(d_nraise[i]); out["d_nlimp"].append(d_nlimp[i])
            out["d_callr"].append(d_callr[i]); out["d_raiser"].append(d_raiser[i])
            out["vpip"].append(vpip[i]); out["pfr"].append(pfr[i]); out["saw_flop"].append(saw_flop[i])
            out["post_aggr"].append(post_aggr[i]); out["post_call"].append(post_call[i]); out["showdown"].append(showdown[i])
            out["stack_bb"].append(stacks[i] / bb); out["contrib_bb"].append(contrib[i] / bb)
            out["net_bb"].append((win[i] - contrib[i]) / bb); out["pot_bb"].append(pot / bb)
    df = pd.DataFrame(out)
    small = {"n": "int8", "pos": "int8", "posgrp": "int8", "d_type": "int8", "d_nraise": "int8", "d_nlimp": "int8",
             "post_aggr": "int8", "post_call": "int8", "d_callr": "float32", "d_raiser": "float32",
             "stack_bb": "float32", "contrib_bb": "float32", "net_bb": "float32", "pot_bb": "float32", "bb": "float32"}
    df = df.astype(small)
    return df, bad, len(t)


def main(raw_dir, out_path):
    shards = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    with Pool(8) as pool:
        sizes = [s for s in pool.map(table_sizes, shards) if s is not None]
        sizes = pd.concat(sizes).groupby(["stake", "table"]).agg(max=("max", "max"), size=("size", "sum")).reset_index()
        six = sizes[(sizes["max"] >= 3) & (sizes["max"] <= 6) & (sizes["size"] >= MIN_TABLE_HANDS)]
        print(f"{VENUE}: {len(sizes)} tables, {sizes['size'].sum():,} hands; "
              f"6-max tables kept: {len(six)}, {six['size'].sum():,} hands")
        print("table size (max players seen) -> hands:", sizes.groupby("max")["size"].sum().to_dict())
        keep = set(zip(six.stake, six.table))
        res = pool.map(parse_shard, [(s, keep) for s in shards])
    df = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
    print(f"hands dropped as unparseable / no winnings: {sum(r[1] for r in res):,}")
    for c in ["stake", "table", "player"]:
        df[c] = df[c].astype("category")
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df):,} player-hand rows, {df.hand.nunique():,} hands, {df.player.nunique():,} players -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
