"""Step 2: turn the player-hand table into sequences for the HMM.

Usage:  python -I code/prepare_sequences.py data/player_hands.parquet data/sequences.npz

A sequence is one playing session of one player: all of that player's hands, over
every table they sit at, ordered by start time and cut wherever they are away for
more than 30 minutes.  We keep sessions of at least 100 hands, and players with at
least 500 hands in such sessions.

Each hand becomes
  ctx  context of the first pre-flop decision: position group (5) x {no raise yet, raised}
  x3   coarse action:  0 fold, 1 check/call, 2 raise
  x8   fine action:    0 fold, 1 check, 2-4 call (small/medium/large price), 5-7 raise (small/medium/large)
  (-1 = the player had no decision, e.g. everyone folded to their big blind)
Net results are stored next to the sequences but are never an emission.
"""
import sys

import numpy as np
import pandas as pd

SESSION_GAP = 30 * 60
MIN_SESSION = 100
MIN_PLAYER_HANDS = 500
TRAIN_HANDS_PER_PLAYER = 2000     # a training player contributes their earliest sessions up to this many hands
MAX_TRAIN_SESSION = 1000          # longer training sessions are truncated
SEED = 150


def main(src, dst):
    cols = ["stake", "player", "tsec", "hand", "bb", "n", "posgrp", "d_type", "d_nraise", "d_nlimp", "d_callr",
            "d_raiser", "vpip", "pfr", "saw_flop", "post_aggr", "post_call", "showdown", "stack_bb", "net_bb", "table"]
    df = pd.read_parquet(src, columns=cols).sort_values(["player", "tsec", "hand"], kind="stable").reset_index(drop=True)
    # A hand's end time is not recorded; the start of the next hand at the same table bounds it.
    # Multi-tabling players start other hands while a big pot is still being played, so
    # "after the loss" must mean "started after the loss hand ended".
    th = df.drop_duplicates("hand")[["stake", "table", "tsec", "hand"]].sort_values(["stake", "table", "tsec", "hand"])
    nxt = th.groupby(["stake", "table"], observed=True).tsec.shift(-1)
    hand_end = pd.Series(np.where(nxt.notna() & (nxt - th.tsec <= 600), nxt, th.tsec + 60).astype(np.int64), index=th.hand.values)
    pl = df.player.cat.codes.values
    newp = np.r_[True, pl[1:] != pl[:-1]]
    gap = np.r_[0, np.diff(df.tsec.values)]
    df["sid"] = np.cumsum(newp | (gap > SESSION_GAP))
    slen = df.groupby("sid").size()
    df = df[df.sid.map(slen) >= MIN_SESSION]
    ph = df.groupby("player", observed=True).size()
    df = df[df.player.map(ph).astype(float) >= MIN_PLAYER_HANDS].reset_index(drop=True)
    print(f"kept {len(df):,} player-hands, {df.sid.nunique():,} sessions, {df.player.nunique():,} players")
    df["tend"] = df.hand.map(hand_end).values

    # ---- sessions
    first = np.r_[True, df.sid.values[1:] != df.sid.values[:-1]]
    start = np.flatnonzero(first)
    length = np.diff(np.r_[start, len(df)])
    s_player_name = df.player.values[start]
    names, s_player = np.unique(np.asarray(s_player_name), return_inverse=True)
    P = len(names)
    low = (df.bb.values < 1.0)                       # 25NL vs 400-1000NL
    s_low = np.array([low[a:a + L].mean() > 0.5 for a, L in zip(start, length)])
    s_tables = np.array([len(np.unique(df.table.cat.codes.values[a:a + L])) for a, L in zip(start, length)])

    # ---- split: half of the players are used for fitting, half are never seen in training
    rng = np.random.default_rng(SEED)
    fit_player = np.zeros(P, bool)
    fit_player[rng.permutation(P)[: P // 2]] = True
    split = np.full(len(start), 2, dtype=np.int8)     # 0 train, 1 later sessions of training players, 2 new players
    train_len = np.zeros(len(start), dtype=np.int64)
    used = np.zeros(P, dtype=np.int64)
    for s in range(len(start)):                       # sessions are in time order within a player
        p = s_player[s]
        if not fit_player[p]:
            continue
        if used[p] < TRAIN_HANDS_PER_PLAYER:
            split[s] = 0
            train_len[s] = min(length[s], MAX_TRAIN_SESSION)
            used[p] += train_len[s]
        else:
            split[s] = 1
    row_split = np.repeat(split, length)
    row_train = np.zeros(len(df), bool)
    for s in np.flatnonzero(split == 0):
        row_train[start[s]: start[s] + train_len[s]] = True

    # ---- symbols
    d = df.d_type.values
    valid = (df.posgrp.values >= 0) & (d >= 0)
    ctx = np.where(valid, df.posgrp.values * 2 + (df.d_nraise.values >= 1), 0).astype(np.int8)
    x3 = np.where(valid, np.select([d == 0, (d == 1) | (d == 2), d == 3], [0, 1, 2], -1), -1).astype(np.int8)
    cr, rr = df.d_callr.values, df.d_raiser.values
    cq = np.quantile(cr[row_train & valid & (d == 2)], [1 / 3, 2 / 3])
    rq = np.quantile(rr[row_train & valid & (d == 3)], [1 / 3, 2 / 3])
    print("call-price terciles (to_call / pot after call):", cq.round(3), " raise-size terciles (raise / pot):", rq.round(3))
    x8 = np.full(len(df), -1, dtype=np.int8)
    x8[valid & (d == 0)] = 0
    x8[valid & (d == 1)] = 1
    m = valid & (d == 2)
    x8[m] = 2 + np.searchsorted(cq, cr[m], side="right")
    m = valid & (d == 3)
    x8[m] = 5 + np.searchsorted(rq, rr[m], side="right")
    print("x8 distribution:", (np.bincount(x8[valid], minlength=8) / valid.sum()).round(4))
    print("missing decisions: %.4f" % (1 - valid.mean()))

    np.savez_compressed(
        dst, start=start, length=length, s_player=s_player, s_low=s_low, s_tables=s_tables, split=split,
        train_len=train_len, fit_player=fit_player, call_q=cq, raise_q=rq,
        ctx=ctx, x3=x3, x8=x8, net=df.net_bb.values.astype(np.float32), stack=df.stack_bb.values.astype(np.float32),
        vpip=df.vpip.values, pfr=df.pfr.values, saw_flop=df.saw_flop.values, showdown=df.showdown.values,
        post_aggr=df.post_aggr.values, post_call=df.post_call.values, nraise=df.d_nraise.values,
        tsec=df.tsec.values.astype(np.int64), tend=df.tend.values.astype(np.int64), low=low, pos=df.posgrp.values)
    for k, name in enumerate(["train", "later sessions of training players", "new players"]):
        sel = split == k
        hands = train_len[sel].sum() if k == 0 else length[sel].sum()
        print(f"{name}: {sel.sum():,} sessions, {hands:,} hands, {len(np.unique(s_player[sel])):,} players")
    print("sessions: median length %d, median tables %d, low-stake share %.3f" % (np.median(length), np.median(s_tables), s_low.mean()))
    print("players per stake group: low %d, high %d" % (len(np.unique(s_player[s_low])), len(np.unique(s_player[~s_low]))))
    net = df.net_bb.values
    print("big losses (<= -50bb): %d, big wins (>= 50bb): %d" % ((net <= -50).sum(), (net >= 50).sum()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
