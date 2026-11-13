"""Experiment 3: what happens in the hands after a big loss?

Event study on held-out sessions.  An event is a hand in which the player loses at
least 50 big blinds.  We compare the 30 hands before the event hand with the first
30 hands the player starts after it has ended.  Controls: big wins (same selection:
the player chose to play a big pot) and neutral flops (saw a flop, small result).

Outcomes
  * model-free: VPIP and PFR (share of hands played / raised pre-flop)
  * HMM: filtered posterior P(z_t = k | actions up to t) of every state, and the
    posterior-expected VPIP ("looseness index")
Uncertainty: bootstrap over players (events of one player are not independent).

Usage: python code/exp_tilt.py <model tag>
"""
from common import *
from data import find_events, BIG, CTX

PRE, POST, PLOT_POST, REPS = 30, 30, 50, 2000


def windows(y, ev, post=PLOT_POST):
    pre = ev["i"][:, None] + np.arange(-PRE, 0)[None]
    pst = ev["j"][:, None] + np.arange(0, post)[None]
    return np.concatenate([y[pre], y[pst]], axis=1)


def effect(w):
    """post (first 30 hands after) minus pre (30 hands before), one number per event."""
    return np.nanmean(w[:, PRE:PRE + POST], axis=1) - np.nanmean(w[:, :PRE], axis=1)


def boot(groups, P, rng, reps=REPS):
    """groups: list of (per-event effect, per-event player).  Returns mean and 95% CI of each
    group's mean effect and of the differences group0 - group_k, resampling players."""
    sums = [np.bincount(pl, weights=d, minlength=P) for d, pl in groups]
    cnts = [np.bincount(pl, minlength=P).astype(float) for d, pl in groups]
    w = rng.multinomial(P, np.ones(P) / P, size=reps).astype(float)      # (reps, P) resampling weights
    means = np.stack([(w @ s) / np.maximum(w @ c, 1) for s, c in zip(sums, cnts)])   # (G, reps)
    pt = np.array([s.sum() / c.sum() for s, c in zip(sums, cnts)])
    res = {"mean": pt, "ci": np.quantile(means, [0.025, 0.975], axis=1).T}
    res["diff"] = pt[0] - pt[1:]
    dd = means[0][None] - means[1:]
    res["diff_ci"] = np.quantile(dd, [0.025, 0.975], axis=1).T
    res["diff_p"] = np.array([2 * min((x <= 0).mean(), (x >= 0).mean()) for x in dd])
    return res


def main(tag):
    S = seqs()
    rng = np.random.default_rng(7)
    m = load_model(tag)
    K = m.K
    held = np.flatnonzero(S.split >= 1)
    held_row = np.repeat(S.split >= 1, S.length)
    P = int(S.s_player.max()) + 1

    # ---- outcome series
    ok = S.x8 >= 0
    w_ctx = np.bincount(S.ctx[ok], minlength=10) / ok.sum()
    vp_state = ((1 - m.B[:, :, 0] - m.B[:, :, 1]) * w_ctx[:, None]).sum(0)         # P(call or raise | state)
    d_all = S.seqdata(np.arange(S.S), "x8", True)
    filt = np.concatenate(m.filtered(d_all)).astype(np.float32)
    smooth = np.concatenate(m.posteriors(d_all)).astype(np.float32)
    series = {"vpip": S.vpip.astype(np.float32), "pfr": S.pfr.astype(np.float32),
              "loose_index": filt @ vp_state.astype(np.float32)}
    for k in range(K):
        series[f"filt{k}"] = filt[:, k]
        series[f"smooth{k}"] = smooth[:, k]
    np.save(os.path.join(RES, "posterior_filtered.npy"), filt)
    np.save(os.path.join(RES, "posterior_smoothed.npy"), smooth)

    out = {"tag": tag, "K": K, "vpip_state": vp_state, "offsets": np.r_[np.arange(-PRE, 0), np.arange(1, PLOT_POST + 1)]}

    def study(thr, sel_sessions=None, label="main", curves=False):
        evs = {}
        for kind in ("loss", "win", "neutral"):
            ev = find_events(S, thr, PRE, PLOT_POST, kind, rng=np.random.default_rng(11))
            keep = held_row[ev["i"]]
            if sel_sessions is not None:
                keep &= sel_sessions[ev["s"]]
            evs[kind] = {k: v[keep] for k, v in ev.items()}
        r = {"n": {k: len(v["i"]) for k, v in evs.items()},
             "players": {k: len(np.unique(S.s_player[v["s"]])) for k, v in evs.items()}}
        for name, y in series.items():
            ws = {k: windows(y, v) for k, v in evs.items()}
            groups = [(effect(ws[k]), S.s_player[evs[k]["s"]]) for k in ("loss", "win", "neutral")]
            b = boot(groups, P, rng)
            r[name] = {"pre": [float(np.nanmean(ws[k][:, :PRE])) for k in ("loss", "win", "neutral")],
                       "effect": b["mean"], "effect_ci": b["ci"], "loss_minus": b["diff"], "loss_minus_ci": b["diff_ci"],
                       "loss_minus_p": b["diff_p"]}
            if curves:
                r[name]["curve"] = {k: np.nanmean(ws[k], axis=0) for k in ws}
        out[label] = r
        return evs

    evs = study(BIG, curves=True)
    study(30.0, label="thr30")
    study(100.0, label="thr100")
    study(BIG, S.s_low, "low_stakes")
    study(BIG, ~S.s_low, "high_stakes")
    study(BIG, S.s_tables <= 2, "tables_1_2")
    study(BIG, S.s_tables >= 3, "tables_3plus")
    hands_p = np.bincount(S.s_player, weights=S.length, minlength=P)
    study(BIG, hands_p[S.s_player] < 3000, "players_lt3000")
    study(BIG, hands_p[S.s_player] >= 3000, "players_ge3000")

    # ---- do players leave after a big loss?  (no post-window requirement here)
    quit = {}
    for kind in ("loss", "win", "neutral"):
        ev = find_events(S, BIG, PRE, 0, kind, need_post=False, rng=np.random.default_rng(11))
        keep = held_row[ev["i"]]
        left = (S.start[ev["s"]] + S.length[ev["s"]] - ev["j"])[keep]
        quit[kind] = {"n": int(keep.sum()), "left_within_30": float((left < 30).mean()), "left_within_10": float((left < 10).mean()),
                      "overlap_hands": float((ev["j"] - ev["i"] - 1)[keep].mean())}
    out["quit"] = quit

    # ---- per-player tilt: is the spread of player-level effects larger than chance?
    y = series["vpip"]
    per = {}
    for kind in ("loss", "win"):
        ev = evs[kind]
        d = effect(windows(y, ev)); pl = S.s_player[ev["s"]]
        n = np.bincount(pl, minlength=P); mu = np.bincount(pl, weights=d, minlength=P) / np.maximum(n, 1)
        sel = n >= 5
        # chance spread: within-player sd of d / sqrt(n)
        ss = np.bincount(pl, weights=d * d, minlength=P) / np.maximum(n, 1) - mu ** 2
        se = np.sqrt(np.maximum(ss, 0) / np.maximum(n - 1, 1))
        zz = (mu[sel] - mu[sel].mean()) / np.maximum(se[sel], 1e-9)
        per[kind] = {"players": int(sel.sum()), "mean": mu[sel], "n": n[sel], "se": se[sel],
                     "sd_observed": float(mu[sel].std()), "sd_expected_noise": float(np.sqrt((se[sel] ** 2).mean())),
                     "share_z_gt2": float((zz > 2).mean()), "share_z_lt_minus2": float((zz < -2).mean())}
    out["per_player"] = per
    dump(out, "tilt.json")
    r = out["main"]
    print("events", r["n"], "players", r["players"])
    for name in ["vpip", "pfr", "loose_index"] + [f"filt{k}" for k in range(K)]:
        e = r[name]
        print(f"{name:12s} pre {np.round(e['pre'], 4)} effect {np.round(e['effect'], 4)}  loss-win {e['loss_minus'][0]:+.4f} "
              f"[{e['loss_minus_ci'][0][0]:+.4f},{e['loss_minus_ci'][0][1]:+.4f}] p={e['loss_minus_p'][0]:.3f}   loss-neutral {e['loss_minus'][1]:+.4f} "
              f"[{e['loss_minus_ci'][1][0]:+.4f},{e['loss_minus_ci'][1][1]:+.4f}]")
    print("quit", quit)
    for lab in ["thr30", "thr100", "low_stakes", "high_stakes", "tables_1_2", "tables_3plus", "players_lt3000", "players_ge3000"]:
        e = out[lab]["vpip"]; print(f"{lab:16s} n={out[lab]['n']['loss']:6d} VPIP effect after loss {e['effect'][0]:+.4f} [{e['effect_ci'][0][0]:+.4f},{e['effect_ci'][0][1]:+.4f}]  loss-win {e['loss_minus'][0]:+.4f} [{e['loss_minus_ci'][0][0]:+.4f},{e['loss_minus_ci'][0][1]:+.4f}]")
    print("per-player", {k: {a: b for a, b in v.items() if a not in ("mean", "n", "se")} for k, v in per.items()})


if __name__ == "__main__":
    main(sys.argv[1])
