"""Experiment 1b: what are the states of the pooled HMM?

On held-out players (never used for fitting):
  * each state's implied pre-flop statistics, read off the emission matrix
  * the players who live mostly in each state: VPIP, PFR, post-flop aggression factor,
    win rate, volume.  Aggression factor and win rate are never shown to the model.
  * how often the decoded state changes inside a session (Viterbi)

Usage: python code/exp_validate.py <model tag>     (run exp_tilt.py first: it saves the posteriors)
"""
from common import *
from data import CTX, SYM8


def main(tag):
    S = seqs()
    m = load_model(tag)
    K = m.K
    ok = S.x8 >= 0
    w = np.bincount(S.ctx[ok], minlength=10) / ok.sum()
    B = m.B
    out = {"tag": tag, "ctx_weights": w, "pi": m.pi, "A": m.A[0], "B": B,
           "dwell": 1 / (1 - np.diag(m.A[0]))}
    ev, vec = np.linalg.eig(m.A[0].T)
    st = np.real(vec[:, np.argmax(np.real(ev))]); out["stationary"] = st / st.sum()
    # implied statistics.  VPIP = call or raise; "limp" = call when nobody has raised (not from the BB)
    nr = np.array([0, 2, 4, 6])                 # contexts: EP/CO/BTN/SB with no raise yet
    rs = np.array([1, 3, 5, 7, 9])              # contexts facing a raise
    out["state_vpip"] = ((B[:, :, 2:].sum(2)) * w[:, None]).sum(0)
    out["state_pfr"] = ((B[:, :, 5:].sum(2)) * w[:, None]).sum(0)
    out["state_limp"] = (B[nr][:, :, 2:5].sum(2) * w[nr, None]).sum(0) / w[nr].sum()
    out["state_open"] = (B[nr][:, :, 5:].sum(2) * w[nr, None]).sum(0) / w[nr].sum()
    out["state_open_size"] = (B[nr][:, :, 5:] * w[nr, None, None]).sum(0) / (B[nr][:, :, 5:].sum(2) * w[nr, None]).sum(0)[:, None]
    out["state_coldcall"] = (B[rs][:, :, 2:5].sum(2) * w[rs, None]).sum(0) / w[rs].sum()
    out["state_3bet"] = (B[rs][:, :, 5:].sum(2) * w[rs, None]).sum(0) / w[rs].sum()

    smooth = np.load(os.path.join(RES, "posterior_smoothed.npy"))
    sess = np.repeat(np.arange(S.S), S.length)
    new = np.flatnonzero(S.split == 2)
    rows_new = np.repeat(S.split == 2, S.length)
    pl = S.s_player[sess]
    P = int(S.s_player.max()) + 1

    def per_player(v, rows):
        return np.bincount(pl[rows], weights=v[rows], minlength=P)

    n = per_player(np.ones(len(pl)), rows_new)
    sel = n > 0
    occ = np.stack([per_player(smooth[:, k].astype(float), rows_new) for k in range(K)], 1)[sel] / n[sel, None]
    stats = dict(
        vpip=per_player(S.vpip.astype(float), rows_new)[sel] / n[sel],
        pfr=per_player(S.pfr.astype(float), rows_new)[sel] / n[sel],
        af=per_player(S.post_aggr.astype(float), rows_new)[sel] / np.maximum(per_player(S.post_call.astype(float), rows_new)[sel], 1),
        winrate=100 * per_player(S.net.astype(float), rows_new)[sel] / n[sel],
        hands=n[sel], low=per_player(S.low.astype(float), rows_new)[sel] / n[sel],
        stack=per_player(S.stack.astype(float), rows_new)[sel] / n[sel])
    dom = occ.argmax(1)
    out["players"] = int(sel.sum())
    out["occ_max_mean"] = float(occ.max(1).mean())
    out["occ_max_gt90"] = float((occ.max(1) > 0.9).mean())
    rows = []
    rng = np.random.default_rng(0)
    for k in range(K):
        g = dom == k
        hands_k = stats["hands"][g]
        # win rate pooled over hands (bb/100) with a bootstrap over players
        tot = stats["winrate"][g] * hands_k / 100
        bs = [tot[i].sum() / hands_k[i].sum() * 100 for i in (rng.integers(0, g.sum(), g.sum()) for _ in range(2000))]
        rows.append(dict(state=k, players=int(g.sum()), share_hands=float(hands_k.sum() / stats["hands"].sum()),
                         vpip=float(np.average(stats["vpip"][g], weights=hands_k)), pfr=float(np.average(stats["pfr"][g], weights=hands_k)),
                         af=float(np.median(stats["af"][g])), winrate=float(tot.sum() / hands_k.sum() * 100),
                         winrate_ci=np.quantile(bs, [0.025, 0.975]), median_hands=float(np.median(hands_k)),
                         low=float(np.average(stats["low"][g], weights=hands_k)), stack=float(np.average(stats["stack"][g], weights=hands_k))))
        print(rows[-1])
    out["by_state"] = rows
    out["corr_occ_vs_stats"] = {name: [float(np.corrcoef(occ[:, k], v)[0, 1]) for k in range(K)] for name, v in stats.items()}
    out["player_occ"] = occ; out["player_vpip"] = stats["vpip"]; out["player_pfr"] = stats["pfr"]; out["player_hands"] = stats["hands"]

    # switching inside sessions
    sw, const, agree, tot = 0, 0, 0, 0
    examples = {}
    for s in new:
        r = S.rows(s)
        path, _ = m.viterbi(S.x8[r].astype(int), S.ctx[r].astype(int))
        c = int((np.diff(path) != 0).sum())
        sw += c; const += c == 0; tot += len(path)
        agree += int((path == smooth[r].argmax(1)).sum())
        if c >= 2 and len(path) >= 400 and len(examples) < 3:
            examples[int(s)] = dict(path=path, post=smooth[r], net=S.net[r], vpip=S.vpip[r])
    out["switches_per_1000"] = 1000 * sw / tot
    out["sessions_no_switch"] = const / len(new)
    out["viterbi_posterior_agreement"] = agree / tot
    out["examples"] = examples
    print("occupancy of dominant state: mean %.3f, share of players > 0.9: %.3f" % (out["occ_max_mean"], out["occ_max_gt90"]))
    print("switches per 1000 hands %.2f; sessions with no switch %.3f; Viterbi = argmax posterior on %.4f of hands" % (
        out["switches_per_1000"], out["sessions_no_switch"], out["viterbi_posterior_agreement"]))
    for k in ["state_vpip", "state_pfr", "state_limp", "state_open", "state_coldcall", "state_3bet", "dwell", "stationary"]:
        print(k, np.round(out[k], 3))
    print("open size S/M/L\n", np.round(out["state_open_size"], 3))
    print("corr", {k: np.round(v, 2) for k, v in out["corr_occ_vs_stats"].items()})
    dump(out, "validate.json")


if __name__ == "__main__":
    main(sys.argv[1])
