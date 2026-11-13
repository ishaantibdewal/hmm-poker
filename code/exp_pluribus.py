"""Experiment 5: negative control with the Pluribus bot.

Pluribus plays a fixed strategy, so a correct method should find NO hidden modes in
its play.  For Pluribus, for the human professionals it played, and for PartyPoker
players with a similar number of hands, we fit personal HMMs with K = 1..4 and
compare 5-fold cross-validated log-likelihood (folds are whole sessions).
We also decode Pluribus with the pooled PartyPoker model.

Usage: python -I code/exp_pluribus.py <model tag> data/raw/train-00000-of-00037.parquet
"""
from common import *
import pandas as pd
import pyarrow.parquet as pq
from build_dataset import parse_hand, position_group
from hmm import SeqData

KS, FOLDS, CHUNK = [1, 2, 3, 4], 5, 110


def pluribus_sequences(path, S):
    t = pq.read_table(path).to_pandas()
    t = t[(t.source_type == "phh") & (t.min_bet == 100) & t.source_file.str.startswith("pluribus/")]
    t = t.assign(sess=t.source_file.map(lambda s: s.split("/")[1]), num=t.hand.astype(int)).sort_values(["sess", "num"])
    seqs_ = {}
    for r in t.itertuples(index=False):
        players = list(r.players); n = len(players)
        stacks = [float(x) for x in r.starting_stacks]; fin = [float(x) for x in r.finishing_stacks]
        res = parse_hand(list(r.actions), [float(b) for b in r.blinds_or_straddles], stacks, None)
        d_type, d_nraise, _, d_callr, d_raiser, vpip = res[0], res[1], res[2], res[3], res[4], res[5]
        for i, p in enumerate(players):
            g = position_group(i, n); d = d_type[i]
            if d < 0:
                x = -1
            elif d == 0:
                x = 0
            elif d == 1:
                x = 1
            elif d == 2:
                x = 2 + int(np.searchsorted(S.call_q, d_callr[i], side="right"))
            else:
                x = 5 + int(np.searchsorted(S.raise_q, d_raiser[i], side="right"))
            seqs_.setdefault(p, {}).setdefault(r.sess, []).append((x, g * 2 + (d_nraise[i] >= 1), (fin[i] - stacks[i]) / 100.0, vpip[i]))
    return {p: [np.array(v) for v in d.values()] for p, d in seqs_.items()}


def cv_job(args):
    name, xs, cs, K = args
    rng = np.random.default_rng(0)
    fold = rng.permutation(len(xs)) % FOLDS
    tot, obs = 0.0, 0
    for f in range(FOLDS):
        tr = [i for i in range(len(xs)) if fold[i] != f]; te = [i for i in range(len(xs)) if fold[i] == f]
        d = SeqData([xs[i] for i in tr], [cs[i] for i in tr])
        best = None
        for seed in range(3 if K > 1 else 1):
            m = HMM(K, 8, C=10, seed=seed, pseudo=0.5)
            m.init_emissions_from(d, np.random.default_rng(seed))
            m.fit(d, n_iter=300, tol=1e-6)
            if best is None or m.history[-1] > best.history[-1]:
                best = m
        dt = SeqData([xs[i] for i in te], [cs[i] for i in te])
        tot += best.score(dt); obs += dt.n_obs
    return dict(name=name, K=K, cv_ll=tot / obs, obs=obs)


def main(tag, path):
    S = seqs()
    P = pluribus_sequences(path, S)
    subjects = {}
    for p, ss in P.items():
        n = sum(len(s) for s in ss)
        if n >= 4000:
            subjects[("bot: " if p == "Pluribus" else "pro: ") + p] = ([s[:, 0].astype(int) for s in ss], [s[:, 1].astype(int) for s in ss])
    print({k: sum(len(x) for x in v[0]) for k, v in subjects.items()})
    # PartyPoker players with a similar amount of data, sessions cut to Pluribus-like length
    hands = np.bincount(S.s_player, weights=S.length)
    rng = np.random.default_rng(3)
    cand = rng.permutation(np.flatnonzero((hands >= 10000) & (hands <= 30000)))[:40]
    for p in cand:
        xs, cs, left = [], [], 10000
        for s in np.flatnonzero(S.s_player == p):
            r = S.rows(s); x, c = S.x8[r].astype(int), S.ctx[r].astype(int)
            for a in range(0, len(x), CHUNK):
                if left <= 0:
                    break
                xs.append(x[a:a + CHUNK][:left]); cs.append(c[a:a + CHUNK][:left]); left -= len(xs[-1])
        subjects[f"party: {p}"] = (xs, cs)
    res = pmap(cv_job, [(name, xs, cs, K) for name, (xs, cs) in subjects.items() for K in KS])
    out = {"cv": res}
    table = {}
    for r in res:
        table.setdefault(r["name"], {})[r["K"]] = r["cv_ll"]
    gains = {n: max(v[k] for k in KS[1:]) - v[1] for n, v in table.items()}
    out["gain"] = gains
    for n, g in gains.items():
        if not n.startswith("party"):
            print(f"{n:16s} K=1 {table[n][1]:.4f}  best K>1 gain {g:+.5f}  (K: {[round(table[n][k] - table[n][1], 5) for k in KS[1:]]})")
    pg = np.array([g for n, g in gains.items() if n.startswith("party")])
    print("PartyPoker players (n=%d): gain median %+.5f, quartiles %s, share > 0: %.2f" % (len(pg), np.median(pg), np.quantile(pg, [.25, .75]).round(5), (pg > 0).mean()))

    # decode Pluribus with the pooled model; event study around its big losses
    m = load_model(tag)
    xs, cs = subjects["bot: Pluribus"]
    post = m.filtered(SeqData(xs, cs))
    occ = np.concatenate(post).mean(0)
    out["pluribus_occupancy"] = occ
    ok = S.x8 >= 0
    w_ctx = np.bincount(S.ctx[ok], minlength=10) / ok.sum()
    vp = ((1 - m.B[:, :, 0] - m.B[:, :, 1]) * w_ctx[:, None]).sum(0)
    eff = {"loss": [], "win": []}
    for sidx, s in enumerate(P["Pluribus"]):
        net, vpip = s[:, 2], s[:, 3]
        idx = post[sidx] @ vp
        for kind, sel in (("loss", net <= -50), ("win", net >= 50)):
            for i in np.flatnonzero(sel):
                if i >= 20 and i + 20 < len(net):
                    eff[kind].append((vpip[i + 1:i + 21].mean() - vpip[i - 20:i].mean(), idx[i + 1:i + 21].mean() - idx[i - 20:i].mean()))
    for kind in eff:
        e = np.array(eff[kind])
        out["pluribus_" + kind] = dict(n=len(e), vpip=float(e[:, 0].mean()), vpip_se=float(e[:, 0].std() / np.sqrt(len(e))),
                                       index=float(e[:, 1].mean()), index_se=float(e[:, 1].std() / np.sqrt(len(e))))
        print("Pluribus after big", kind, out["pluribus_" + kind])
    print("Pluribus state occupancy under pooled model", occ.round(3), " VPIP", np.mean(np.concatenate([s[:, 3] for s in P["Pluribus"]])).round(3))
    dump(out, "pluribus.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
