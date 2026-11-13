"""Experiment 2: ablations around the main model (same K).

  (a) alphabet and context: {3, 8 symbols} x {with, without decision context}.
      All four are scored on the same task: one-step-ahead log-probability of the
      coarse action (fold / check-call / raise) on held-out sessions.
      For each we also measure how much the decoded state depends on table position.
  (b) sequence length: training sessions are cut into chunks of L hands (same hands,
      less continuity).
  (c) pooled vs per-player: shared emissions with per-player transition matrices
      (shrunk to the pooled matrix with strength kappa), and fully personal models
      trained on the first n hands of high-volume players.

Usage: python code/exp_ablations.py <K>
"""
from common import *
from data import FINE2COARSE

SEEDS = [0, 1, 2]


def best_of(jobs_out):
    return max(jobs_out, key=lambda r: r[0]["train_ll"])


def fit_cfg(job):
    return fit_job(job)


def position_dependence(m, S, alphabet, context, sessions):
    """Cramer's V between the most probable state and the position group (0 = independent)."""
    d = S.seqdata(sessions, alphabet, context)
    post = m.posteriors(d)
    z = np.concatenate([p.argmax(1) for p in post])
    pos = np.concatenate([S.pos[S.rows(s)] for s in sessions])
    ok = pos >= 0
    tab = np.zeros((m.K, 5))
    np.add.at(tab, (z[ok], pos[ok]), 1)
    tab = tab[tab.sum(1) > 0]
    e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    chi2 = ((tab - e) ** 2 / e).sum()
    return float(np.sqrt(chi2 / tab.sum() / (min(tab.shape) - 1)))


def chunk_job(args):
    K, L, seed = args
    S = seqs()
    tr = np.flatnonzero(S.split == 0)
    xs, cs = [], []
    for s in tr:
        r = S.rows(s, S.train_len[s])
        x, c = S.x8[r], S.ctx[r]
        step = len(x) if L is None else L
        for a in range(0, len(x), step):
            xs.append(x[a:a + step]); cs.append(c[a:a + step])
    from hmm import SeqData
    d = SeqData(xs, cs)
    m = HMM(K, 8, C=10, seed=seed, pseudo=PSEUDO)
    m.init_emissions_from(d, np.random.default_rng(1000 + seed))
    m.fit(d, n_iter=N_ITER, tol=TOL)
    r = dict(L=L, seed=seed, n_seq=len(xs), train_ll=m.history[-1], n_obs=d.n_obs, n_iter=len(m.history),
             stay=float(np.diag(m.A[0]).mean()))
    for name, k in (("later", 1), ("new", 2)):
        dd = S.seqdata(np.flatnonzero(S.split == k), "x8", True)
        r[name] = m.score(dd) / dd.n_obs
    return r


def personal_job(args):
    """Personal model for one high-volume player, trained on their first n hands."""
    tag, p, n = args
    S = seqs()
    base = load_model(tag)
    sess = np.flatnonzero(S.s_player == p)
    L = S.length[sess]
    total = L.sum()
    cum = np.cumsum(L)
    test = sess[cum > total - 5000]                    # last ~5,000 hands (whole sessions)
    tr_sess = sess[cum <= total - 5000]
    xs, cs, left = [], [], n
    for s in tr_sess:
        take = min(left, S.length[s])
        if take < 20:
            break
        r = S.rows(s, take); xs.append(S.x8[r]); cs.append(S.ctx[r]); left -= take
    from hmm import SeqData
    d_tr = SeqData(xs, cs)
    d_te = S.seqdata(test, "x8", True)
    out = dict(player=int(p), n=n, n_train=d_tr.n_steps, test_obs=d_te.n_obs, pooled=base.score(d_te) / d_te.n_obs)
    m = base.copy(); m.pseudo = PSEUDO; m.history = []
    m.fit(d_tr, n_iter=200, tol=1e-6)
    out["personal"] = m.score(d_te) / d_te.n_obs
    # only the transition matrix personal (emissions stay pooled), with shrinkage
    m2 = base.copy(); m2.history = []
    d_tr.groups = np.zeros(d_tr.N, dtype=np.int64)
    m2.fit(d_tr, n_iter=1, tol=-1, per_group_A=True, kappa=200.0)       # one E-step under pooled params
    A = m2.A_group[0]
    m3 = base.copy(); m3.A = A[None]
    out["personal_A"] = m3.score(d_te) / d_te.n_obs
    return out


def main(K):
    S = seqs()
    tag = json.load(open(os.path.join(RES, "main.json")))["tag"]
    out = {"K": K}
    held = {name: np.flatnonzero(S.split == k) for name, k in (("later", 1), ("new", 2))}

    # (a) alphabet x context
    jobs = [dict(K=K, alphabet=a, context=c, seed=s) for a in ("x3", "x8") for c in (False, True) for s in SEEDS]
    res = pmap(fit_cfg, jobs)
    rows = []
    for a in ("x3", "x8"):
        for c in (False, True):
            info, m = best_of([r for r in res if r[0]["alphabet"] == a and r[0]["context"] == c])
            g = FINE2COARSE if a == "x8" else np.arange(3)
            row = dict(alphabet=a, context=c, n_params=m.n_params, train=info["train_ll"] / info["n_obs"], stay=float(np.diag(m.A[0]).mean()))
            for name, sess in held.items():
                d = S.seqdata(sess, a, c)
                row[name + "_coarse"] = m.coarse_score(d, g) / d.n_obs
                row[name + "_own"] = m.score(d) / d.n_obs
            row["pos_dependence"] = position_dependence(m, S, a, c, held["new"][:3000])
            rows.append(row); print(row, flush=True)
            save_model(m, f"abl_{a}_{'ctx' if c else 'noctx'}_K{K}")
    # K = 1 references for the coarse score
    for a in ("x3", "x8"):
        for c in (False, True):
            info, m = fit_job(dict(K=1, alphabet=a, context=c, seed=0))
            g = FINE2COARSE if a == "x8" else np.arange(3)
            d = S.seqdata(held["new"], a, c); d2 = S.seqdata(held["later"], a, c)
            rows.append(dict(alphabet=a, context=c, K=1, new_coarse=m.coarse_score(d, g) / d.n_obs, later_coarse=m.coarse_score(d2, g) / d2.n_obs))
    out["alphabet_context"] = rows

    # (b) sequence length
    res = pmap(chunk_job, [(K, L, s) for L in (5, 10, 25, 50, 100, 250, None) for s in SEEDS])
    out["chunks"] = res
    for L in (5, 10, 25, 50, 100, 250, None):
        b = max((r for r in res if r["L"] == L), key=lambda r: r["train_ll"])
        print("chunk", L, "seqs", b["n_seq"], "stay %.4f later %.5f new %.5f" % (b["stay"], b["later"], b["new"]), flush=True)

    # (c) per-player transitions with shrinkage (training players with later sessions)
    base = load_model(tag)
    tr = np.flatnonzero(S.split == 0)
    d_tr = S.seqdata(tr, "x8", True, train=True); d_tr.groups = S.s_player[tr]
    d_te = S.seqdata(held["later"], "x8", True); d_te.groups = S.s_player[held["later"]]
    pooled_later = base.score(d_te) / d_te.n_obs
    rows = [dict(kappa=None, later=pooled_later)]
    for kappa in (10.0, 30.0, 100.0, 300.0, 1000.0):
        m = base.copy(); m.pseudo = PSEUDO; m.history = []
        m.fit(d_tr, n_iter=40, tol=-np.inf, per_group_A=True, kappa=kappa)
        rows.append(dict(kappa=kappa, later=m.score(d_te) / d_te.n_obs, train=m.history[-1] / d_tr.n_obs,
                         stay_sd=float(np.diagonal(m.A_group[np.unique(d_tr.groups)], axis1=1, axis2=2).mean(1).std())))
        print(rows[-1], flush=True)
        if kappa == 100.0:
            save_model(m, f"playerA_K{K}")
    out["per_player_A"] = rows

    # fully personal models: learning curve for high-volume training players
    hands = np.bincount(S.s_player, weights=S.length)
    cand = np.flatnonzero((hands >= 20000) & S.fit_player)
    print("high-volume training players:", len(cand))
    res = pmap(personal_job, [(tag, p, n) for p in cand for n in (500, 1000, 2000, 5000, 10000)])
    out["personal"] = res
    for n in (500, 1000, 2000, 5000, 10000):
        r = [x for x in res if x["n"] == n]
        print("n=%5d  pooled %.5f  personal %.5f  personal-A %.5f  (share of players where personal beats pooled: %.2f)" % (
            n, np.mean([x["pooled"] for x in r]), np.mean([x["personal"] for x in r]), np.mean([x["personal_A"] for x in r]),
            np.mean([x["personal"] > x["pooled"] for x in r])), flush=True)
    dump(out, "ablations.json")


if __name__ == "__main__":
    main(int(sys.argv[1]))
