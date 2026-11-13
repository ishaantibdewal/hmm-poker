"""Experiment 4: input-output HMM.  The transition matrix depends on an observed input:

    u_t = 0  nothing special
    u_t = 1  first hand started after a big loss (>= 50bb) ended
    u_t = 2  first hand started after a big win  (>= 50bb) ended

so P(enter state k | previous state, just lost big) is estimated directly.
  * fit by EM, warm-started from the plain HMM (emissions are free to move too)
  * held-out log-likelihood gain over the plain HMM
  * placebo null: the same inputs moved to random hands of the same session (20 draws)
  * bootstrap over training players for confidence intervals on the transition rows

Usage: python code/exp_iohmm.py <model tag>
"""
from common import *
from data import arrival_inputs

N_NULL, N_BOOT = 20, 200
_cache = {}


def setup(tag):
    if not _cache:
        S = seqs()
        _cache.update(S=S, u=arrival_inputs(S), base=load_model(tag), tr=np.flatnonzero(S.split == 0))
    return _cache["S"], _cache["u"], _cache["base"], _cache["tr"]


def io_from(base):
    m = HMM(base.K, base.M, C=base.C, U=3, pseudo=PSEUDO)
    m.pi, m.B, m.A = base.pi.copy(), base.B.copy(), np.repeat(base.A[:1], 3, axis=0)
    return m


def heldout(m, S, u):
    r = {}
    for name, k in (("later", 1), ("new", 2)):
        d = S.seqdata(np.flatnonzero(S.split == k), "x8", True, u=u)
        r[name] = m.score(d); r[name + "_obs"] = d.n_obs
    return r


def shuffled(S, u, rng):
    """Move every non-zero input to a random hand (not the first) of the same session."""
    v = np.zeros_like(u)
    sess = np.repeat(np.arange(S.S), S.length)
    idx = np.flatnonzero(u > 0)
    lo = S.start[sess[idx]]; L = S.length[sess[idx]]
    v[lo + 1 + (rng.random(len(idx)) * (L - 1)).astype(np.int64)] = u[idx]
    return v


def null_job(args):
    tag, seed = args
    S, u, base, tr = setup(tag)
    v = shuffled(S, u, np.random.default_rng(seed))
    m = io_from(base).fit(S.seqdata(tr, "x8", True, u=v, train=True), n_iter=N_ITER, tol=TOL)
    return dict(train=m.history[-1], **heldout(m, S, v))


def boot_job(args):
    tag, seed, pi, A, B = args
    S, u, base, tr = setup(tag)
    rng = np.random.default_rng(seed)
    players = np.unique(S.s_player[tr])
    by = {p: [] for p in players}
    for s in tr:
        by[S.s_player[s]].append(s)
    pick = rng.choice(players, len(players))
    sess = np.array([s for p in pick for s in by[p]])
    m = HMM(base.K, base.M, C=base.C, U=3, pseudo=PSEUDO)
    m.pi, m.A, m.B = pi.copy(), A.copy(), B.copy()
    m.fit(S.seqdata(sess, "x8", True, u=u, train=True), n_iter=25, tol=TOL)
    return m.A


def main(tag):
    S, u, base, tr = setup(tag)
    d_tr = S.seqdata(tr, "x8", True, u=u, train=True)
    print("inputs: share after-loss %.5f, after-win %.5f; in training rows: %d losses, %d wins" % (
        (u == 1).mean(), (u == 2).mean(), sum(int((u[S.rows(s, S.train_len[s])] == 1).sum()) for s in tr),
        sum(int((u[S.rows(s, S.train_len[s])] == 2).sum()) for s in tr)))
    base_tr = base.score(S.seqdata(tr, "x8", True, train=True))
    base_ho = heldout(base, S, None)
    io = io_from(base).fit(d_tr, n_iter=N_ITER, tol=TOL)
    save_model(io, "iohmm_" + tag)
    io_ho = heldout(io, S, u)
    out = {"tag": tag, "A": io.A, "base_A": base.A[0], "n_obs_train": d_tr.n_obs,
           "gain_train": io.history[-1] - base_tr,
           "gain_later": io_ho["later"] - base_ho["later"], "gain_new": io_ho["new"] - base_ho["new"],
           "later_obs": io_ho["later_obs"], "new_obs": io_ho["new_obs"], "iters": len(io.history),
           "max_B_change": float(np.abs(io.B - base.B).max())}
    print("gain in log-likelihood (nats): train %.1f, later sessions %.1f, new players %.1f" % (out["gain_train"], out["gain_later"], out["gain_new"]))
    null = pmap(null_job, [(tag, 100 + i) for i in range(N_NULL)])
    out["null_train"] = [n["train"] - base_tr for n in null]
    out["null_later"] = [n["later"] - base_ho["later"] for n in null]
    out["null_new"] = [n["new"] - base_ho["new"] for n in null]
    print("placebo gains: train mean %.1f max %.1f | later mean %.1f max %.1f | new mean %.1f max %.1f" % (
        np.mean(out["null_train"]), np.max(out["null_train"]), np.mean(out["null_later"]), np.max(out["null_later"]),
        np.mean(out["null_new"]), np.max(out["null_new"])))
    boots = np.stack(pmap(boot_job, [(tag, 500 + i, io.pi, io.A, io.B) for i in range(N_BOOT)]))
    out["A_lo"], out["A_hi"] = np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)
    # summary: probability that the next state is looser / tighter than the current one (states are ordered tight -> loose)
    ok = S.x8 >= 0
    w_ctx = np.bincount(S.ctx[ok], minlength=10) / ok.sum()
    vp = ((1 - io.B[:, :, 0] - io.B[:, :, 1]) * w_ctx[:, None]).sum(0)
    out["vpip_state"] = vp
    def next_vpip(A):      # expected VPIP of the next state, by current state
        return A @ vp
    out["next_vpip"] = np.stack([next_vpip(io.A[k]) for k in range(3)])
    nb = np.stack([np.stack([next_vpip(b[k]) for k in range(3)]) for b in boots])
    out["next_vpip_loss_minus_none_ci"] = np.quantile(nb[:, 1] - nb[:, 0], [0.025, 0.975], axis=0)
    out["next_vpip_loss_minus_win_ci"] = np.quantile(nb[:, 1] - nb[:, 2], [0.025, 0.975], axis=0)
    out["next_vpip_win_minus_none_ci"] = np.quantile(nb[:, 2] - nb[:, 0], [0.025, 0.975], axis=0)
    dump(out, "iohmm.json")
    np.set_printoptions(precision=3, suppress=True)
    for k, name in enumerate(["no event", "after big loss", "after big win"]):
        print(name); print(io.A[k])
    print("state VPIP", vp)
    print("expected VPIP of next state: none/loss/win\n", out["next_vpip"])
    print("loss - none CI\n", out["next_vpip_loss_minus_none_ci"], "\nloss - win CI\n", out["next_vpip_loss_minus_win_ci"])


if __name__ == "__main__":
    main(sys.argv[1])
