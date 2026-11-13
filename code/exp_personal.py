"""Experiment 6: a model in which every player is their own baseline.

    P(x_t | z_t = k, c_t, player p) = B[(p, c_t), k, x_t]      emissions: one table per player
    P(z_t | z_{t-1}, u_t)           = A[u_t, z_{t-1}, z_t]      transitions: shared by all players

The hidden state can no longer describe differences BETWEEN players (the pooled HMM
spends all its states on those); it can only describe how one player's behaviour moves
around that player's own average.  Modes are ordered by looseness within each player
(mode 0 = tightest) so that the shared transition matrix has one meaning.

Split: for each player the earliest sessions holding 70% of their hands are training
data, the rest is held out.

Usage: python code/exp_personal.py fit         (K sweep, static baseline, alphabets)
       python code/exp_personal.py io <alphabet> <K>
"""
from common import *
from data import arrival_inputs, find_events, FINE2COARSE
from hmm import SeqData

TRAIN_SHARE = 0.7
KAPPA = 10.0          # shrinkage of a player's baseline towards the population (chosen on validation data)
_c = {}


def split_sessions(S):
    """Returns (train pieces, test pieces); a piece is (session, first row, n rows)."""
    if "split" in _c:
        return _c["split"]
    tr, te = [], []
    P = int(S.s_player.max()) + 1
    by = [[] for _ in range(P)]
    for s in range(S.S):
        by[S.s_player[s]].append(s)
    for p in range(P):
        total = S.length[by[p]].sum(); need = TRAIN_SHARE * total; used = 0
        for s in by[p]:
            L = int(S.length[s]); a = int(S.start[s])
            if used >= need:
                te.append((s, a, L))
            elif used + L <= need or L < 200:
                tr.append((s, a, L))
            else:                                   # the session straddles the cut: split it
                k = int(max(100, min(L - 100, need - used)))
                tr.append((s, a, k)); te.append((s, a + k, L - k))
            used += L
    _c["split"] = (tr, te)
    return tr, te


def make(S, pieces, alphabet, u=None, personal=True):
    X = getattr(S, alphabet)
    xs = [X[a:a + L] for _, a, L in pieces]
    if personal:
        cs = [S.ctx[a:a + L].astype(np.int64) + 10 * int(S.s_player[s]) for s, a, L in pieces]
    else:
        cs = [S.ctx[a:a + L] for _, a, L in pieces]
    us = None if u is None else [u[a:a + L] for _, a, L in pieces]
    return SeqData(xs, cs, us)


def init_modes(m, d, rng, spread=0.15, kappa2=20.0, personal=True):
    """Hierarchical shrinkage, then a tight-to-loose start.

    1. A player's baseline table b_p(c, .) is their own action counts shrunk towards the
       population table of that context with KAPPA pseudo-observations.
    2. Every mode of that player gets a Dirichlet prior centred on b_p with kappa2
       pseudo-observations, so a mode moves away from the player's baseline only when the
       data support it, and a mode the data do not need stays at the baseline.
    3. Mode k starts as b_p made tighter (k small) or looser (k large)."""
    cnt = np.zeros((m.C, m.M))
    for _, X, C, _, L in d.batches:
        ok = (np.arange(X.shape[0])[:, None] < L[None]) & (X >= 0)
        np.add.at(cnt, (C[ok], X[ok]), 1.0)
    if personal:
        pop = cnt.reshape(-1, 10, m.M).sum(0)
        pop = (pop + 1e-3) / (pop + 1e-3).sum(1, keepdims=True)
        base = cnt + KAPPA * np.tile(pop, (m.C // 10, 1))
    else:
        base = cnt + 1.0
    base /= base.sum(1, keepdims=True)
    m.prior_B = kappa2 * np.repeat(base[:, None, :], m.K, axis=1)
    if m.K == 1:
        m.prior_B = (base * (cnt.sum(1, keepdims=True) + KAPPA) - cnt)[:, None, :] if personal else m.prior_B
    noise = rng.gamma(50.0, 1 / 50.0, (m.C, m.K, m.M))
    m.B = base[:, None, :] * noise
    m.B[:, :, 0] *= np.linspace(1 + spread, 1 - spread, m.K)[None, :]
    m.B /= m.B.sum(2, keepdims=True)
    m.A = np.repeat(((np.ones((m.K, m.K)) + 30 * np.eye(m.K)) / (m.K + 30))[None], m.U, axis=0)
    m.pi = np.ones(m.K) / m.K


def looseness(m, w):
    """1 - P(fold) of every (player, mode), averaged over that player's contexts.  (P, K)."""
    nf = 1.0 - m.B[:, :, 0]                               # (P*10, K)
    P = m.C // 10
    return (nf.reshape(P, 10, m.K) * w.reshape(P, 10, 1)).sum(1) / np.maximum(w.reshape(P, 10, 1).sum(1), 1e-12)


def context_weights(S, pieces, C):
    w = np.zeros(C)
    for s, a, L in pieces:
        ok = S.x3[a:a + L] >= 0
        np.add.at(w, S.ctx[a:a + L][ok].astype(np.int64) + 10 * int(S.s_player[s]), 1.0)
    return w


def fit_job(job):
    S = seqs()
    tr, te = split_sessions(S)
    if job.get("validate"):                         # tune on a validation part of the training data
        isval = np.random.default_rng(0).random(len(tr)) < 0.2
        te = [p for p, v in zip(tr, isval) if v]
        tr = [p for p, v in zip(tr, isval) if not v]
    alphabet, K, seed = job["alphabet"], job["K"], job["seed"]
    personal = job.get("personal", True)
    P = int(S.s_player.max()) + 1
    C = 10 * P if personal else 10
    M = 8 if alphabet == "x8" else 3
    u = arrival_inputs(S) if job.get("io") else None
    d = make(S, tr, alphabet, u, personal)
    m = HMM(K, M, C=C, U=3 if job.get("io") else 1, seed=seed, pseudo=job.get("pseudo", 0.5), fix_A=job.get("static", False))
    init_modes(m, d, np.random.default_rng(seed), kappa2=job.get("kappa2", 20.0), personal=personal)
    if job.get("static"):
        m.A = np.eye(K)[None]
    t0 = time.time()
    m.fit(d, n_iter=job.get("n_iter", 300), tol=TOL)
    dt = make(S, te, alphabet, u, personal)
    out = dict(job, train=m.history[-1] / d.n_obs, test=m.score(dt) / dt.n_obs, test_obs=dt.n_obs, train_obs=d.n_obs,
               n_iter=len(m.history), seconds=time.time() - t0, A=m.A, pi=m.pi, history=m.history)
    if alphabet == "x8":
        out["test_coarse"] = m.coarse_score(dt, FINE2COARSE) / dt.n_obs
    if personal and K > 1:
        lo = looseness(m, context_weights(S, tr, C))
        out["order_ok"] = float((np.diff(lo, axis=1) > 0).all(1).mean())
        out["loose_mean"] = lo.mean(0)
    tag = "personal_%s_K%d_%s%s%s_k%g_s%d" % (alphabet, K, "p" if personal else "pooled", "_static" if job.get("static") else "", "_io" if job.get("io") else "", job.get("kappa2", 20.0), seed)
    if not job.get("validate"):
        save_model(m, tag)
    out["tag"] = tag + ("_val" if job.get("validate") else "")
    tag = out["tag"]
    print(tag, "train %.5f test %.5f iters %d (%.0fs)" % (out["train"], out["test"], out["n_iter"], out["seconds"]),
          "A", np.round(m.A[0], 4).tolist(), "order_ok", out.get("order_ok"), "loose", np.round(out.get("loose_mean", []), 3), flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "tune":       # choose the mode-prior strength kappa2 on validation data
        jobs = [dict(alphabet=a, K=1, seed=0, validate=True) for a in ("x3", "x8")]
        jobs += [dict(alphabet=a, K=2, seed=0, kappa2=k, validate=True, static=st) for a in ("x3", "x8") for k in (5.0, 20.0, 80.0, 320.0) for st in (False, True)]
        dump(pmap(fit_job, jobs), "personal_tune.json")
    if sys.argv[1] == "fit":        # final fits on all training data, scored on the held-out 30%
        jobs = []
        for alphabet in ("x3", "x8"):
            jobs.append(dict(alphabet=alphabet, K=1, seed=0))
            jobs.append(dict(alphabet=alphabet, K=1, seed=0, personal=False))
            for k2 in (20.0, 320.0):
                jobs.append(dict(alphabet=alphabet, K=2, seed=0, kappa2=k2))
                jobs.append(dict(alphabet=alphabet, K=2, seed=0, kappa2=k2, static=True))
        jobs.sort(key=lambda j: -j["K"])
        dump(pmap(fit_job, jobs), "personal_fit.json")
