"""Correctness tests for hmm.py: brute-force enumeration, hmmlearn cross-check,
monotone likelihood, and parameter recovery on synthetic data."""
import itertools, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hmm import HMM, SeqData


def brute(m, x, c, u):
    T = len(x); K = m.K
    post = np.zeros((T, K)); tot = 0.0; best = (-1, None)
    for z in itertools.product(range(K), repeat=T):
        p = m.pi[z[0]]
        for t in range(T):
            if t: p *= m.A[u[t], z[t - 1], z[t]]
            if x[t] >= 0: p *= m.B[c[t], z[t], x[t]]
        tot += p
        if p > best[0]: best = (p, z)
        for t in range(T): post[t, z[t]] += p
    return np.log(tot), post / tot, np.array(best[1]), np.log(best[0])


def test_bruteforce():
    rng = np.random.default_rng(1)
    for C, U in [(1, 1), (3, 1), (1, 2), (3, 3)]:
        m = HMM(3, 4, C=C, U=U, seed=5)
        m.A = np.stack([np.random.default_rng(u).dirichlet(np.ones(3), 3) for u in range(U)])
        xs, cs, us = [], [], []
        for T in (1, 4, 7, 6):
            x = rng.integers(0, 4, T); x[rng.random(T) < 0.2] = -1
            xs.append(x); cs.append(rng.integers(0, C, T)); us.append(rng.integers(0, U, T))
        d = SeqData(xs, cs, us, batch_cells=14)     # forces padding inside a batch
        ll = m.score(d, per_seq=True); post = m.posteriors(d)
        for n in range(len(xs)):
            bl, bp, bz, bv = brute(m, xs[n], cs[n], us[n])
            assert abs(ll[n] - bl) < 1e-10, (ll[n], bl)
            assert np.abs(post[n] - bp).max() < 1e-10
            path, v = m.viterbi(xs[n], cs[n], us[n])
            assert abs(v - bv) < 1e-10 and (path == bz).all()
    print("brute-force: likelihood, posteriors and Viterbi match enumeration")


def test_hmmlearn():
    from hmmlearn.hmm import CategoricalHMM
    rng = np.random.default_rng(2)
    true = HMM(3, 6, seed=11)
    xs = [true.sample(int(T), rng)[1] for T in rng.integers(30, 200, 40)]
    m = HMM(3, 6, seed=3)
    h = CategoricalHMM(n_components=3, n_iter=25, tol=-np.inf, init_params="", params="ste")
    h.startprob_, h.transmat_, h.emissionprob_ = m.pi.copy(), m.A[0].copy(), m.B[0].copy()
    h.n_features = 6
    Xc = np.concatenate(xs)[:, None]; lens = [len(x) for x in xs]
    d = SeqData(xs, batch_cells=1500)
    assert abs(m.score(d) - h.score(Xc, lens)) < 1e-8
    h.fit(Xc, lens)
    m.fit(d, n_iter=25, tol=-np.inf)
    err = max(np.abs(m.pi - h.startprob_).max(), np.abs(m.A[0] - h.transmat_).max(), np.abs(m.B[0] - h.emissionprob_).max())
    assert err < 1e-8, err
    assert abs(m.score(d) - h.score(Xc, lens)) < 1e-6
    path, _ = m.viterbi(xs[0]); assert (path == h.predict(xs[0][:, None])).all()
    assert all(b >= a - 1e-9 for a, b in zip(m.history, m.history[1:]))
    print(f"hmmlearn: 25 EM iterations from the same start agree, max |param diff| = {err:.1e}")


def test_recovery():
    rng = np.random.default_rng(4)
    true = HMM(2, 3, C=2, U=2, seed=0)
    true.pi = np.array([0.7, 0.3])
    true.A = np.array([[[0.97, 0.03], [0.05, 0.95]], [[0.60, 0.40], [0.05, 0.95]]])
    true.B = np.array([[[0.8, 0.15, 0.05], [0.4, 0.3, 0.3]], [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]]])
    xs, cs, us = [], [], []
    for _ in range(300):
        T = 400; c = rng.integers(0, 2, T); u = (rng.random(T) < 0.05).astype(int)
        z, x = true.sample(T, rng, c, u); xs.append(x); cs.append(c); us.append(u)
    d = SeqData(xs, cs, us)
    best = max((HMM(2, 3, C=2, U=2, seed=s).fit(d, n_iter=300, tol=1e-8) for s in range(3)), key=lambda m: m.history[-1])
    if best.B[0, 0, 0] < best.B[0, 1, 0]:       # undo label switching
        p = [1, 0]; best.pi = best.pi[p]; best.A = best.A[:, p][:, :, p]; best.B = best.B[:, p]
    eA, eB = np.abs(best.A - true.A).max(), np.abs(best.B - true.B).max()
    assert eA < 0.05 and eB < 0.03, (eA, eB)
    assert all(b >= a - 1e-7 for a, b in zip(best.history, best.history[1:]))
    print(f"recovery: input-output HMM with contexts recovered, max err A {eA:.3f}, B {eB:.3f}")


def test_group_A():
    rng = np.random.default_rng(6)
    xs, groups = [], []
    for g, stay in enumerate([0.99, 0.80]):
        t = HMM(2, 3, seed=0); t.pi = np.array([.5, .5])
        t.A = np.array([[[stay, 1 - stay], [1 - stay, stay]]]); t.B = np.array([[[.8, .1, .1], [.1, .1, .8]]])
        for _ in range(20):
            xs.append(t.sample(500, rng)[1]); groups.append(g)
    d = SeqData(xs, groups=groups)
    m = HMM(2, 3, seed=1).fit(d, n_iter=200, tol=1e-9, per_group_A=True, kappa=5.0)
    stays = np.sort([np.diag(m.A_group[g]).mean() for g in range(2)])
    assert abs(stays[1] - 0.99) < 0.01 and abs(stays[0] - 0.80) < 0.03, stays
    assert all(b >= a - 1e-6 for a, b in zip(m.history[2:], m.history[3:]))
    print(f"per-group transitions: recovered self-transition {stays.round(3)} (truth 0.80, 0.99)")




def test_shift_hmm():
    """Shared log-odds shift on fixed baseline tables, with input-dependent transitions."""
    from shift_hmm import ShiftHMM
    rng = np.random.default_rng(0)
    C = 40
    base = rng.dirichlet([6, 1.5, 1.5], C)
    true = ShiftHMM(base, np.array([[0, 0, 0], [0, 0.5, 0.8]]), U=2)
    true.A = np.array([[[0.99, 0.01], [0.03, 0.97]], [[0.7, 0.3], [0.03, 0.97]]]); true.pi = np.array([0.8, 0.2])
    xs, cs, us = [], [], []
    for _ in range(600):
        T = 500; c = rng.integers(0, C, T); u = (rng.random(T) < 0.02).astype(int)
        z, x = true.sample(T, rng, c, u); xs.append(x); cs.append(c); us.append(u)
    m = ShiftHMM(base, np.array([[0, 0, 0], [0, 0.2, 0.2]]), U=2).fit(SeqData(xs, cs, us), n_iter=300, tol=1e-9)
    assert all(b >= a - 1e-6 for a, b in zip(m.history, m.history[1:]))
    assert np.abs(m.theta[1] - true.theta[1]).max() < 0.08 and np.abs(m.A - true.A).max() < 0.04
    print(f"shift HMM: recovered theta {m.theta[1].round(2)} (truth [0, 0.5, 0.8]), P(enter | input) {m.A[1, 0, 1]:.2f} (truth 0.30)")


if __name__ == "__main__":
    test_bruteforce(); test_hmmlearn(); test_recovery(); test_group_A(); test_shift_hmm()
    print("all tests passed")
