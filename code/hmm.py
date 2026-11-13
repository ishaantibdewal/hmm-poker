"""Discrete hidden Markov models in NumPy, written for CSE 150A/250A.

One class covers every model in the report:

    z_t  in {0..K-1}   hidden mode of one player at their t-th hand
    x_t  in {0..M-1}   observed action symbol (-1 = no decision, treated as missing)
    c_t  in {0..C-1}   observed context of the decision (position x facing a raise)
    u_t  in {0..U-1}   observed input for the transition INTO hand t (e.g. "big loss last hand")

    P(z_1)                = pi[z_1]
    P(z_t | z_{t-1}, u_t) = A[u_t, z_{t-1}, z_t]
    P(x_t | z_t, c_t)     = B[c_t, z_t, x_t]

With C = U = 1 this is the textbook HMM.  C > 1 gives context-dependent emissions,
U > 1 gives an input-output HMM.  All sequences share the same parameters ("pooled"
model).  Optionally each group of sequences (a player) has its own transition matrix,
shrunk towards the pooled one.

Learning is Baum-Welch (EM) with the scaled forward-backward recursions, vectorised
over a batch of sequences: sequences are padded to a common length and a padded step
is a no-op (identity transition, emission probability 1).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- data
class SeqData:
    """A set of sequences, stored padded and time-major in length-sorted batches."""

    def __init__(self, xs, cs=None, us=None, groups=None, batch_cells=400_000):
        self.N = len(xs)
        self.lengths = np.array([len(x) for x in xs], dtype=np.int64)
        self.groups = None if groups is None else np.asarray(groups, dtype=np.int64)
        self.n_obs = int(sum(int((np.asarray(x) >= 0).sum()) for x in xs))
        self.n_steps = int(self.lengths.sum())
        order = np.argsort(-self.lengths, kind="stable")
        self.batches = []
        i = 0
        while i < self.N:
            Tm = int(self.lengths[order[i]])
            n = max(1, min(self.N - i, batch_cells // max(Tm, 1)))
            idx = order[i:i + n]
            X = np.full((Tm, n), -1, dtype=np.int64)
            C = np.zeros((Tm, n), dtype=np.int64)
            U = np.zeros((Tm, n), dtype=np.int64)
            for j, s in enumerate(idx):
                L = self.lengths[s]
                X[:L, j] = xs[s]
                if cs is not None:
                    C[:L, j] = cs[s]
                if us is not None:
                    U[:L, j] = us[s]
            self.batches.append((idx, X, C, U, self.lengths[idx]))
            i += n


def _xlogy(a, p):
    """sum of a * log(p), with 0 * log(0) = 0 and nothing added when a is 0."""
    a = np.broadcast_to(a, p.shape)
    m = (a > 0) & (p > 0)
    return float((a[m] * np.log(p[m])).sum()) if m.any() else 0.0


def _norm(a, axis=-1):
    s = a.sum(axis=axis, keepdims=True)
    return a / np.where(s > 0, s, 1.0)


# -------------------------------------------------------------------------- model
class HMM:
    def __init__(self, K, M, C=1, U=1, seed=0, pseudo=0.0, fix_A=False):
        """pseudo: Dirichlet pseudo-count added to every expected count in the M-step.
        fix_A:  keep the transition matrix fixed (used for the static-mixture baseline)."""
        self.K, self.M, self.C, self.U = K, M, C, U
        self.pseudo, self.fix_A = pseudo, fix_A
        rng = np.random.default_rng(seed)
        self.pi = _norm(rng.uniform(0.5, 1.5, K))
        A = rng.uniform(0.5, 1.5, (K, K)) + 8.0 * np.eye(K)   # sticky start
        self.A = np.repeat(_norm(A)[None], U, axis=0)
        self.B = _norm(rng.uniform(0.5, 1.5, (C, K, M)))
        self.A_group = None        # (G, K, K) when per-group transitions are used
        self.prior_B = None        # optional (C, K, M) pseudo-counts for the emissions (replaces `pseudo` for B)
        self.history = []          # log-likelihood before every M-step
        self.objective = []        # log-likelihood + log-prior: the quantity EM increases monotonically

    def init_emissions_from(self, data: SeqData, rng, jitter=0.5):
        """Start every state near the empirical P(x | c), perturbed, so that no state
        starts with zero mass on a symbol that is common in some context."""
        cnt = np.zeros((self.C, self.M))
        for _, X, C, _, L in data.batches:
            ok = (np.arange(X.shape[0])[:, None] < L[None]) & (X >= 0)
            np.add.at(cnt, (C[ok], X[ok]), 1.0)
        base = _norm(cnt + 1.0)
        noise = rng.gamma(1.0 / jitter, jitter, (self.C, self.K, self.M))
        self.B = _norm(base[:, None, :] * noise)

    # ----------------------------------------------------------------- E-step core
    def _forward(self, X, C, U, L, coarse=None):
        """Scaled forward pass.  Returns alpha (T,n,K), scale factors cs (T,n).
        If coarse=(Bc, GX) is given also returns the one-step predictive probability
        of the coarse symbol GX[t] (used to compare alphabets)."""
        T, n = X.shape
        K = self.K
        Bx = np.concatenate([self.B, np.ones((self.C, K, 1))], axis=2)  # x = -1 -> 1
        alpha = np.empty((T, n, K))
        cs = np.ones((T, n))
        pc = None
        if coarse is not None:
            Bc, GX = coarse
            Bc = np.concatenate([Bc, np.ones((self.C, K, 1))], axis=2)
            pc = np.ones((T, n))
            pc[0] = (self.pi[None] * Bc[C[0], :, GX[0]]).sum(1)
        a = self.pi[None] * Bx[C[0], :, X[0]]
        c = a.sum(1)
        alpha[0] = a / c[:, None]
        cs[0] = c
        for t in range(1, T):
            prev = alpha[t - 1]
            if self.A_group is not None:
                pred = np.einsum("ni,nij->nj", prev, self._Ag_batch)
            elif self.U == 1:
                pred = prev @ self.A[0]
            else:                                   # inputs are rare: use A[0], then redo the rows with u != 0
                pred = prev @ self.A[0]
                nz = np.flatnonzero(U[t])
                if nz.size:
                    pred[nz] = np.einsum("ni,nij->nj", prev[nz], self.A[U[t][nz]])
            a = pred * Bx[C[t], :, X[t]]
            c = a.sum(1)
            act = t < L
            if act.all():
                alpha[t] = a / c[:, None]
                cs[t] = c
            else:
                alpha[t] = np.where(act[:, None], a / c[:, None], prev)
                cs[t] = np.where(act, c, 1.0)
            if pc is not None:
                p = (pred * Bc[C[t], :, GX[t]]).sum(1)
                pc[t] = np.where(act, p, 1.0)
        return alpha, cs, pc

    def _estep_batch(self, batch, stats=None, keep_gamma=False):
        idx, X, C, U, L = batch
        T, n = X.shape
        K = self.K
        if self.A_group is not None:
            self._Ag_batch = self.A_group[self._groups[idx]]
        alpha, cs, _ = self._forward(X, C, U, L)
        ll_seq = np.log(cs).sum(0)
        if stats is None and not keep_gamma:
            return ll_seq, None
        Bx = np.concatenate([self.B, np.ones((self.C, K, 1))], axis=2)
        beta = np.ones((n, K))
        per_seq = stats is not None and "xi_seq" in stats
        xi_seq = np.zeros((n, K, K)) if per_seq else None
        for t in range(T - 1, 0, -1):
            act = t < L
            w = Bx[C[t], :, X[t]] * beta / cs[t][:, None]          # (n,K)
            prev = alpha[t - 1]
            if self.A_group is not None:
                At = self._Ag_batch
                if stats is not None:
                    xi_seq += np.where(act[:, None, None], prev[:, :, None] * At * w[:, None, :], 0.0)
                nb = np.einsum("nij,nj->ni", At, w)
            elif self.U == 1:
                A0 = self.A[0]
                if stats is not None:
                    if per_seq:
                        xi_seq += np.where(act[:, None, None], prev[:, :, None] * A0[None] * w[:, None, :], 0.0)
                    else:
                        m = act
                        stats["xi"][0] += A0 * (prev[m].T @ w[m])
                nb = w @ A0.T
            else:
                if stats is not None:
                    for u in range(self.U):
                        m = act & (U[t] == u)
                        if m.any():
                            stats["xi"][u] += self.A[u] * (prev[m].T @ w[m])
                nb = w @ self.A[0].T
                nz = np.flatnonzero(U[t])
                if nz.size:
                    nb[nz] = np.einsum("nij,nj->ni", self.A[U[t][nz]], w[nz])
            alpha[t] *= beta                                           # gamma_t
            beta = np.where(act[:, None], nb, beta)
        alpha[0] *= beta
        gamma = alpha
        if stats is not None:
            stats["pi"] += gamma[0].sum(0)
            ok = (np.arange(T)[:, None] < L[None]) & (X >= 0)
            flat = (C * self.M + X)[ok]
            g = gamma[ok]
            for i in range(K):
                stats["B"][:, i, :] += np.bincount(flat, weights=g[:, i], minlength=self.C * self.M).reshape(self.C, self.M)
            if per_seq:
                np.add.at(stats["xi_seq"], self._groups[idx], xi_seq)
        return ll_seq, (gamma if keep_gamma else None)

    # ------------------------------------------------------------------- learning
    def fit(self, data: SeqData, n_iter=200, tol=1e-6, per_group_A=False, kappa=50.0, verbose=False):
        """Baum-Welch.  Stops when the gain in log-likelihood per observation < tol.
        per_group_A: every group (player) gets its own transition matrix, estimated as
        the posterior mean under a Dirichlet prior kappa * (pooled A)."""
        assert not (per_group_A and self.U > 1)
        self._groups = data.groups
        G = int(data.groups.max()) + 1 if per_group_A else 0
        prev = -np.inf
        for it in range(n_iter):
            stats = {"pi": np.zeros(self.K), "xi": np.zeros((self.U, self.K, self.K)),
                     "B": np.zeros((self.C, self.K, self.M))}
            if per_group_A:
                stats["xi_seq"] = np.zeros((G, self.K, self.K))
            ll = 0.0
            for b in data.batches:
                ll_seq, _ = self._estep_batch(b, stats)
                ll += ll_seq.sum()
            self.history.append(ll)
            aB = self.pseudo if self.prior_B is None else self.prior_B
            obj = ll + _xlogy(aB, self.B) + self.pseudo * (_xlogy(1.0, self.pi) + (0 if self.fix_A else _xlogy(1.0, self.A)))
            self.objective.append(obj)
            # M-step (posterior mode under Dirichlet priors with the pseudo-counts as exponents)
            self.pi = _norm(stats["pi"] + self.pseudo)
            self._update_B(stats["B"], aB)
            if not self.fix_A:
                if per_group_A:
                    pooled = _norm(stats["xi_seq"].sum(0) + self.pseudo)
                    self.A = pooled[None]
                    self.A_group = _norm(stats["xi_seq"] + kappa * pooled[None])
                else:
                    newA = _norm(stats["xi"] + self.pseudo)
                    dead = newA.sum(-1) == 0            # a (input, state) pair that never occurred keeps its old row
                    newA[dead] = self.A[dead]
                    self.A = newA
            if verbose:
                print(f"  iter {it:3d}  loglik/obs {ll / data.n_obs:.6f}")
            if it > 0 and (obj - prev) / data.n_obs < tol:
                break
            prev = obj
        return self

    def _update_B(self, counts, prior):
        """Emission M-step.  Subclasses with structured emissions override this."""
        self.B = _norm(counts + prior)

    # ------------------------------------------------------------------ inference
    def score(self, data: SeqData, per_seq=False):
        """Total log-likelihood log P(x | c, u) of the data (sum over sequences)."""
        self._groups = data.groups
        out = np.zeros(data.N)
        for b in data.batches:
            out[b[0]], _ = self._estep_batch(b)
        return out if per_seq else float(out.sum())

    def posteriors(self, data: SeqData):
        """Smoothed posteriors P(z_t | x_1..T) for every sequence: list of (T_n, K)."""
        self._groups = data.groups
        out = [None] * data.N
        for b in data.batches:
            _, gamma = self._estep_batch(b, keep_gamma=True)
            for j, s in enumerate(b[0]):
                out[s] = gamma[: b[4][j], j].copy()
        return out

    def filtered(self, data: SeqData):
        """Filtered posteriors P(z_t | x_1..t): uses the past only."""
        self._groups = data.groups
        out = [None] * data.N
        for idx, X, C, U, L in data.batches:
            if self.A_group is not None:
                self._Ag_batch = self.A_group[self._groups[idx]]
            alpha, _, _ = self._forward(X, C, U, L)
            for j, s in enumerate(idx):
                out[s] = alpha[: L[j], j].copy()
        return out

    def coarse_score(self, data: SeqData, sym2group):
        """Sum over hands of log P(g(x_t) | x_1..t-1): the one-step-ahead predictive
        log-probability of a coarsened symbol.  Lets models with different alphabets
        be compared on the same prediction task."""
        sym2group = np.asarray(sym2group)
        Mc = int(sym2group.max()) + 1
        Bc = np.zeros((self.C, self.K, Mc))
        for k in range(self.M):
            Bc[:, :, sym2group[k]] += self.B[:, :, k]
        total = 0.0
        for idx, X, C, U, L in data.batches:
            GX = np.where(X >= 0, sym2group[np.maximum(X, 0)], -1)
            _, _, pc = self._forward(X, C, U, L, coarse=(Bc, GX))
            total += np.log(pc).sum()
        return float(total)

    def viterbi(self, x, c=None, u=None, A=None):
        """Most likely state path for one sequence (log-space dynamic programme)."""
        T = len(x)
        c = np.zeros(T, dtype=int) if c is None else c
        u = np.zeros(T, dtype=int) if u is None else u
        with np.errstate(divide="ignore"):
            lpi = np.log(self.pi)
            lA = np.log(self.A if A is None else A[None])
            lB = np.log(np.concatenate([self.B, np.ones((self.C, self.K, 1))], axis=2))
        d = lpi + lB[c[0], :, x[0]]
        back = np.zeros((T, self.K), dtype=np.int64)
        for t in range(1, T):
            s = d[:, None] + lA[u[t] if A is None else 0]
            back[t] = s.argmax(0)
            d = s.max(0) + lB[c[t], :, x[t]]
        path = np.empty(T, dtype=np.int64)
        path[-1] = d.argmax()
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path, float(d.max())

    def sample(self, T, rng, c=None, u=None):
        c = np.zeros(T, dtype=int) if c is None else c
        u = np.zeros(T, dtype=int) if u is None else u
        z = np.empty(T, dtype=np.int64)
        x = np.empty(T, dtype=np.int64)
        z[0] = rng.choice(self.K, p=self.pi)
        for t in range(T):
            if t > 0:
                z[t] = rng.choice(self.K, p=self.A[u[t], z[t - 1]])
            x[t] = rng.choice(self.M, p=self.B[c[t], z[t]])
        return z, x

    @property
    def n_params(self):
        """Free parameters (for BIC)."""
        kA = 0 if self.fix_A else self.U * self.K * (self.K - 1)
        return (self.K - 1) + kA + self.C * self.K * (self.M - 1)

    def copy(self):
        m = HMM(self.K, self.M, self.C, self.U, pseudo=self.pseudo, fix_A=self.fix_A)
        m.pi, m.A, m.B = self.pi.copy(), self.A.copy(), self.B.copy()
        m.A_group = None if self.A_group is None else self.A_group.copy()
        m.prior_B = self.prior_B
        m.history = list(self.history)
        return m
