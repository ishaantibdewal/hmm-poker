"""An HMM whose states are shifts of a player's own baseline.

    P(x | z = k, context c, player p)  is proportional to  b_p(c, x) * exp(theta[k, x])

b_p(c, .) is the player's own action table (fixed, estimated beforehand), theta[k] is a
vector of log-odds shifts shared by ALL players, and theta[0] = 0 ("normal" play).
This is a multinomial logit with player-by-context intercepts and a state effect.
Only theta (a handful of numbers), the transition matrices and pi are learned, so the
states cannot describe differences between players and cannot overfit to one player.

The M-step for theta has no closed form; it is a small concave problem solved by a few
Newton steps (generalised EM: every step increases the expected log-likelihood).
"""
import numpy as np

from hmm import HMM


class ShiftHMM(HMM):
    def __init__(self, base, theta, U=1, fix_A=False, pseudo=0.0):
        """base: (C, M) baseline tables;  theta: (K, M) initial shifts, theta[0] must be 0."""
        C, M = base.shape
        K = theta.shape[0]
        super().__init__(K, M, C=C, U=U, pseudo=pseudo, fix_A=fix_A)
        self.base = base
        self.theta = theta.astype(float).copy()
        self.free = base.sum(0) > 0          # symbols that occur at all
        self.free[0] = False                 # symbol 0 (fold) is the reference: theta[k, 0] = 0
        self._set_B()
        self.pi = np.ones(K) / K
        self.A = np.repeat(((np.ones((K, K)) + 60 * np.eye(K)) / (K + 60))[None], U, axis=0)

    def _table(self, th):
        w = self.base * np.exp(th)[None, :]
        return w / w.sum(1, keepdims=True)

    def _set_B(self):
        self.B = np.stack([self._table(self.theta[k]) for k in range(self.K)], axis=1)

    def _update_B(self, counts, prior):
        f = self.free
        for k in range(1, self.K):
            n = counts[:, k, :]                       # expected counts (C, M) in state k
            G = n.sum(1)
            for _ in range(8):
                P = self._table(self.theta[k])
                grad = (n.sum(0) - G @ P)[f]
                H = np.diag((G @ P)[f]) - (P[:, f] * G[:, None]).T @ P[:, f]
                step = np.linalg.solve(H + 1e-9 * np.eye(f.sum()), grad)
                self.theta[k, f] += step
                if np.abs(step).max() < 1e-7:
                    break
        self._set_B()

    @property
    def n_params(self):
        kA = 0 if self.fix_A else self.U * self.K * (self.K - 1)
        return (self.K - 1) + kA + (self.K - 1) * int(self.free.sum())
