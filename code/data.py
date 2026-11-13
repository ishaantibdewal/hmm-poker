"""Loading helpers shared by the experiment scripts."""
import os
import numpy as np
from hmm import SeqData

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS = ["EP", "CO", "BTN", "SB", "BB"]
CTX = [f"{p}, {f}" for p in POS for f in ("no raise", "raised")]
SYM3 = ["fold", "check/call", "raise"]
SYM8 = ["fold", "check", "call S", "call M", "call L", "raise S", "raise M", "raise L"]
FINE2COARSE = np.array([0, 1, 1, 1, 1, 2, 2, 2])
BIG = 50.0                       # a "big" loss or win, in big blinds


class Sequences:
    def __init__(self, path=None):
        z = np.load(path or os.path.join(ROOT, "data", "sequences.npz"))
        self.__dict__.update({k: z[k] for k in z.files})
        self.S = len(self.start)

    def rows(self, s, limit=None):
        a = int(self.start[s]); L = int(self.length[s]) if limit is None else int(limit)
        return slice(a, a + L)

    def inputs(self, thr=BIG, lag=1):
        """u_t = 1 if one of the player's previous `lag` hands in the session was a loss >= thr,
        2 if it was a win >= thr (a loss takes precedence), else 0."""
        u = np.zeros(len(self.net), dtype=np.int8)
        for k in range(1, lag + 1):
            prev = np.r_[np.zeros(k, np.float32), self.net[:-k]]
            idx = np.arange(len(u)) - np.repeat(self.start, self.length)   # position inside the session
            okp = idx >= k
            u[(prev >= thr) & okp & (u == 0)] = 2
            u[(prev <= -thr) & okp] = 1
        return u

    def seqdata(self, sessions, alphabet="x8", context=True, u=None, train=False, groups=False, max_len=None, batch_cells=400_000):
        X = getattr(self, alphabet)
        xs, cs, us, gs = [], [], [], []
        for s in sessions:
            L = self.train_len[s] if train else self.length[s]
            if max_len is not None:
                L = min(L, max_len)
            r = self.rows(s, L)
            xs.append(X[r]); cs.append(self.ctx[r] if context else None)
            us.append(None if u is None else u[r]); gs.append(self.s_player[s])
        g = None
        if groups:
            _, g = np.unique(gs, return_inverse=True)
        return SeqData(xs, cs if context else None, None if u is None else us, groups=g, batch_cells=batch_cells)


def find_events(S, thr=BIG, pre=30, post=50, kind="loss", clean=True, need_post=True, rng=None, neutral_frac=0.1):
    """Events inside sessions, with windows that do not overlap the event hand.

    kind: "loss" (net <= -thr), "win" (net >= thr) or "neutral" (saw a flop, |net| < 10bb;
          a random `neutral_frac` of them).
    Returns dict of arrays:
      i   row of the event hand
      j   row of the first hand of the session that STARTED after the event hand ended
      s   session index
    Requirements: at least `pre` hands before i in the session, (if need_post) at least
    `post` hands from j on, and (if clean) no big win or loss in the `pre` hands before i.
    """
    net = S.net
    big = np.abs(net) >= thr
    if kind == "loss":
        cand = net <= -thr
    elif kind == "win":
        cand = net >= thr
    else:
        cand = S.saw_flop & (np.abs(net) < 10)
        cand &= (rng or np.random.default_rng(0)).random(len(net)) < neutral_frac
    sess = np.repeat(np.arange(S.S), S.length)
    a = S.start[sess]; b = a + S.length[sess]
    idx = np.flatnonzero(cand)
    cb = np.r_[0, np.cumsum(big)]
    out_i, out_j = [], []
    for i in idx:
        lo, hi = a[i], b[i]
        if i - lo < pre:
            continue
        if clean and cb[i] - cb[i - pre] > 0:
            continue
        j = lo + np.searchsorted(S.tsec[lo:hi], S.tend[i], side="left")
        j = max(j, i + 1)
        if j >= hi or (need_post and j + post > hi):
            continue
        out_i.append(i); out_j.append(j)
    i = np.array(out_i, dtype=np.int64); j = np.array(out_j, dtype=np.int64)
    return dict(i=i, j=j, s=sess[i])


def arrival_inputs(S, thr=BIG):
    """Input for the input-output HMM: u = 1 at the first hand a player starts after a
    big loss has ended, 2 after a big win, 0 otherwise."""
    u = np.zeros(len(S.net), dtype=np.int8)
    for code, kind in ((2, "win"), (1, "loss")):          # a loss overrides a win
        ev = find_events(S, thr, pre=0, post=0, kind=kind, clean=False, need_post=False)
        u[ev["j"]] = code
    return u
