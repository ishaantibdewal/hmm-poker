"""Shared fitting utilities: restarts, parallel map, model save/load."""
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from data import ROOT, Sequences
from hmm import HMM

RES = os.path.join(ROOT, "results")
N_PROC = 12
N_ITER, TOL, PSEUDO = 400, 2e-7, 0.1
_S = None


def seqs():
    global _S
    if _S is None:
        _S = Sequences()
    return _S


def save_model(m, name):
    np.savez(os.path.join(RES, "models", name + ".npz"), pi=m.pi, A=m.A, B=m.B, history=np.array(m.history),
             A_group=np.zeros(0) if m.A_group is None else m.A_group)


def load_model(name):
    z = np.load(os.path.join(RES, "models", name + ".npz"))
    C, K, M = z["B"].shape
    m = HMM(K, M, C=C, U=z["A"].shape[0])
    m.pi, m.A, m.B, m.history = z["pi"], z["A"], z["B"], list(z["history"])
    if z["A_group"].size:
        m.A_group = z["A_group"]
    return m


def order_states(m, ctx_weights):
    """Relabel states by how often they put money in voluntarily (tightest first), so
    that state numbers mean the same thing across restarts and models."""
    fold = (m.B[:, :, 0] * ctx_weights[:, None]).sum(0)
    p = np.argsort(-fold)
    m.pi = m.pi[p]; m.A = m.A[:, p][:, :, p]; m.B = m.B[:, p]
    if m.A_group is not None:
        m.A_group = m.A_group[:, p][:, :, p]
    return m


def fit_job(job):
    """One EM run.  job: dict(K, alphabet, context, seed, static)."""
    S = seqs()
    t0 = time.time()
    tr = np.flatnonzero(S.split == 0)
    d = S.seqdata(tr, job["alphabet"], job["context"], train=True)
    M = 8 if job["alphabet"] == "x8" else 3
    m = HMM(job["K"], M, C=10 if job["context"] else 1, seed=job["seed"], pseudo=PSEUDO, fix_A=job.get("static", False))
    m.init_emissions_from(d, np.random.default_rng(1000 + job["seed"]))
    if job.get("static"):
        m.A = np.eye(job["K"])[None]
    m.fit(d, n_iter=N_ITER, tol=TOL)
    out = dict(job, train_ll=m.history[-1], n_obs=d.n_obs, n_iter=len(m.history), seconds=time.time() - t0,
               n_params=m.n_params, history=[float(h) for h in m.history])
    return out, m


def pmap(fn, jobs, n=N_PROC):
    with Pool(min(n, len(jobs))) as pool:
        return pool.map(fn, jobs, chunksize=1)


def dump(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
