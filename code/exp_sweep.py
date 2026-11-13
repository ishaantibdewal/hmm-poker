"""Experiment 1: number of states K, against two baselines.

  K = 1            no hidden state: actions are i.i.d. given the context
  static mixture   A = I: each session has one fixed style (no switching within a session)
  HMM              full transition matrix

Every configuration is run from 5 random starts.  We report train and held-out
log-likelihood per decision, and BIC.
Usage: python code/exp_sweep.py
"""
from common import *

KS = [1, 2, 3, 4, 5, 6, 7, 8]
SEEDS = [0, 1, 2, 3, 4]


def run(job):
    out, m = fit_job(job)
    S = seqs()
    w = np.bincount(S.ctx[S.x8 >= 0], minlength=10) / (S.x8 >= 0).sum()
    order_states(m, w)
    for name, k in (("later", 1), ("new", 2)):
        d = S.seqdata(np.flatnonzero(S.split == k), job["alphabet"], job["context"])
        out[f"{name}_ll"] = m.score(d); out[f"{name}_obs"] = d.n_obs
    tag = f"{'static' if job.get('static') else 'hmm'}_{job['alphabet']}_{'ctx' if job['context'] else 'noctx'}_K{job['K']}_s{job['seed']}"
    save_model(m, tag)
    out["tag"] = tag
    print(tag, "train %.5f later %.5f new %.5f iters %d (%.0fs)" % (
        out["train_ll"] / out["n_obs"], out["later_ll"] / out["later_obs"], out["new_ll"] / out["new_obs"], out["n_iter"], out["seconds"]), flush=True)
    return out


if __name__ == "__main__":
    jobs = [dict(K=K, alphabet="x8", context=True, seed=s) for K in KS for s in (SEEDS if K > 1 else [0])]
    jobs += [dict(K=K, alphabet="x8", context=True, seed=s, static=True) for K in KS[1:] for s in SEEDS]
    jobs.sort(key=lambda j: -j["K"])
    dump(pmap(run, jobs), "sweep.json")
