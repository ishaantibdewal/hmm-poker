"""Experiment 7: tilt as a shift of each player's own baseline (shift_hmm.py).

Same split as exp_personal.py (first 70% of each player's hands train, last 30% test).
Baseline tables b_p(c, .) come from training data (shrunk to the population, kappa = 10).

  drift     no switching inside a session: every session is tight, normal or loose
  free      2 states (normal, loose) with a free transition matrix
  io        the same with transitions that depend on the input u (first hand after a big
            loss / big win has ended); warm-started from `free`, so any gain is due to u
  placebo   `io` with the inputs moved to random hands of the same session (20 draws)
  bootstrap over players for confidence intervals of the `io` transition probabilities
  event study on test data with the filtered posterior of the loose state of `free`

`python code/exp_shift.py x3 trigger` instead runs a variant in which the event states
can ONLY be entered at an event (structural zeros); see the appendix of the report.

Usage: python code/exp_shift.py <alphabet>
"""
from common import *
import exp_personal as ep
from data import arrival_inputs, find_events, BIG
from exp_iohmm import shuffled
from exp_tilt import windows, effect, boot, PRE, PLOT_POST
from hmm import SeqData
from shift_hmm import ShiftHMM

_c = {}


def setup(alphabet):
    if alphabet not in _c:
        S = seqs()
        tr, te = ep.split_sessions(S)
        P = int(S.s_player.max()) + 1
        M = 8 if alphabet == "x8" else 3
        X = getattr(S, alphabet)
        cnt = np.zeros((10 * P, M))
        for s, a, L in tr:
            x = X[a:a + L]; ok = x >= 0
            np.add.at(cnt, (S.ctx[a:a + L][ok].astype(np.int64) + 10 * int(S.s_player[s]), x[ok]), 1.0)
        pop = cnt.reshape(P, 10, M).sum(0)
        pop = pop / pop.sum(1, keepdims=True)
        base = cnt + ep.KAPPA * np.tile(pop, (P, 1))
        base /= base.sum(1, keepdims=True)
        _c[alphabet] = (S, tr, te, base, arrival_inputs(S))
    return _c[alphabet]


STOL = 1e-9          # these effects are small: use a much tighter stopping rule than for the pooled HMM


def trigger_init(M, sign=1.0):
    """Structural zeros are kept by EM.  States: 0 normal, 1 after-loss, 2 after-win."""
    th = np.zeros((3, M)); th[1, 1:] = 0.2 * sign; th[2, 1:] = -0.1 * sign
    A = np.zeros((3, 3, 3))
    A[0] = [[1, 0, 0], [0.03, 0.97, 0], [0.03, 0, 0.97]]            # no event: can only decay to normal
    A[1] = [[0.6, 0.4, 0], [0.1, 0.9, 0], [0.5, 0.5, 0]]            # a big loss just ended
    A[2] = [[0.6, 0, 0.4], [0.5, 0, 0.5], [0.1, 0, 0.9]]            # a big win just ended
    return th, A, np.array([1.0, 0.0, 0.0])


def fit(job):
    alphabet, K, U = job["alphabet"], job["K"], job.get("U", 1)
    S, tr, te, base, u = setup(alphabet)
    M = base.shape[1]
    if job.get("shuffle") is not None:
        u = shuffled(S, u, np.random.default_rng(job["shuffle"]))
    if job.get("boot") is not None:
        rng = np.random.default_rng(job["boot"])
        by = {}
        for piece in tr:
            by.setdefault(S.s_player[piece[0]], []).append(piece)
        players = np.array(sorted(by))
        tr = [piece for p in rng.choice(players, len(players)) for piece in by[p]]
    uu = u if U > 1 else None
    d = ep.make(S, tr, alphabet, uu)
    th = np.zeros((K, M))
    if K == 2:
        th[1, 1:] = 0.3
    elif K == 3:
        th[1, 1:] = -0.3; th[2, 1:] = 0.3
    m = ShiftHMM(base, th, U=U, fix_A=job.get("static", False))
    if job.get("static"):
        m.A = np.eye(K)[None]
    if job.get("init") is not None:                 # explicit start: (theta, A, pi)
        th, A, pi = job["init"]
        m.theta = np.array(th, dtype=float); m._set_B(); m.pi = np.array(pi, dtype=float); m.A = np.array(A, dtype=float)
    t0 = time.time()
    m.fit(d, n_iter=job.get("n_iter", 400), tol=STOL)
    out = dict({k: v for k, v in job.items() if k != "init"}, theta=m.theta, A=m.A, pi=m.pi, train=m.history[-1] / d.n_obs,
               train_obs=d.n_obs, n_iter=len(m.history), seconds=time.time() - t0, history=m.history)
    if job.get("boot") is None:
        dt = ep.make(S, te, alphabet, uu)
        out["test"] = m.score(dt) / dt.n_obs; out["test_obs"] = dt.n_obs
        print(job.get("name", ""), "train %.6f test %.6f iters %d (%.0fs)" % (out["train"], out["test"], out["n_iter"], out["seconds"]),
              "theta", np.round(m.theta[1:], 3).tolist(), "A", np.round(m.A, 4).tolist(), flush=True)
    return out


def summarize_trigger(A, theta):
    A = np.array(A); theta = np.array(theta)
    return dict(p_enter_loss=A[1, 0, 1], p_stay_after_loss_event=A[1, 1, 1], dwell_loss=1 / max(1 - A[0, 1, 1], 1e-9),
                p_enter_win=A[2, 0, 2], dwell_win=1 / max(1 - A[0, 2, 2], 1e-9), theta_loss=theta[1], theta_win=theta[2])


def trigger_main(alphabet):
    S, tr, te, base, u = setup(alphabet)
    M = base.shape[1]
    th, A, pi = trigger_init(M)
    jobs = [dict(name="trigger", alphabet=alphabet, K=3, U=3, init=(th, A, pi), n_iter=250)]
    jobs += [dict(name="placebo", alphabet=alphabet, K=3, U=3, shuffle=100 + i, init=(th, A, pi), n_iter=250) for i in range(20)]
    r = pmap(fit, jobs)
    dump(dict(trigger=r[0], placebo=[dict(train=x["train"], test=x["test"], theta=x["theta"], A=x["A"]) for x in r[1:]]), f"shift_{alphabet}_trigger.json")


def main(alphabet):
    S, tr, te, base, u = setup(alphabet)
    M = base.shape[1]
    out = {"alphabet": alphabet}
    ref = ShiftHMM(base, np.zeros((1, M)))
    d_tr, d_te = ep.make(S, tr, alphabet), ep.make(S, te, alphabet)
    out["k1_train"], out["k1_test"] = ref.score(d_tr) / d_tr.n_obs, ref.score(d_te) / d_te.n_obs
    n_tr, n_te = d_tr.n_obs, d_te.n_obs
    out["train_obs"], out["test_obs"] = n_tr, n_te
    print("K=1 baseline: train %.6f test %.6f" % (out["k1_train"], out["k1_test"]), flush=True)
    r = pmap(fit, [dict(name="drift2", alphabet=alphabet, K=2, static=True), dict(name="drift3", alphabet=alphabet, K=3, static=True)])
    res = {x["name"]: x for x in r}
    d2 = res["drift2"]
    res["free2"] = fit(dict(name="free2", alphabet=alphabet, K=2, init=(d2["theta"], [[[0.998, 0.002], [0.002, 0.998]]], d2["pi"])))
    f2 = res["free2"]
    start = (f2["theta"], np.repeat(np.array(f2["A"]), 3, axis=0), f2["pi"])
    jobs = [dict(name="io2", alphabet=alphabet, K=2, U=3, init=start)]
    jobs += [dict(name="placebo", alphabet=alphabet, K=2, U=3, shuffle=100 + i, init=start) for i in range(20)]
    r = pmap(fit, jobs)
    res["io2"] = io = r[0]
    out["fits"] = res
    out["placebo"] = [dict(train=x["train"], test=x["test"], theta=x["theta"], A=x["A"]) for x in r[1:]]
    boots = pmap(fit, [dict(alphabet=alphabet, K=2, U=3, boot=500 + i, n_iter=40, init=(io["theta"], io["A"], io["pi"])) for i in range(100)])
    BA = np.stack([b["A"] for b in boots]); BT = np.stack([b["theta"] for b in boots])
    A = np.array(io["A"])
    out["boot_A"], out["boot_theta"] = BA, BT
    summ = dict(enter=A[:, 0, 1], enter_ci=np.quantile(BA[:, :, 0, 1], [0.025, 0.975], axis=0).T,
                leave=A[:, 1, 0], leave_ci=np.quantile(BA[:, :, 1, 0], [0.025, 0.975], axis=0).T,
                enter_loss_over_win_ci=np.quantile(BA[:, 1, 0, 1] / BA[:, 2, 0, 1], [0.025, 0.975]),
                enter_loss_over_none_ci=np.quantile(BA[:, 1, 0, 1] / BA[:, 0, 0, 1], [0.025, 0.975]),
                theta=np.array(io["theta"])[1], theta_ci=np.quantile(BT[:, 1], [0.025, 0.975], axis=0).T,
                placebo_enter=np.array([np.array(x["A"])[:, 0, 1] for x in out["placebo"]]),
                placebo_leave=np.array([np.array(x["A"])[:, 1, 0] for x in out["placebo"]]))
    out["io_summary"] = summ
    print("P(enter loose) none/loss/win:", np.round(summ["enter"], 5).tolist(), "CI", np.round(summ["enter_ci"], 5).tolist())
    print("   placebo mean:", np.round(summ["placebo_enter"].mean(0), 5).tolist(), "placebo max:", np.round(summ["placebo_enter"].max(0), 5).tolist())
    print("P(leave loose) none/loss/win:", np.round(summ["leave"], 5).tolist(), "CI", np.round(summ["leave_ci"], 5).tolist())
    print("   placebo mean:", np.round(summ["placebo_leave"].mean(0), 5).tolist())
    print("enter loss/win ratio CI", np.round(summ["enter_loss_over_win_ci"], 2).tolist(), " loss/none ratio CI", np.round(summ["enter_loss_over_none_ci"], 1).tolist())
    print("theta", np.round(summ["theta"], 3).tolist(), "CI", np.round(summ["theta_ci"], 3).tolist())
    for name in ("drift2", "drift3", "free2", "io2"):
        x = res[name]
        print("%-8s gain over K=1 (nats): train %8.1f  test %8.1f   per decision: test %+.6f" % (name, (x["train"] - out["k1_train"]) * n_tr, (x["test"] - out["k1_test"]) * n_te, x["test"] - out["k1_test"]))
    g_tr, g_te = (io["train"] - f2["train"]) * n_tr, (io["test"] - f2["test"]) * n_te
    p_tr = [(x["train"] - f2["train"]) * n_tr for x in out["placebo"]]; p_te = [(x["test"] - f2["test"]) * n_te for x in out["placebo"]]
    out["io_gain"] = dict(train=g_tr, test=g_te, placebo_train=p_tr, placebo_test=p_te)
    print("gain of io over free (nats): train %.1f (placebo mean %.1f, max %.1f); test %.1f (placebo mean %.1f, max %.1f)" % (g_tr, np.mean(p_tr), np.max(p_tr), g_te, np.mean(p_te), np.max(p_te)), flush=True)

    # ---- event study with the free 2-state shift HMM (it is not told about losses)
    h = res["free2"]
    m = ShiftHMM(base, np.array(h["theta"])); m.A = np.array(h["A"]); m.pi = np.array(h["pi"])
    allp = [(s, int(S.start[s]), int(S.length[s])) for s in range(S.S)]
    d_all = ep.make(S, allp, alphabet)
    filt = np.concatenate(m.filtered(d_all))[:, 1].astype(np.float32)
    smooth = np.concatenate(m.posteriors(d_all))[:, 1].astype(np.float32)
    np.save(os.path.join(RES, f"shift_{alphabet}_filtered.npy"), filt)
    np.save(os.path.join(RES, f"shift_{alphabet}_smoothed.npy"), smooth)
    row_test = np.zeros(len(S.net), bool)
    for s, a, L in te:
        row_test[a:a + L] = True
    P = int(S.s_player.max()) + 1
    rng = np.random.default_rng(7)
    evs, ws = {}, {}
    for kind in ("loss", "win", "neutral"):
        ev = find_events(S, BIG, PRE, PLOT_POST, kind, rng=np.random.default_rng(11))
        keep = row_test[ev["i"] - PRE]
        evs[kind] = {k: v[keep] for k, v in ev.items()}
    study = {"n": {k: len(v["i"]) for k, v in evs.items()}}
    for name, y in (("filt", filt), ("smooth", smooth), ("vpip", S.vpip.astype(np.float32))):
        ws = {k: windows(y, v) for k, v in evs.items()}
        b = boot([(effect(ws[k]), S.s_player[evs[k]["s"]]) for k in ("loss", "win", "neutral")], P, rng)
        study[name] = {"pre": [float(w[:, :PRE].mean()) for w in ws.values()], "effect": b["mean"], "effect_ci": b["ci"],
                       "loss_minus": b["diff"], "loss_minus_ci": b["diff_ci"], "loss_minus_p": b["diff_p"],
                       "curve": {k: w.mean(0) for k, w in ws.items()}}
        print(name, "events", study["n"], "pre", np.round(study[name]["pre"], 4), "effect", np.round(b["mean"], 4),
              "loss-win %+.4f [%+.4f, %+.4f]" % (b["diff"][0], *b["diff_ci"][0]), "loss-neutral %+.4f [%+.4f, %+.4f]" % (b["diff"][1], *b["diff_ci"][1]))
    out["study"] = study
    out["occupancy_test"] = float(smooth[row_test].mean())
    # implied VPIP shift of the loose state for an average player
    okr = getattr(S, alphabet) >= 0
    w = np.bincount(S.ctx[okr].astype(np.int64) + 10 * S.s_player[np.repeat(np.arange(S.S), S.length)][okr], minlength=base.shape[0]).astype(float)
    w /= w.sum()
    nf = [float(((1 - mm[:, 0]) * w).sum()) for mm in (m.B[:, 0], m.B[:, 1])]
    out["nonfold_normal_loose"] = nf
    print("share of non-fold decisions: normal %.4f, loose %.4f; loose-state occupancy on test %.4f" % (nf[0], nf[1], out["occupancy_test"]))
    dump(out, f"shift_{alphabet}.json")


if __name__ == "__main__":
    (trigger_main if len(sys.argv) > 2 and sys.argv[2] == "trigger" else main)(sys.argv[1])
