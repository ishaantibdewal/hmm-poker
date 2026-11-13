"""Draw the report figures from the JSON files in results/.   Usage: python code/make_figures.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from data import ROOT, CTX, SYM8

RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
STATE_COL = [BLUE, ORANGE, AQUA, VIOLET]
STATE_NAME = ["S1 tight, big raises", "S2 aggressive, pot raises", "S3 aggressive, small raises", "S4 loose-passive"]
STATE_SHORT = ["S1", "S2", "S3", "S4"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=170, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_selection():
    r = load("sweep.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.7), gridspec_kw={"wspace": 0.62})
    Ks = range(1, 9)

    def best(K, static, key, obs):
        rr = [x for x in r if x["K"] == K and bool(x.get("static")) == static]
        return max(rr, key=lambda x: x["train_ll"])[key] / max(rr, key=lambda x: x["train_ll"])[obs] if rr else np.nan
    for static, col, lab, mk in ((False, BLUE, "HMM", "o"), (True, ORANGE, "no switching (A = I)", "s")):
        y = [best(K, static, "new_ll", "new_obs") for K in Ks]
        if static:
            y[0] = best(1, False, "new_ll", "new_obs")
        ax[0].plot(Ks, y, color=col, lw=1.6, marker=mk, ms=4, label=lab)
    ax[0].axvline(4, color=MUTED, lw=0.6, ls=":")
    ax[0].set_xlabel("number of states K"); ax[0].set_ylabel("log-lik. per decision")
    ax[0].set_title("(a) Held-out players"); ax[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.06), fontsize=7, handlelength=1.6)
    ax[0].set_xticks(list(Ks))
    # gain from switching
    g = [best(K, False, "new_ll", "new_obs") - best(K, True, "new_ll", "new_obs") for K in range(2, 9)]
    g2 = [best(K, False, "later_ll", "later_obs") - best(K, True, "later_ll", "later_obs") for K in range(2, 9)]
    w = 0.36
    ax[1].bar(np.arange(2, 9) - w / 2 - 0.02, g, w, color=BLUE, label="new players")
    ax[1].bar(np.arange(2, 9) + w / 2 + 0.02, g2, w, color=AQUA, label="later sessions")
    ax[1].axhline(0, color=MUTED, lw=0.6)
    ax[1].set_xlabel("number of states K"); ax[1].set_ylabel("log-lik. gain per decision")
    ax[1].set_title("(b) HMM minus no-switching"); ax[1].legend(fontsize=6.5, loc="lower left", handlelength=1.0); ax[1].set_xticks(range(2, 9)); ax[1].set_ylim(-0.0021, 0.0034)
    # convergence
    for K, col in ((4, BLUE), (6, ORANGE)):
        for x in [x for x in r if x["K"] == K and not x.get("static")]:
            h = np.array(x["history"]) / x["n_obs"]
            ax[2].plot(np.arange(1, len(h) + 1), h, color=col, lw=0.9, alpha=0.85)
    ax[2].set_xscale("log"); ax[2].set_ylim(-0.83, -0.785)
    ax[2].set_xlabel("EM iteration"); ax[2].set_ylabel("log-lik. per decision")
    ax[2].set_title("(c) EM, 5 restarts each"); ax[2].set_xlim(8, 200)
    ax[2].text(60, -0.8045, "K = 4", color=BLUE, fontsize=7.5, ha="left", va="top")
    ax[2].text(38, -0.7925, "K = 6", color=ORANGE, fontsize=7.5, ha="right", va="bottom")
    save(fig, "fig_selection")


def fig_states():
    v = load("validate.json")
    B = np.array(v["B"])
    fig = plt.figure(figsize=(7.0, 1.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.2, 1.2, 1.0], wspace=0.14)
    sym = ["F", "X", "Cs", "Cm", "Cl", "Rs", "Rm", "Rl"]
    for j, (c, title) in enumerate(((2, "(a) Cutoff, unraised"), (4, "(b) Button, unraised"), (9, "(c) Big blind vs. raise"))):
        ax = fig.add_subplot(gs[0, j]); ax.grid(False)
        M = B[c]
        ax.imshow(M, cmap=SEQ, vmin=0, vmax=0.9, aspect="auto")
        for i in range(4):
            for k in range(8):
                if M[i, k] >= 0.005:
                    ax.text(k, i, f"{M[i, k]:.2f}".lstrip("0"), ha="center", va="center", fontsize=6.3, color="white" if M[i, k] > 0.45 else INK)
        ax.set_xticks(range(8)); ax.set_xticklabels(sym, fontsize=7)
        ax.set_yticks(range(4)); ax.set_yticklabels(STATE_SHORT if j == 0 else [])
        ax.tick_params(length=0); ax.set_title(title, fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = fig.add_subplot(gs[0, 3])
    rows = v["by_state"]
    for k, row in enumerate(rows):
        lo, hi = row["winrate_ci"]
        ax.plot([lo, hi], [k, k], color=STATE_COL[k], lw=1.4)
        ax.plot(row["winrate"], k, "o", color=STATE_COL[k], ms=5)
        ax.text(hi + 0.6, k, f"{row['winrate']:+.1f}", va="center", fontsize=7)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_ylim(3.5, -0.5); ax.set_yticks(range(4)); ax.set_yticklabels([]); ax.set_xlim(-11, 7.5)
    ax.set_xlabel("bb won per 100 hands", fontsize=7.5); ax.set_title("(d) Win rate", fontsize=8); ax.grid(axis="y", visible=False)
    save(fig, "fig_states")


MODEL_KEY, MODEL_TITLE = "loose_index", "Pooled HMM"


def fig_tilt():
    t = load("tilt.json")
    r = t["main"]
    off = np.array(t["offsets"])
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.75), gridspec_kw={"wspace": 0.5, "width_ratios": [1.15, 1.15, 0.95]})

    def binned(curve, width=5):
        c = np.array(curve)
        pre, post = c[:30], c[30:]
        xs = np.r_[off[:30].reshape(-1, width).mean(1), off[30:].reshape(-1, width).mean(1)]
        ys = np.r_[pre.reshape(-1, width).mean(1), post.reshape(-1, width).mean(1)]
        return xs, ys
    kinds = (("loss", ORANGE, "after a big loss", "o"), ("win", BLUE, "after a big win", "s"), ("neutral", MUTED, "neutral flop", "^"))
    for a, key, ylab, title in ((ax[0], "vpip", "change in VPIP (points)", "(a) Hands played, raw"),
                                (ax[1], MODEL_KEY, "change in expected VPIP", "(b) " + MODEL_TITLE)):
        for kind, col, lab, mk in kinds:
            xs, ys = binned(r[key]["curve"][kind])
            base = np.mean(np.array(r[key]["curve"][kind])[:30])
            for seg in (xs < 0, xs > 0):
                a.plot(xs[seg], 100 * (ys[seg] - base), color=col, lw=1.5, marker=mk, ms=3.2, label=lab if seg[0] else None)
        a.axvline(0, color=MUTED, lw=0.6, ls=":"); a.axhline(0, color=MUTED, lw=0.6)
        a.set_xlabel("hands relative to the event"); a.set_ylabel(ylab)
        a.set_title(title); a.set_ylim(-0.75, 1.85)
    ax[1].legend(fontsize=6.8, loc="upper left", handlelength=1.5, borderaxespad=0.1)
    # (c) shift HMM with input-dependent transitions: chance that a loose spell starts right after an event
    sh = load("shift_x3.json")["io_summary"]
    a = ax[2]
    pe = 100 * np.array(sh["placebo_enter"])
    for i, (col, lab) in enumerate(((1, "after a\nbig loss"), (2, "after a\nbig win"))):
        a.plot(pe[:, col], np.full(len(pe), i + 0.16), "|", color=MUTED, ms=7, mew=0.9, label="event positions shuffled" if i == 0 else None)
        lo, hi = 100 * np.array(sh["enter_ci"][col])
        a.plot([lo, hi], [i - 0.16, i - 0.16], color=ORANGE if i == 0 else BLUE, lw=1.6)
        a.plot(100 * sh["enter"][col], i - 0.16, "o", color=ORANGE if i == 0 else BLUE, ms=5, label="real events" if i == 0 else None)
        a.text(100 * sh["enter"][col], i - 0.42, f"{100 * sh['enter'][col]:.1f}%" if i == 0 else f"{100 * sh['enter'][col]:.2f}%", ha="center", fontsize=7)
    a.set_yticks([0, 1]); a.set_yticklabels(["after a\nbig loss", "after a\nbig win"], fontsize=7); a.set_ylim(1.55, -0.8); a.set_xlim(-0.4, 5.4)
    a.set_xlabel("P(loose spell starts), %"); a.set_title("(c) Shift HMM"); a.grid(axis="y", visible=False)
    a.text(pe[:, 1].mean(), 0.52, "same events,\npositions shuffled", ha="center", va="top", fontsize=6.3, color=MUTED)
    save(fig, "fig_tilt")


def fig_ablation():
    a = load("ablations.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.9), gridspec_kw={"wspace": 0.55})
    # (a) sequence length
    best = {}
    for r in a["chunks"]:
        key = r["L"] or 400
        if key not in best or r["train_ll"] > best[key]["train_ll"]:
            best[key] = r
    Ls = sorted(best)
    ax[0].plot(Ls, [best[L]["new"] for L in Ls], color=BLUE, lw=1.6, marker="o", ms=4)
    ax[0].set_xscale("log"); ax[0].set_xticks([5, 10, 25, 50, 100, 250, 400]); ax[0].set_xticklabels(["5", "10", "25", "50", "100", "250", "all"], fontsize=6.8)
    ax[0].minorticks_off()
    ax[0].set_xlabel("training sequences cut to (hands)"); ax[0].set_ylabel("held-out log-lik. per decision")
    ax[0].set_title("(a) Sequence length")
    # (b) per-player transitions
    rows = [r for r in a["per_player_A"] if r["kappa"]]
    ax[1].plot([r["kappa"] for r in rows], [r["later"] for r in rows], color=BLUE, lw=1.6, marker="o", ms=4)
    pooled = [r for r in a["per_player_A"] if not r["kappa"]][0]["later"]
    ax[1].axhline(pooled, color=ORANGE, lw=1.4)
    ax[1].text(10, pooled + 0.0004, "one shared A", color=INK, fontsize=7, va="bottom")
    ax[1].text(130, -0.7005, "one A\nper player", color=INK, fontsize=7, va="bottom")
    ax[1].set_xscale("log"); ax[1].set_xlabel("shrinkage towards the shared A"); ax[1].set_ylabel("log-lik., later sessions")
    ax[1].set_title("(b) Per-player transitions"); ax[1].set_ylim(-0.7125, -0.6965)
    # (c) personal model learning curve
    ns = [500, 1000, 2000, 5000, 10000]
    res = a["personal"]
    pers = [np.mean([x["personal"] for x in res if x["n"] == n]) for n in ns]
    pool = np.mean([x["pooled"] for x in res if x["n"] == ns[0]])
    ax[2].plot(ns, pers, color=BLUE, lw=1.6, marker="o", ms=4)
    ax[2].axhline(pool, color=ORANGE, lw=1.4)
    ax[2].text(2500, pool - 0.0015, "pooled model", color=INK, fontsize=7, va="top")
    ax[2].text(520, -0.662, "personal\nmodel", color=INK, fontsize=7, va="bottom")
    ax[2].set_xscale("log"); ax[2].set_xticks(ns); ax[2].set_xticklabels(["500", "1k", "2k", "5k", "10k"]); ax[2].minorticks_off()
    ax[2].set_xlabel("that player's training hands"); ax[2].set_ylabel("log-lik., last 5,000 hands")
    ax[2].set_title("(c) Personal vs. pooled"); ax[2].set_ylim(-0.706, -0.646)
    save(fig, "fig_ablation")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    which = sys.argv[1:] or ["selection", "states", "tilt", "ablation"]
    for w in which:
        if "fig_" + w in globals():
            globals()["fig_" + w]()
            print("drew", w)
