# Latent playing styles and tilt in poker with HMMs

Code for the CSE 150A/250A final project (group "Bob's Clan"). Python 3.13.

## Setup

```
python -m venv .venv && .venv/bin/pip install -r code/requirements.txt
```

Data: download the 37 parquet shards of the `takara-ai/poker_hands` dataset from
Hugging Face into `data/raw/` (1.3 GB). Only the shards holding PartyPoker and
Pluribus hands are read.

## Files

| File | What it does |
|---|---|
| `hmm.py` | The HMM: scaled forward-backward, Baum-Welch over many sequences, Viterbi, context-dependent emissions, input-dependent transitions, per-player transitions, missing observations. NumPy only. |
| `shift_hmm.py` | HMM whose states are log-odds shifts of each player's own baseline (generalised EM with Newton steps). |
| `test_hmm.py` | Tests: brute-force enumeration, agreement with `hmmlearn`, parameter recovery. |
| `build_dataset.py` | Raw hands -> one row per (player, hand): decision, context, net result. |
| `survey_sites.py` | Share of hands with missing winnings per site (why PartyPoker is used). |
| `validate_dataset.py` | Data checks quoted in the report (stack continuity, session statistics). |
| `prepare_sequences.py` | Sessions, action symbols, train / held-out splits. |
| `data.py`, `common.py` | Loading, event finding, fitting with restarts. |
| `exp_sweep.py` | Number of states K; no-hidden-state and no-switching baselines. |
| `exp_validate.py` | What the states are; validation against VPIP, PFR, aggression and win rate. |
| `exp_tilt.py` | Event study around big losses with big-win and neutral controls. |
| `exp_iohmm.py` | Input-output HMM on the pooled model, placebo inputs, bootstrap. |
| `exp_ablations.py` | Alphabet, context, sequence length, pooled vs per-player. |
| `exp_personal.py` | Per-player emission tables with free hidden modes. |
| `exp_shift.py` | Shift HMM: session drift, free two-state model, input-dependent transitions, shuffled-input placebos, bootstrap, event study. `python code/exp_shift.py x3 trigger` runs the event-only variant described in the appendix. |
| `exp_pluribus.py` | Negative control: the Pluribus bot. |
| `make_figures.py` | Figures from `results/*.json`. |

## Reproduce

```
python code/test_hmm.py
python -I code/survey_sites.py data/raw
python -I code/build_dataset.py data/raw data/player_hands.parquet
python -I code/validate_dataset.py data/player_hands.parquet
python -I code/prepare_sequences.py data/player_hands.parquet data/sequences.npz
python code/exp_sweep.py
echo '{"tag": "hmm_x8_ctx_K4_s0", "K": 4}' > results/main.json
python code/exp_tilt.py hmm_x8_ctx_K4_s0
python code/exp_validate.py hmm_x8_ctx_K4_s0
python code/exp_iohmm.py hmm_x8_ctx_K4_s0
python code/exp_ablations.py 4
python code/exp_personal.py tune && python code/exp_personal.py fit
python code/exp_shift.py x3
python code/exp_pluribus.py hmm_x8_ctx_K4_s0 data/raw/train-00000-of-00037.parquet
python code/make_figures.py
```

All random seeds are fixed in the scripts. The full pipeline takes about two hours
on a 15-core laptop; one EM iteration over the 2M training hands takes 0.4 s.
The report sources are in `report/` (`pdflatex main.tex`, twice); figures are written to `figures/`.
