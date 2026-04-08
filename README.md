# Tiny-network experiments

A workspace for running small, fast neural-network experiments and recording confirmed findings.

## Structure

```
.
├── experiments/    # All experiment scripts and raw result JSON files.
│                   # Anything here is exploratory - may or may not be a real finding.
│
└── findings/       # Confirmed findings only. Each one has survived robustness
                    # checks (multiple seeds, alternate baselines, replication
                    # on a second setting, novelty check).
```

## Workflow

1. Build an experiment in `experiments/experimentN.py`.
2. Run it. Save raw output to `experiments/results_*.json`.
3. If the result looks interesting, write a robustness experiment that tries to **kill it** (untrained baseline, parameter sweeps, second task).
4. **Only if the finding survives all the kill attempts**, promote it to `findings/NN_short_name.md` with the data, the simple-words explanation, the scope limits, and the reproduction steps.
5. The criteria for promotion are listed in `findings/README.md`.

## Running

All scripts are pure NumPy and run on CPU in seconds to a few minutes.

```bash
cd experiments
python3 experiment4.py   # quick demo of concept attractor counting
python3 experiment5.py   # full robustness sweep (~3 minutes)
```

Each script writes its own `results_*.json` next to itself.
