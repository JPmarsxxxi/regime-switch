# regime-switch

Four regime-switching papers re-implemented exactly as written, each validated against simulated ground truth and replicated on free real data. `SPEC.md` is the build spec; `DEVIATIONS.md` logs every place the code departs from a paper or fills a gap the paper leaves; `papers/README.md` links the sources.

| Division | Paper | Module |
|---|---|---|
| Jump model | Shu, Yu & Mulvey (2024) | `regime_switch/shu2024.py`, `regime_switch/jump.py` |
| Classical HMM | Nystrup, Madsen & Lindström (2018) | `regime_switch/nystrup2018.py` |
| Distribution clustering | Horvath, Issa & Muguruza (2021) | `regime_switch/horvath2021.py` |
| Strategy switching | Daniel & Moskowitz, *Momentum Crashes* | `regime_switch/daniel_moskowitz.py` |

## Tech stack & commands

- Python 3.11+, numpy/pandas/scipy, numba for the hot loops, cvxpy for MPC. Reference packages used only as test oracles: `jumpmodels`, `hmmlearn`, `arch`.
- Install: `pip install -e ".[dev]"`
- Fast tests: `pytest -q`
- Simulation (ground truth) tests: `pytest -q -m slow`
- Lint: `ruff check .`
- Simulation reports: `python scripts/sim_<paper>.py` -> `reports/sim_<paper>.md`
- Real-data replication: `python scripts/replicate_<paper>.py` -> `reports/replicate_<paper>.md` (downloads are cached in `data_cache/`, gitignored)

## Code style

- Lean, precise, concise code. No unnecessary abstractions — three similar lines beats a premature helper.
- No emojis, in code, comments, or commit messages.
- No comments explaining WHAT code does. Only comment a non-obvious WHY. Equation references (`# eq. 7`) are welcome: they tie code to the paper.

## Workflow contract

- For anything beyond a small fix: write or update `SPEC.md` via `/spec` before implementation starts.
- Before calling a feature/task done, run `/verify` against `SPEC.md`.
- List the steps you're about to take before writing code on anything nontrivial.
- Build in parts with a checkpoint between them; one paper per commit series, not one giant diff.
- Never commit to `main`/`master` unless explicitly told to. Confirm the current branch before committing.
- Never force-push, rebase, or run other history-rewriting git commands without asking first.

## Project-specific rules

- **Copy the paper, don't improve it.** Every equation, window, delay, cost and metric comes from the paper. If the paper is silent or the data is unavailable, make the smallest reasonable choice, add an entry to `DEVIATIONS.md`, and reference it at the code site with `# see DEVIATIONS.md#<id>`.
- **Truth before tables.** A method is not trusted on real data until it passes its simulation tests, where the true regimes are known.
- **Report failures as failures.** A replication that misses the paper's number is written up as NOT REPRODUCED with the numbers. Never tune hyperparameters on the test period to close the gap.
- **Oracles, not dependencies.** `jumpmodels`, `hmmlearn` and `arch` may be imported in `tests/` to cross-check, and `hmmlearn` in `shu2024.py` because the paper itself uses it. Nothing else in `regime_switch/` depends on them.
- Everything that takes randomness takes a seed.
