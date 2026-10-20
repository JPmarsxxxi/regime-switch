# regime-switch

Four regime-switching papers, each re-implemented as written, checked against simulated ground truth, and replicated on free data.

| Division | Paper | Code | Ground truth | Replication |
|---|---|---|---|---|
| Jump model | Shu, Yu & Mulvey (2024) | `regime_switch/shu2024.py`, `jump.py` | `reports/sim_shu2024.md` | `reports/replicate_shu2024.md` |
| Classical HMM | Nystrup, Madsen & Lindström (2018) | `regime_switch/nystrup2018.py` | `reports/sim_nystrup2018.md` | `reports/replicate_nystrup2018.md` |
| Distribution clustering | Horvath, Issa & Muguruza (2021) | `regime_switch/horvath2021.py` | `reports/sim_horvath2021.md` (this is also their replication: the paper's experiments are synthetic) | - |
| Strategy switching | Daniel & Moskowitz, *Momentum Crashes* | `regime_switch/daniel_moskowitz.py` | `reports/sim_daniel_moskowitz.md` | `reports/replicate_daniel_moskowitz.md` |

- `SPEC.md`: what was built and the acceptance criteria.
- `DEVIATIONS.md`: every place the code departs from a paper or fills a gap it leaves, with the reason.
- `papers/README.md`: links to the papers (PDFs are not committed).

## Running

```
pip install -e ".[dev]"
pytest -q                 # unit tests
pytest -q -m slow         # simulation criteria (reuses cached study results when the code is unchanged)
python -m scripts.sim_shu2024            # and sim_nystrup2018, sim_horvath2021, sim_daniel_moskowitz
python -m scripts.replicate_shu2024      # and replicate_nystrup2018, replicate_daniel_moskowitz
```

Studies cache their raw results in `reports/data/` (gitignored), keyed on a hash of the source files they depend on; downloads are cached in `data_cache/`.
