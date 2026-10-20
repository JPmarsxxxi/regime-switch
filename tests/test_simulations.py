"""SPEC criteria 13-18: the simulation studies. Each study is cached on the source of the code it depends on, so this
reuses a run of `python scripts/sim_<paper>.py` on the same code and recomputes otherwise."""

import pytest

pytestmark = pytest.mark.slow


def _check(crit):
    failed = {k: txt for k, (ok, txt) in crit.items() if not ok}
    assert not failed, failed


def test_shu2024_ground_truth():
    from scripts import sim_shu2024 as s

    _check(s.criteria(s.results()[0]))


def test_nystrup2018_ground_truth():
    from scripts import sim_nystrup2018 as s

    _check(s.criteria(s.results()))


def test_horvath2021_ground_truth():
    from scripts import sim_horvath2021 as s

    _check(s.criteria(s.results()))


def test_daniel_moskowitz_ground_truth():
    from scripts import sim_daniel_moskowitz as s

    _check(s.criteria(s.results()))
