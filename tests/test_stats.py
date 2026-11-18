import numpy as np

from pno.stats.bootstrap import contrast_bootstrap, noninferior, paired_diff


def test_paired_diff_detects_effect():
    rng = np.random.default_rng(0)
    n = 1000
    base = rng.random((3, n)) < 0.6
    better = base | (rng.random((3, n)) < 0.25)
    r = paired_diff({"a": better, "b": base}, "a", "b", n_boot=2000)
    assert r["ci_low"] > 0 and r["ci_low"] < r["estimate"] < r["ci_high"]
    assert abs(r["estimate"] - (better.mean() - base.mean())) < 1e-12


def test_no_effect_ci_covers_zero_and_pairing_tightens():
    rng = np.random.default_rng(1)
    n = 800
    item = rng.random(n)
    a = (rng.random((3, n)) < item).astype(float)
    b = (rng.random((3, n)) < item).astype(float)
    r = paired_diff({"a": a, "b": b}, "a", "b", n_boot=2000)
    assert r["ci_low"] < 0 < r["ci_high"]
    assert r["ci_high"] - r["ci_low"] < 0.12


def test_interaction_contrast():
    n = 500
    z = np.zeros((3, n))
    o = np.ones((3, n))
    r = contrast_bootstrap({"c1": z, "c2": z, "c3": z, "c4": o}, {"c4": 1, "c3": -1, "c2": -1, "c1": 1},
                           n_boot=500)
    assert r["estimate"] == 1.0 and r["ci_low"] == 1.0


def test_noninferiority():
    assert noninferior({"ci_low": -0.01, "ci_high": 0.02}, 0.02)
    assert not noninferior({"ci_low": -0.03, "ci_high": 0.0}, 0.02)
    assert noninferior({"ci_low": -0.01, "ci_high": 0.015}, 0.02, higher_is_better=False)
