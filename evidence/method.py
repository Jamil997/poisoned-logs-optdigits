"""Two-source evidence gates. This module receives no population quality parameters.

A main sample may have edited rewards; an independently collected reference sample
is trusted in the main experiment. Both samples come from the same population.
All gates test positive paired improvement. Scores are approval margins, not
confidence lower bounds on the magnitude of improvement.
"""
from functools import lru_cache
from fractions import Fraction
from math import gcd
import numpy as np
from scipy.stats import binom

GUARD = 1e-12

def integer_weights(n_main, n_trusted, fraction):
    f = Fraction(str(fraction)).limit_denominator(100000)
    if n_trusted == 0 or f == 0:
        return 1, 0, n_main
    if n_main == 0 or f == 1:
        return 0, 1, n_trusted
    a = (f.denominator - f.numerator) * n_trusted
    c = f.numerator * n_main
    d = gcd(a, c)
    a, c = a // d, c // d
    return a, c, a * n_main + c * n_trusted

@lru_cache(None)
def distribution(n, p):
    v = np.arange(n + 1)
    return v, binom.pmf(v, n, p)

def tail(h, n_main, n_trusted, a, c, p=.5):
    """P[a U + c V >= h] for independent binomial counts, without rounding h."""
    if a == 0:
        return float(binom.sf((int(h) - 1) // c, n_trusted, p))
    if c == 0:
        return float(binom.sf((int(h) - 1) // a, n_main, p))
    v, pmf = distribution(n_trusted, p)
    need = (int(h) - 1 - c * v) // a
    return float(np.sum(pmf * binom.sf(need, n_main, p), dtype=np.longdouble))

@lru_cache(None)
def threshold(n_main, n_trusted, a, c, risk):
    """Smallest integer rejection threshold with guarded null tail <= risk."""
    lo, hi = 0, a * n_main + c * n_trusted + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if tail(mid, n_main, n_trusted, a, c) + GUARD <= risk:
            hi = mid
        else:
            lo = mid + 1
    return lo

def damage(a, c, k, trusted_cap=0, n_main=None):
    """Largest weighted count change for one total budget, cap on trusted edits.

Experiments have enough main rows for all k edits. A positive trusted cap permits
redistributing the SAME k edits, not adding an independent attack budget.
"""
    if n_main == 0:
        return c * min(k, trusted_cap), 0, min(k, trusted_cap)
    kt = min(k, trusted_cap) if c > a else 0
    return a * (k - kt) + c * kt, k - kt, kt

@lru_cache(None)
def design(n_main, n_trusted, k, alpha, trusted_cap=0):
    """Minimize an attacked mean-approval threshold, not oracle expected utility.

The finite search uses only sizes, threat budget and significance level. Every
candidate has the SAME standardized weight. It is frozen before any outcomes.
"""
    total = n_main + n_trusted
    grid = {Fraction(i, 20) for i in range(21)} | {Fraction(n_trusted, total)}
    if n_trusted == 0: grid = {Fraction(0)}
    if n_main == 0: grid = {Fraction(1)}
    rows = []
    for f in sorted(grid):
        a, c, d = integer_weights(n_main, n_trusted, f)
        h = threshold(n_main, n_trusted, a, c, alpha / 2)
        bias, _, _ = damage(a, c, k, trusted_cap, n_main)
        objective = (h - .5 + 2 * bias) / d - .5
        rows.append(dict(fraction=float(f), a=a, c=c, denominator=d,
                         threshold=h, objective=objective))
    return min(rows, key=lambda r: (r['objective'], r['fraction'])), rows


def weights_for(method, n_main, n_trusted, k, alpha, B, trusted_favorable,
                declared_trusted_cap=0):
    """Return integer coefficients, per-policy thresholds and correction.

No beta, theta, true gain, good-policy index, or evaluation labels are accepted.
Trusted pilot methods allocate conditional on their independent observations.
"""
    count = trusted_favorable.shape[1]
    risks = np.full((2, count), alpha / 2)
    if method.startswith('evidence_'):
        best, _ = design(n_main, n_trusted, k, alpha, declared_trusted_cap)
        a, c, d = best['a'], best['c'], best['denominator']
    elif method == 'trusted_only':
        a, c, d = 0, 1, max(1, n_trusted)
    elif method in ('main_only', 'pilot_select_main', 'pilot_weight_main'):
        a, c, d = 1, 0, max(1, n_main)
    else:
        a, c, d = 1, 1, n_main + n_trusted
    if method == 'pooled_influence':
        risks[:] = (alpha / np.asarray(B) / (1 / np.asarray(B)).sum())[:, None]
    if n_trusted and method in ('pilot_select_main', 'pilot_weight_main'):
        # Standardized improvement; no use of known candidate truth or influence.
        diff = trusted_favorable[0] - trusted_favorable[1]
        if method == 'pilot_select_main':
            risks[:] = 0
            risks[0, diff >= 0] = alpha
            risks[1, diff < 0] = alpha
        else:
            # Fixed soft weights: 90/10 in the observed direction, 50/50 on ties.
            risks[0, diff > 0] = .9 * alpha
            risks[1, diff > 0] = .1 * alpha
            risks[0, diff < 0] = .1 * alpha
            risks[1, diff < 0] = .9 * alpha
    h = np.empty((2, count), dtype=np.int64)
    for risk in np.unique(risks):
        hh = (a * n_main + c * n_trusted + 1 if risk == 0 else
              threshold(n_main, n_trusted, a, c, float(risk)))
        h[risks == risk] = hh
    corr, _, _ = damage(a, c, k, declared_trusted_cap, n_main)
    if method == 'evidence_nominal': corr = 0
    if method == 'trusted_only' and n_trusted == 0: h[:] = 1
    return a, c, d, h, corr


def margins(main_favorable, trusted_favorable, B, rule):
    a, c, d, h, corr = rule
    return np.asarray(B)[:, None] * (a * main_favorable + c * trusted_favorable - corr - h + .5) / d


def select(scores):
    ix = np.argmax(scores, axis=0)
    return np.where(np.max(scores, axis=0) > 0, ix, -1)
