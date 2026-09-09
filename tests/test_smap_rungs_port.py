"""The ported mechanisms do what the source says, on data built to make them fire.

`docs/MODELS.md` 27.8 measured arm `L1` identical to `A0` in every cell, which
would say the magnitude conjunct (`third_party/telemanom/telemanom/errors.py:342-343`)
changed nothing on SMAP/MSL. That is only a finding about the data if the
transcription is right, so these tests construct windows where each mechanism
*must* bind and check that it does. Data and transcription, separated.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "smap_rungs", ROOT / "scripts" / "smap_rungs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["smap_rungs"] = module           # dataclasses need it registered
    spec.loader.exec_module(module)
    return module


M = _load()


def _window(peak: float, spread: float, n: int = 600):
    """A quiet error series with one excursion, over values of a known spread.

    `inter_range` is the 95th minus the 5th percentile of the raw values, so
    `spread` sets the conjunct's cut at `0.05 * spread` directly.
    """
    rng = np.random.default_rng(11)
    e_s = np.abs(rng.normal(0.0, 0.001, n)) + 0.001
    e_s[300:340] = peak
    values = np.linspace(-spread / 2.0, spread / 2.0, n)
    return e_s.astype(np.float64), values.astype(np.float64)


def _anoms(e_s, values, mech):
    eps = M.find_epsilon(e_s, float(np.mean(e_s)), float(np.std(e_s)),
                         M.telemanom.ERROR_BUFFER, mech)
    i_anom, _, _ = M.compare_to_epsilon(
        e_s, e_s, eps, values, 0, 70, 0, np.array([], dtype=np.int64), 0,
        M.telemanom.ERROR_BUFFER, mech, None)
    return i_anom


def test_the_magnitude_conjunct_removes_a_candidate_below_the_cut():
    """errors.py:343 -- `e_s > 0.05 * inter_range`. Peak 0.5 against a cut of 1.0."""
    e_s, values = _window(peak=0.5, spread=40.0)          # cut = 0.05 * 40 = 2.0
    assert _anoms(e_s, values, M.Mech(magnitude=False)).size > 0
    assert _anoms(e_s, values, M.Mech(magnitude=True)).size == 0


def test_the_magnitude_conjunct_keeps_a_candidate_above_the_cut():
    """The control: the same excursion against a spread that puts it over."""
    e_s, values = _window(peak=0.5, spread=0.2)            # cut = 0.05 * 0.2 = 0.01
    assert _anoms(e_s, values, M.Mech(magnitude=False)).size > 0
    assert _anoms(e_s, values, M.Mech(magnitude=True)).size > 0


def test_the_whole_window_bailout_silences_a_window_whose_errors_are_tiny():
    """errors.py:337-340 -- max(e_s) must clear 0.05 absolute as well."""
    e_s, values = _window(peak=0.02, spread=0.05)
    assert _anoms(e_s, values, M.Mech()).size > 0
    assert _anoms(e_s, values, M.Mech(bailout=True)).size == 0


def test_the_coverage_guard_rejects_a_candidate_that_floods_its_window():
    """errors.py:315 -- `len(i_anom) < len(e_s) * 0.5`.

    26.30.2's regime: one exceedance buffered by +/-99 covers 199 samples, so in
    a window shorter than ~400 the guard is the difference between a candidate
    and none.
    """
    e_s, values = _window(peak=5.0, spread=0.2, n=260)
    guarded = _anoms(e_s, values, M.Mech(guards=True))
    plain = _anoms(e_s, values, M.Mech(guards=False))
    assert plain.size > len(e_s) * 0.5
    assert guarded.size < plain.size


def test_the_inverse_pass_only_ever_adds():
    """errors.py:132-148 -- the two index sets are unioned, never intersected."""
    e_s, values = _window(peak=5.0, spread=0.2)
    base = M.Mech(magnitude=True, bailout=True, guards=True)
    without, _ = M.run_ladder(e_s, values, M.published_config(),
                              M.telemanom.PRUNING_P, base)
    with_inv, _ = M.run_ladder(e_s, values, M.published_config(),
                               M.telemanom.PRUNING_P,
                               M.replace(base, inverse=True))
    assert not (without & ~with_inv).any()


def test_the_clip_is_what_makes_the_union_permissive():
    """errors.py:355-359 against 26.29.1: clipped is a subset of unclipped."""
    e_s, values = _window(peak=5.0, spread=0.2, n=3000)
    clipped, _ = M.run_ladder(e_s, values, M.proportional_config(3000),
                              M.telemanom.PRUNING_P, M.Mech(clip=True))
    unclipped, _ = M.run_ladder(e_s, values, M.proportional_config(3000),
                                M.telemanom.PRUNING_P, M.Mech(clip=False))
    assert not (clipped & ~unclipped).any()
    assert unclipped.sum() > clipped.sum()
