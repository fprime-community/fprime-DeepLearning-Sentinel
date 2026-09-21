"""D76's warm start: the weights that go in are the weights that come out.

`docs/DECISIONS.md` D76 rules that the flown retrainer initialises the shadow
from the flying model's weights and fine-tunes, so "there is no random
initialisation on the flight path". **That had no implementation anywhere in this
stack** -- `lstm.train` took no weights argument, and `GRUForecastDetector`'s
`reuse_weights` is a fit CACHE, not a warm start. `docs/MODELS.md` 74 is the arm
that measures it, and this file is the mechanism's own check.

(!) IT IS NOT A TEST OF THE ARM. It asserts one thing: that
`load_rnn_weights` is the exact inverse of `_rnn_weights`. If it is not, the arm
below would be measuring a reshape rather than a warm start, and it would look
like a result.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sentinel_models import lstm  # noqa: E402


def _hyper() -> lstm.Hyper:
    """Small, and the shape is not the point -- the round trip is."""
    return lstm.Hyper(window=20, hidden=(8, 8), n_predictions=2, max_epochs=1, seed=0)


def _flatten(weights: lstm.Weights) -> np.ndarray:
    parts = []
    for layer in weights.layers:
        parts += [layer.w_ih.ravel(), layer.w_hh.ravel(),
                  layer.b_ih.ravel(), layer.b_hh.ravel()]
    parts += [np.asarray(weights.head_w).ravel(), np.asarray(weights.head_b).ravel()]
    return np.concatenate(parts)


def test_the_loader_is_the_exact_inverse_of_the_extractor() -> None:
    """Out, in, out again -- and the two must be bit-identical, not close."""
    hyper = _hyper()
    lstm.configure_determinism(0)
    source = lstm.build_model(3, hyper)
    original = lstm.to_weights(source)

    lstm.configure_determinism(999)          # a DIFFERENT initialisation
    target = lstm.build_model(3, hyper)
    before = _flatten(lstm.to_weights(target))
    assert not np.array_equal(before, _flatten(original)), (
        "the two models started identical, so this round trip would pass "
        "without the loader doing anything at all")

    lstm.load_rnn_weights(target, original)
    after = _flatten(lstm.to_weights(target))

    assert np.array_equal(after, _flatten(original)), (
        "the weights that came back are not the weights that went in; a warm "
        "start from these would begin somewhere other than the flying model")


def test_a_transposed_gate_block_cannot_even_be_BUILT() -> None:
    """(!) AND THE FIRST GUARD IS NOT MINE, WHICH IS WORTH KNOWING.

    Planting the error a GRU's three stacked gate blocks invite -- a transpose --
    does not reach `load_rnn_weights` at all: `Weights` refuses to be constructed
    from it (`src/sentinel_models/reference.py`). That is a stronger position
    than checking at load time, and it is recorded here rather than duplicated,
    because a second check that can never fire is not evidence of anything.
    """
    from sentinel_models import reference

    hyper = _hyper()
    lstm.configure_determinism(0)
    good = lstm.to_weights(lstm.build_model(3, hyper))
    layers = list(good.layers)
    broken = layers[0]
    layers[0] = lstm.LayerWeights(
        w_ih=np.ascontiguousarray(broken.w_ih.T),   # (gates, in) -> (in, gates)
        w_hh=broken.w_hh, b_ih=broken.b_ih, b_hh=broken.b_hh)

    with pytest.raises(reference.ReferenceError, match="w_ih"):
        lstm.Weights(
            layers=tuple(layers), head_w=good.head_w, head_b=good.head_b,
            n_channels=good.n_channels, window=good.window,
            n_predictions=good.n_predictions, n_exogenous=good.n_exogenous,
            outputs=good.outputs)


def test_the_loader_refuses_weights_for_a_different_channel_width() -> None:
    """The check `load_rnn_weights` DOES own, and it is seen to fire.

    `Weights` validates a block against its own declared shapes; it cannot know
    what model it is about to be loaded into. Warm-starting a five-channel
    shadow from a three-channel flying model is well-formed on both sides and
    wrong in the middle, and this is the only place that can say so.
    """
    hyper = _hyper()
    lstm.configure_determinism(0)
    narrow = lstm.to_weights(lstm.build_model(3, hyper))
    wide = lstm.build_model(5, hyper)

    with pytest.raises(ValueError, match="weight_ih_l0"):
        lstm.load_rnn_weights(wide, narrow)


def test_a_warm_started_fit_is_not_the_cold_fit_at_the_same_seed() -> None:
    """The mechanism reaches the optimiser, not just the constructor.

    Same data, same seed, same everything except where the weights start. If the
    two agree, `init_weights` is being accepted and ignored -- which is the
    failure mode that would make every arm in `docs/MODELS.md` 74 a null result
    wearing a finding's clothes.
    """
    hyper = _hyper()
    rng = np.random.default_rng(0)
    values = rng.normal(size=(600, 3)).astype(np.float32)
    usable = np.ones(len(values), dtype=bool)

    cold, _ = lstm.train(values, usable, hyper)

    lstm.configure_determinism(4242)
    elsewhere = lstm.to_weights(lstm.build_model(3, hyper))
    warm, _ = lstm.train(values, usable, hyper, init_weights=elsewhere)

    assert not np.allclose(_flatten(cold), _flatten(warm)), (
        "a fit started from different weights produced the same answer; "
        "init_weights is not reaching the model")
