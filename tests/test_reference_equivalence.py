"""The contract between the trainer and the flight blueprint. A hard assertion.

Work item 4 trains with PyTorch autograd and scores through
`sentinel_models.reference`, plain NumPy, because that NumPy is what Phase 2's
C++ is transcribed from. The arrangement is only sound while the two agree, so
this file asserts it rather than hoping for it -- and asserts it at float32
noise, not at some tolerance loose enough to hide a transposed matrix.

If anything here fails, every number the LSTM has produced is suspect and the
Phase 2 loader has nothing to verify against. That is why the tolerance is a
constant with a reason beside it and not a knob.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sentinel_models import reference as ref
from sentinel_models.lstm import Hyper, TelemanomLSTM, to_weights

#: Float32 has ~1.2e-7 of relative resolution and the recurrence compounds it
#: over 250 timesteps and two layers. Anything structural -- a swapped gate, a
#: dropped bias, a transposed weight -- moves the difference by orders of
#: magnitude, so this separates noise from defect with a wide margin either way.
TOLERANCE = 1e-5


def _model(n_channels=4, hidden=(6, 6), n_predictions=3, seed=0) -> TelemanomLSTM:
    torch.manual_seed(seed)
    model = TelemanomLSTM(n_channels, Hyper(window=12, hidden=hidden,
                                            n_predictions=n_predictions))
    model.eval()
    return model


def _both(model, x, *, last_only=False):
    with torch.no_grad():
        expected = model(torch.from_numpy(x), last_only=last_only).numpy()
    actual, _ = ref.forward(to_weights(model), x, last_only=last_only)
    return expected, actual


# -- the assertion ---------------------------------------------------------
@pytest.mark.parametrize("hidden", [(6,), (6, 6), (16, 16)])
def test_the_numpy_forward_pass_matches_torch(hidden):
    model = _model(hidden=hidden)
    x = np.random.default_rng(1).standard_normal((5, 30, 4)).astype(np.float32)
    expected, actual = _both(model, x)
    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < TOLERANCE


def test_it_matches_at_the_configuration_actually_flown():
    """The real thing: telemanom's 2x80 over 12 channels, a full 250-step window.

    Measured divergence here is ~4e-8 and does not grow with sequence length,
    so the tolerance carries roughly two orders of magnitude of headroom. That
    margin is the point -- it means a failure is a defect, not a bad day.
    """
    torch.manual_seed(0)
    model = TelemanomLSTM(12, Hyper())
    model.eval()
    x = np.random.default_rng(11).standard_normal((4, 250, 12)).astype(np.float32)
    expected, actual = _both(model, x)
    assert np.abs(actual - expected).max() < TOLERANCE


def test_it_matches_on_a_trained_model_over_real_fixture_data(loaded):
    """The path actually used, not just random weights on random input."""
    from sentinel_models.lstm import train

    values = loaded.values[:4000]
    usable = np.isfinite(values).all(axis=1)
    weights, _ = train(values, usable,
                       Hyper(window=20, hidden=(8, 8), n_predictions=3, batch_size=16,
                             max_epochs=2, max_validation_sequences=64))
    x = np.ascontiguousarray(values[None, 100:400])
    model = TelemanomLSTM(values.shape[1], Hyper(window=20, hidden=(8, 8), n_predictions=3))
    _load(model, weights)
    model.eval()
    with torch.no_grad():
        expected = model(torch.from_numpy(x), last_only=False).numpy()
    actual, _ = ref.forward(weights, x)
    assert np.abs(actual - expected).max() < TOLERANCE


def test_last_only_matches_the_final_step_of_the_full_sequence():
    """Training reads one block from the window's end; scoring reads every step."""
    model = _model()
    x = np.random.default_rng(2).standard_normal((3, 25, 4)).astype(np.float32)
    full, _ = ref.forward(to_weights(model), x)
    last, _ = ref.forward(to_weights(model), x, last_only=True)
    assert np.abs(last - full[:, -1]).max() < TOLERANCE


# -- what a wrong file format would look like ------------------------------
def test_gate_order_is_input_forget_cell_output():
    """Pin the slice layout the C++ loader will be written against.

    Saturating the forget gate must make the cell state persist and the input
    gate must stop contributing; if the slices were in any other order this
    would not hold, whatever the aggregate error looked like.
    """
    model = _model(hidden=(6,))
    hidden = 6
    with torch.no_grad():
        model.lstm.bias_ih_l0[hidden:2 * hidden] = 30.0     # forget -> 1
        model.lstm.bias_ih_l0[0:hidden] = -30.0             # input  -> 0
    weights = to_weights(model)
    x = np.random.default_rng(3).standard_normal((2, 20, 4)).astype(np.float32)

    state = ref.zero_state(weights, 2)
    _, out = ref.forward(weights, x, state)
    assert np.abs(out[0][1]).max() < 1e-6, "cell state should have been held at zero"
    assert ref.GATES == ("input", "forget", "cell", "output")


def test_both_biases_are_applied():
    """torch carries b_ih and b_hh separately; a loader that keeps one is wrong."""
    model = _model(hidden=(6,))
    x = np.random.default_rng(4).standard_normal((2, 15, 4)).astype(np.float32)
    weights = to_weights(model)
    crippled = ref.Weights(
        layers=(ref.LayerWeights(weights.layers[0].w_ih, weights.layers[0].w_hh,
                                 weights.layers[0].b_ih,
                                 np.zeros_like(weights.layers[0].b_hh)),),
        head_w=weights.head_w, head_b=weights.head_b, n_channels=weights.n_channels,
        window=weights.window, n_predictions=weights.n_predictions,
    )
    full, _ = ref.forward(weights, x)
    partial, _ = ref.forward(crippled, x)
    assert np.abs(full - partial).max() > TOLERANCE


# -- what chunked scoring depends on ---------------------------------------
def test_state_carries_across_calls():
    """Scoring cuts a 3.7M-step fold into chunks; this is why that is legitimate."""
    model = _model()
    x = np.random.default_rng(5).standard_normal((2, 40, 4)).astype(np.float32)
    weights = to_weights(model)

    whole, _ = ref.forward(weights, x)
    first, state = ref.forward(weights, x[:, :17])
    second, _ = ref.forward(weights, x[:, 17:], state)
    assert np.abs(np.concatenate([first, second], axis=1) - whole).max() < TOLERANCE


def test_dropout_is_absent_from_the_exported_weights():
    """It is identity at inference, so it has no line in the model file."""
    weights = to_weights(_model())
    assert weights.n_parameters == sum(
        p.numel() for p in _model().parameters()
    )


# -- refusals rather than silent nonsense ----------------------------------
def test_a_channel_count_mismatch_is_refused():
    weights = to_weights(_model())
    with pytest.raises(ref.ReferenceError, match="channels"):
        ref.forward(weights, np.zeros((1, 5, 7), dtype=np.float32))


def test_a_two_dimensional_input_is_refused():
    weights = to_weights(_model())
    with pytest.raises(ref.ReferenceError, match="batch, steps, channels"):
        ref.forward(weights, np.zeros((5, 4), dtype=np.float32))


def _load(model: TelemanomLSTM, weights: ref.Weights) -> None:
    with torch.no_grad():
        for i, layer in enumerate(weights.layers):
            getattr(model.lstm, f"weight_ih_l{i}").copy_(torch.from_numpy(layer.w_ih))
            getattr(model.lstm, f"weight_hh_l{i}").copy_(torch.from_numpy(layer.w_hh))
            getattr(model.lstm, f"bias_ih_l{i}").copy_(torch.from_numpy(layer.b_ih))
            getattr(model.lstm, f"bias_hh_l{i}").copy_(torch.from_numpy(layer.b_hh))
        model.head.weight.copy_(torch.from_numpy(weights.head_w))
        model.head.bias.copy_(torch.from_numpy(weights.head_b))
