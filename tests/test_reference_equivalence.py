"""The contract between the trainer and the flight blueprint. A hard assertion.

Work item 4 trains with PyTorch autograd and scores through
`sentinel_models.reference`, plain NumPy, because that NumPy is what Phase 2's
C++ is transcribed from. The arrangement is only sound while the two agree, so
this file asserts it rather than hoping for it -- and asserts it at float32
noise, not at some tolerance loose enough to hide a transposed matrix.

If anything here fails, every number the LSTM has produced is suspect and the
Phase 2 loader has nothing to verify against. That is why the tolerance is a
constant with a reason beside it and not a knob.

Work item 5's GRU is held to the same assertion, at the same tolerance, through
the same tests -- parametrised over the cell -- plus the ones a GRU can fail on
its own: the slice order of its three gates, and where its third recurrent bias
sits. Those are the two mistakes a C++ author transcribing it is most likely to
make, and each has a test that a wrong transcription cannot pass.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sentinel_models import reference as ref
from sentinel_models.lstm import (Hyper, TelemanomGRU, TelemanomLSTM, TelemanomRNN,
                                  TelemanomTCN, build_model, to_weights)

CELLS = ("lstm", "gru")
PLAYERS = ("lstm", "gru", "tcn")

#: Float32 has ~1.2e-7 of relative resolution and the recurrence compounds it
#: over 250 timesteps and two layers. Anything structural -- a swapped gate, a
#: dropped bias, a transposed weight -- moves the difference by orders of
#: magnitude, so this separates noise from defect with a wide margin either way.
TOLERANCE = 1e-5


def _model(n_channels=4, hidden=(6, 6), n_predictions=3, seed=0, cell="lstm"):
    torch.manual_seed(seed)
    model = build_model(n_channels, Hyper(window=12, hidden=hidden,
                                          n_predictions=n_predictions, cell=cell))
    model.eval()
    return model


def _flown(cell):
    """The configuration each player actually flies at."""
    from sentinel_models.detectors import TCN_HYPER
    return TCN_HYPER if cell == "tcn" else Hyper(cell=cell)


def _both(model, x, *, last_only=False):
    with torch.no_grad():
        expected = model(torch.from_numpy(x), last_only=last_only).numpy()
    actual, _ = ref.forward(to_weights(model), x, last_only=last_only)
    return expected, actual


# -- the assertion ---------------------------------------------------------
@pytest.mark.parametrize("cell", PLAYERS)
@pytest.mark.parametrize("hidden", [(6,), (6, 6), (16, 16)])
def test_the_numpy_forward_pass_matches_torch(hidden, cell):
    model = _model(hidden=hidden, cell=cell)
    x = np.random.default_rng(1).standard_normal((5, 30, 4)).astype(np.float32)
    expected, actual = _both(model, x)
    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < TOLERANCE


@pytest.mark.parametrize("cell", PLAYERS)
def test_it_matches_at_the_configuration_actually_flown(cell):
    """The real thing: 12 channels, a full 250-step window, the flown shape.

    Measured divergence here is ~4e-8 for the LSTM, ~1.2e-7 for the GRU and
    ~5e-7 for the TCN, and none grows with sequence length, so the tolerance
    carries at least an order of magnitude of headroom. That margin is the
    point -- it means a failure is a defect, not a bad day.
    """
    torch.manual_seed(0)
    model = build_model(12, _flown(cell))
    model.eval()
    x = np.random.default_rng(11).standard_normal((4, 250, 12)).astype(np.float32)
    expected, actual = _both(model, x)
    assert np.abs(actual - expected).max() < TOLERANCE


@pytest.mark.parametrize("cell", PLAYERS)
def test_it_matches_over_a_scoring_chunk(cell):
    """The 4,096-step chunk `detectors._forecast` batches; divergence must not grow."""
    torch.manual_seed(0)
    model = build_model(12, _flown(cell))
    model.eval()
    x = np.random.default_rng(12).standard_normal((2, 4096, 12)).astype(np.float32)
    expected, actual = _both(model, x)
    assert np.abs(actual - expected).max() < TOLERANCE


@pytest.mark.parametrize("cell", PLAYERS)
def test_it_matches_on_a_trained_model_over_real_fixture_data(loaded, cell):
    """The path actually used, not just random weights on random input."""
    from sentinel_models.lstm import train

    values = loaded.values[:4000]
    usable = np.isfinite(values).all(axis=1)
    weights, _ = train(values, usable,
                       Hyper(window=20, hidden=(8, 8), n_predictions=3, batch_size=16,
                             max_epochs=2, max_validation_sequences=64, cell=cell))
    assert weights.cell == cell
    x = np.ascontiguousarray(values[None, 100:400])
    model = build_model(values.shape[1],
                        Hyper(window=20, hidden=(8, 8), n_predictions=3, cell=cell))
    _load(model, weights)
    model.eval()
    with torch.no_grad():
        expected = model(torch.from_numpy(x), last_only=False).numpy()
    actual, _ = ref.forward(weights, x)
    assert np.abs(actual - expected).max() < TOLERANCE


@pytest.mark.parametrize("cell", PLAYERS)
def test_last_only_matches_the_final_step_of_the_full_sequence(cell):
    """Training reads one block from the window's end; scoring reads every step."""
    model = _model(cell=cell)
    x = np.random.default_rng(2).standard_normal((3, 25, 4)).astype(np.float32)
    full, _ = ref.forward(to_weights(model), x)
    last, _ = ref.forward(to_weights(model), x, last_only=True)
    assert np.abs(last - full[:, -1]).max() < TOLERANCE


def test_the_two_cells_are_the_same_class_with_one_line_changed():
    """What makes the gate a comparison of cells: nothing else differs."""
    lstm = TelemanomLSTM(4, Hyper(window=12, hidden=(6, 6), n_predictions=3))
    gru = TelemanomGRU(4, Hyper(window=12, hidden=(6, 6), n_predictions=3))
    assert (lstm.cell, gru.cell) == ("lstm", "gru")
    assert isinstance(lstm.rnn, torch.nn.LSTM) and lstm.lstm is lstm.rnn
    assert isinstance(gru.rnn, torch.nn.GRU) and gru.gru is gru.rnn
    assert type(lstm.head) is type(gru.head) and type(lstm.drop) is type(gru.drop)
    assert lstm.head.weight.shape == gru.head.weight.shape
    assert lstm.rnn.dropout == gru.rnn.dropout and lstm.drop.p == gru.drop.p
    assert to_weights(lstm).n_parameters > to_weights(gru).n_parameters


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
        model.rnn.bias_ih_l0[hidden:2 * hidden] = 30.0      # forget -> 1
        model.rnn.bias_ih_l0[0:hidden] = -30.0              # input  -> 0
    weights = to_weights(model)
    x = np.random.default_rng(3).standard_normal((2, 20, 4)).astype(np.float32)

    state = ref.zero_state(weights, 2)
    _, out = ref.forward(weights, x, state)
    assert np.abs(out[0][1]).max() < 1e-6, "cell state should have been held at zero"
    assert ref.GATES == ("input", "forget", "cell", "output")


def test_gru_gate_order_is_reset_update_new():
    """Pin the GRU's slice layout, two ways, each of which a wrong order fails.

    With the *update* gate saturated to 1 the state must be held exactly --
    `h' = (1 - z) n + z h` -- and it is tested from a **nonzero** initial state,
    because from a zero state a wrong slice order would hold zero just as well.
    With *reset* and *update* both saturated to 0 every output is
    `tanh(W_in x + b_in)`, independent of the state, which pins the other two
    slices.
    """
    assert ref.GRU_GATES == ("reset", "update", "new")
    hidden = 6
    x = np.random.default_rng(3).standard_normal((2, 20, 4)).astype(np.float32)
    h0 = (0.5 * np.random.default_rng(4).standard_normal((2, hidden))).astype(np.float32)

    held = _model(hidden=(hidden,), cell="gru")
    with torch.no_grad():
        held.rnn.bias_ih_l0[hidden:2 * hidden] = 30.0       # update -> 1: h' = h
    weights = to_weights(held)
    out, state = ref.gru_layer(x, weights.layers[0], (h0,))
    assert np.abs(state[0] - h0).max() < 1e-6, "the update gate did not hold the state"
    assert np.abs(out - h0[:, None, :]).max() < 1e-6

    fresh = _model(hidden=(hidden,), cell="gru")
    with torch.no_grad():
        fresh.rnn.bias_ih_l0[0:hidden] = -30.0              # reset  -> 0
        fresh.rnn.bias_ih_l0[hidden:2 * hidden] = -30.0     # update -> 0: h' = n
    weights = to_weights(fresh)
    layer = weights.layers[0]
    out, _ = ref.gru_layer(x, layer, (h0,))
    w_in, b_in = layer.w_ih[2 * hidden:3 * hidden], layer.b_ih[2 * hidden:3 * hidden]
    assert np.abs(out - np.tanh(x @ w_in.T + b_in)).max() < 1e-6, (
        "with reset and update off, the output is the new gate alone"
    )


def test_the_gru_recurrent_bias_of_the_new_gate_sits_inside_the_reset_product():
    """The one place an LSTM habit produces a wrong GRU.

    An LSTM's two bias vectors enter the same pre-activation and may be summed
    once. A GRU's `b_hn` is multiplied by the reset gate first, so a loader that
    folds `b_hh` into `b_ih` is wrong on exactly that gate. Two-sided: the
    folded weights must diverge in general, and must **agree** once the reset
    gate is saturated to 1 -- which proves the placement is the reset product
    and nothing else.
    """
    hidden = 6

    def folded(weights):
        layer = weights.layers[0]
        return ref.Weights(
            layers=(ref.LayerWeights(layer.w_ih, layer.w_hh, layer.b_ih + layer.b_hh,
                                     np.zeros_like(layer.b_hh)),),
            head_w=weights.head_w, head_b=weights.head_b, n_channels=weights.n_channels,
            window=weights.window, n_predictions=weights.n_predictions)

    x = np.random.default_rng(5).standard_normal((2, 15, 4)).astype(np.float32)
    weights = to_weights(_model(hidden=(hidden,), cell="gru"))
    true, _ = ref.forward(weights, x)
    wrong, _ = ref.forward(folded(weights), x)
    assert np.abs(true - wrong).max() > TOLERANCE, "folding b_hn went unnoticed"

    model = _model(hidden=(hidden,), cell="gru")
    with torch.no_grad():
        model.rnn.bias_ih_l0[0:hidden] = 30.0               # reset -> 1
    weights = to_weights(model)
    true, _ = ref.forward(weights, x)
    same, _ = ref.forward(folded(weights), x)
    assert np.abs(true - same).max() < TOLERANCE, (
        "with the reset gate open the fold is harmless; the divergence above was "
        "coming from somewhere other than r * b_hn"
    )


@pytest.mark.parametrize("cell", CELLS)
def test_both_biases_are_applied(cell):
    """torch carries b_ih and b_hh separately; a loader that keeps one is wrong."""
    model = _model(hidden=(6,), cell=cell)
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
@pytest.mark.parametrize("cell", CELLS)
def test_state_carries_across_calls(cell):
    """Scoring cuts a 3.7M-step fold into chunks; this is why that is legitimate."""
    model = _model(cell=cell)
    x = np.random.default_rng(5).standard_normal((2, 40, 4)).astype(np.float32)
    weights = to_weights(model)

    whole, _ = ref.forward(weights, x)
    first, state = ref.forward(weights, x[:, :17])
    assert all(len(s) == ref.N_STATE[cell] for s in state)
    second, _ = ref.forward(weights, x[:, 17:], state)
    assert np.abs(np.concatenate([first, second], axis=1) - whole).max() < TOLERANCE


@pytest.mark.parametrize("cell", PLAYERS)
def test_dropout_is_absent_from_the_exported_weights(cell):
    """It is identity at inference, so it has no line in the model file."""
    weights = to_weights(_model(cell=cell))
    assert weights.n_parameters == sum(
        p.numel() for p in _model(cell=cell).parameters()
    )


def test_the_sizes_the_file_format_has_to_carry():
    """docs/MODELS.md section 3 quotes these; they are measured here, not typed."""
    lstm = to_weights(TelemanomRNN(12, Hyper(cell="lstm")))
    gru = to_weights(TelemanomRNN(12, Hyper(cell="gru")))
    tcn = to_weights(build_model(12, _flown("tcn")))
    assert (lstm.n_parameters, lstm.nbytes()) == (91_640, 366_560)
    assert (gru.n_parameters, gru.nbytes()) == (71_160, 284_640)
    assert (tcn.n_parameters, tcn.nbytes()) == (91_670, 366_680)
    assert (lstm.cell, gru.cell, tcn.cell) == ("lstm", "gru", "tcn")
    assert gru.describe().startswith("GRU 12ch -> [80, 80] -> 10x12   71,160 parameters")
    assert tcn.describe().startswith("TCN 12ch -> 6x50 k3 rf253 -> 10x12   91,670 parameters")
    assert to_weights(build_model(6, _flown("tcn"))).n_parameters == 87_410


# -- the TCN: causal, a receptive field of exactly 253, and no state -----------
def test_the_tcn_is_causal():
    """A change to the input at t+1 cannot reach the forecast made at t."""
    weights = to_weights(_model(hidden=(6, 6), cell="tcn"))
    x = np.random.default_rng(8).standard_normal((1, 40, 4)).astype(np.float32)
    y, _ = ref.forward(weights, x)
    later = x.copy()
    later[0, 21:] += 5.0
    y2, _ = ref.forward(weights, later)
    assert np.array_equal(y[:, :21], y2[:, :21]), "the future leaked into the past"
    assert np.abs(y[:, 21:] - y2[:, 21:]).max() > 0, "the perturbation did nothing at all"


def test_the_flown_tcn_sees_exactly_253_steps():
    """Objective.md 8 needs the window stated; 1 + 2(k-1)(2^L - 1) = 253 for k=3, L=6.

    Perturbing the input 253 steps back must not touch the forecast; 252 steps
    back must. Read from the arithmetic, then confirmed on the arrays.
    """
    from sentinel_models.detectors import TCN_HYPER

    weights = to_weights(build_model(12, TCN_HYPER))
    assert weights.receptive_field == 253 == TCN_HYPER.receptive_field
    t = 300
    x = np.random.default_rng(9).standard_normal((1, t + 1, 12)).astype(np.float32)
    base, _ = ref.forward(weights, x)
    for back, touches in ((253, False), (252, True)):
        bumped = x.copy()
        bumped[0, t - back] += 10.0
        y, _ = ref.forward(weights, bumped)
        moved = not np.array_equal(base[0, t], y[0, t])
        assert moved == touches, f"{back} steps back {'moved' if moved else 'did not move'} the forecast"


def test_the_tcn_carries_no_state():
    """The third contract: no state in, none out, and a state handed in is refused."""
    weights = to_weights(_model(hidden=(6, 6), cell="tcn"))
    x = np.random.default_rng(10).standard_normal((2, 30, 4)).astype(np.float32)
    _, state = ref.forward(weights, x)
    assert state == []
    with pytest.raises(ref.ReferenceError, match="no state"):
        ref.forward(weights, x, [(np.zeros((2, 6), np.float32),)] * 2)


def test_a_warmed_chunk_reproduces_the_stream_for_the_tcn():
    """How `detectors._forecast` scores: chunks each preceded by real history.

    With a prefix at least ``receptive_field - 1`` long, a chunk's forecasts
    equal the uncut stream's exactly -- which is the ring buffer after it fills.
    """
    weights = to_weights(_model(hidden=(6, 6), cell="tcn"))      # rf = 1 + 2*2*3 = 13
    x = np.random.default_rng(12).standard_normal((1, 80, 4)).astype(np.float32)
    whole, _ = ref.forward(weights, x)
    cut = 40
    prefix = weights.receptive_field - 1
    chunk, _ = ref.forward(weights, x[:, cut - prefix:])
    assert np.abs(chunk[:, prefix:] - whole[:, cut:]).max() < TOLERANCE


# -- refusals rather than silent nonsense ----------------------------------
@pytest.mark.parametrize("cell", CELLS)
def test_a_channel_count_mismatch_is_refused(cell):
    weights = to_weights(_model(cell=cell))
    with pytest.raises(ref.ReferenceError, match="channels"):
        ref.forward(weights, np.zeros((1, 5, 7), dtype=np.float32))


def test_layers_of_different_cells_are_refused():
    lstm, gru = to_weights(_model()), to_weights(_model(cell="gru"))
    with pytest.raises(ref.ReferenceError, match="disagree"):
        ref.Weights(layers=(lstm.layers[0], gru.layers[1]), head_w=gru.head_w,
                    head_b=gru.head_b, n_channels=4, window=12, n_predictions=3)


def test_a_gate_count_that_is_neither_cell_is_refused():
    with pytest.raises(ref.ReferenceError, match="neither"):
        ref.LayerWeights(np.zeros((10, 4), np.float32), np.zeros((10, 5), np.float32),
                         np.zeros(10, np.float32), np.zeros(10, np.float32)).validate()


def test_an_lstm_state_handed_to_gru_weights_is_refused():
    """A dead cell vector is not carried silently; the flight state is exactly `(h,)`."""
    weights = to_weights(_model(cell="gru"))
    x = np.zeros((2, 5, 4), dtype=np.float32)
    lstm_shaped = [(np.zeros((2, 6), np.float32), np.zeros((2, 6), np.float32))] * 2
    with pytest.raises(ref.ReferenceError, match="state vector"):
        ref.forward(weights, x, lstm_shaped)


def test_a_two_dimensional_input_is_refused():
    weights = to_weights(_model())
    with pytest.raises(ref.ReferenceError, match="batch, steps, channels"):
        ref.forward(weights, np.zeros((5, 4), dtype=np.float32))


def _load(model, weights) -> None:
    assert model.cell == weights.cell
    if isinstance(weights, ref.ConvWeights):
        with torch.no_grad():
            for block, w in zip(model.blocks, weights.blocks):
                block.conv1.weight.copy_(torch.from_numpy(w.w1)); block.conv1.bias.copy_(torch.from_numpy(w.b1))
                block.conv2.weight.copy_(torch.from_numpy(w.w2)); block.conv2.bias.copy_(torch.from_numpy(w.b2))
                if w.res_w is not None:
                    block.res.weight.copy_(torch.from_numpy(w.res_w)); block.res.bias.copy_(torch.from_numpy(w.res_b))
            model.head.weight.copy_(torch.from_numpy(weights.head_w))
            model.head.bias.copy_(torch.from_numpy(weights.head_b))
        return
    with torch.no_grad():
        for i, layer in enumerate(weights.layers):
            getattr(model.rnn, f"weight_ih_l{i}").copy_(torch.from_numpy(layer.w_ih))
            getattr(model.rnn, f"weight_hh_l{i}").copy_(torch.from_numpy(layer.w_hh))
            getattr(model.rnn, f"bias_ih_l{i}").copy_(torch.from_numpy(layer.b_ih))
            getattr(model.rnn, f"bias_hh_l{i}").copy_(torch.from_numpy(layer.b_hh))
        model.head.weight.copy_(torch.from_numpy(weights.head_w))
        model.head.bias.copy_(torch.from_numpy(weights.head_b))
