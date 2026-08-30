"""The plain-NumPy forward pass, and the weights it reads. No framework.

This module is the Phase 2 blueprint. Objective.md 4.3 step 2 is categorical --
*no Python, no ML framework, no interpreter goes to space* -- so the flight
component is C++ reading a file of numbers, and something has to define what
those numbers mean and in what order they are consumed. That definition lives
here, in the smallest form that can still be executed and therefore tested.

Three consequences, each deliberate:

**Training and scoring are separated.** :mod:`sentinel_models.lstm` trains with
PyTorch autograd, because a hand-written LSTM backward pass that is subtly wrong
produces a *plausible* bad number, which is the one failure mode this project's
documents rail against. Scoring then runs through the code below. The two are
held together by `tests/test_reference_equivalence.py`, which is a hard
assertion and not a warning: if they ever disagree beyond float32 noise, the
suite fails.

**The cell is a named function.** :func:`lstm_cell` is exactly the arithmetic one
rate-group tick performs -- four gates, two state vectors, no allocation that
depends on the data. :func:`gru_cell` is the same for work item 5's cell --
three gates, one state vector. The C++ is a transcription of whichever the
architecture gate selects. :func:`lstm_layer` and :func:`gru_layer` exist only
to batch that cell efficiently on a laptop, and hoist the input projection out
of the loop; the flight code has no batch and computes it per tick, which is the
same arithmetic in a different order of evaluation.

**State is explicit, in and out.** Scoring 14.7 million timesteps one at a time
in Python is not viable, so a window is cut into chunks that run as a batch, each
warmed by a short zero-initialised prefix. That is not an approximation of
telemanom: telemanom's own inference reads a 250-step window from a zero state,
so a 250-step warmed chunk sees precisely the history the published method sees.
Carrying state across calls is what makes the chunking work, and it is also how
the flight component runs -- one state, forever, updated per tick.

The dataclass is plain arrays with an explicit gate order, not a Python object
graph, because Objective.md 14.2 must freeze a *file format* and a format cannot
be defined by a class definition. What we learn writing it is recorded in
docs/MODELS.md and feeds that decision. **A file does not say which cell it
holds by declaration alone: the gate count in its arrays says it**, 4H rows for
an LSTM and 3H for a GRU, and a declared cell that disagrees with the arrays is
refused.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Gate order inside the stacked weight matrices, as PyTorch stacks them and as
#: the C++ loader will have to expect. Named here once so that a reader of the
#: file format never has to infer it from a slice index.
GATES = ("input", "forget", "cell", "output")

#: The GRU's, likewise PyTorch's order: `w_ih` is `(W_ir | W_iz | W_in)` stacked
#: along axis 0. The third gate is the one a loader gets wrong -- see
#: :func:`gru_cell` for where its recurrent bias sits.
GRU_GATES = ("reset", "update", "new")

#: How many gate blocks each cell stacks, and how many state vectors it carries.
#: The gate count is what the arrays themselves say, so it is the identity a
#: reader trusts; the cell name is derived from it, never the other way round.
N_GATES = {"lstm": len(GATES), "gru": len(GRU_GATES)}
#: The TCN has no gates and no state; see :class:`ConvWeights`.
CELL_BY_GATES = {n: cell for cell, n in N_GATES.items()}
N_STATE = {"lstm": 2, "gru": 1}

DTYPE = np.float32


class ReferenceError(RuntimeError):
    """The reference implementation was handed something it cannot execute.

    Local rather than imported: `tests/test_layering.py` allows this package to
    import `sentinel_eval.detector` and nothing else from the harness, and that
    one-way dependency is worth more than sharing an exception base class.
    """


@dataclass(frozen=True)
class LayerWeights:
    """One recurrent layer. Gates stacked along axis 0 in :data:`GATES` order
    for an LSTM (``4H`` rows) or :data:`GRU_GATES` order for a GRU (``3H``)."""

    w_ih: np.ndarray          # (G*H, n_in)
    w_hh: np.ndarray          # (G*H, H)
    b_ih: np.ndarray          # (G*H,)
    b_hh: np.ndarray          # (G*H,)

    @property
    def hidden(self) -> int:
        return self.w_hh.shape[1]

    @property
    def n_in(self) -> int:
        return self.w_ih.shape[1]

    @property
    def n_gates(self) -> int:
        """Read from the arrays, so it cannot disagree with them."""
        return self.w_hh.shape[0] // max(self.hidden, 1)

    @property
    def cell(self) -> str:
        try:
            return CELL_BY_GATES[self.n_gates]
        except KeyError:
            raise ReferenceError(
                f"w_hh is {self.w_hh.shape}: {self.n_gates} gate blocks of {self.hidden}, "
                f"which is neither an LSTM (4) nor a GRU (3)"
            ) from None

    @property
    def n_state(self) -> int:
        return N_STATE[self.cell]

    def validate(self) -> None:
        h, g = self.hidden, self.n_gates
        if self.w_hh.shape != (g * h, h):
            raise ReferenceError(f"w_hh is {self.w_hh.shape}, expected {(g * h, h)}")
        self.cell                                   # refuses a gate count of neither 3 nor 4
        if self.w_ih.shape != (g * h, self.n_in):
            raise ReferenceError(f"w_ih is {self.w_ih.shape}, expected {(g * h, self.n_in)}")
        for name, array, shape in (("b_ih", self.b_ih, (g * h,)),
                                   ("b_hh", self.b_hh, (g * h,))):
            if array.shape != shape:
                raise ReferenceError(f"{name} is {array.shape}, expected {shape}")


@dataclass(frozen=True)
class Weights:
    """Everything the forward pass needs, as plain arrays. The `model.bin` payload.

    ``n_predictions`` is telemanom's ``l_p``: the head emits that many future
    timesteps for every channel, flattened as ``(step, channel)`` in C order.
    """

    layers: tuple[LayerWeights, ...]
    head_w: np.ndarray        # (n_predictions * n_channels, H)
    head_b: np.ndarray        # (n_predictions * n_channels,)
    n_channels: int
    window: int
    n_predictions: int
    #: Exogenous input columns -- telecommand features. The model reads
    #: ``n_channels + n_exogenous`` and forecasts ``n_channels``: it is told what
    #: was commanded, and asked only what the telemetry will do about it. A
    #: `model.bin` has to carry this separately from the channel count for
    #: exactly that reason (docs/DECISIONS.md D6).
    n_exogenous: int = 0

    def __post_init__(self) -> None:
        if not self.layers:
            raise ReferenceError("a model with no layers cannot forecast")
        for layer in self.layers:
            layer.validate()
        cells = {layer.cell for layer in self.layers}
        if len(cells) != 1:
            raise ReferenceError(
                f"layers disagree on the cell: {[layer.cell for layer in self.layers]}"
            )
        if self.layers[0].n_in != self.n_inputs:
            raise ReferenceError(
                f"layer 0 takes {self.layers[0].n_in} inputs but the model declares "
                f"{self.n_channels} channels + {self.n_exogenous} exogenous"
            )
        expected = self.n_predictions * self.n_channels
        if self.head_w.shape != (expected, self.layers[-1].hidden):
            raise ReferenceError(
                f"head is {self.head_w.shape}, expected "
                f"{(expected, self.layers[-1].hidden)}"
            )

    @property
    def cell(self) -> str:
        """Derived from the arrays' gate count. Not stored, so it cannot lie."""
        return self.layers[0].cell

    @property
    def n_inputs(self) -> int:
        return self.n_channels + self.n_exogenous

    @property
    def hidden(self) -> tuple[int, ...]:
        return tuple(layer.hidden for layer in self.layers)

    @property
    def n_parameters(self) -> int:
        total = sum(a.size for layer in self.layers
                    for a in (layer.w_ih, layer.w_hh, layer.b_ih, layer.b_hh))
        return int(total + self.head_w.size + self.head_b.size)

    def nbytes(self) -> int:
        """Float32 payload size. What the file format has to carry, before any
        header, quantization or CRC -- the input to Objective.md 14.2."""
        return self.n_parameters * 4

    def describe(self) -> str:
        commanded = f"+{self.n_exogenous}cmd" if self.n_exogenous else ""
        return (f"{self.cell.upper()} {self.n_channels}ch{commanded} -> {list(self.hidden)} -> "
                f"{self.n_predictions}x{self.n_channels}   "
                f"{self.n_parameters:,} parameters, {self.nbytes() / 1024:.1f} KiB float32")

    def arrays(self) -> list[tuple[str, np.ndarray]]:
        """Every array, named as the weight file stores it, in a fixed order."""
        out = []
        for i, layer in enumerate(self.layers):
            for name in ("w_ih", "w_hh", "b_ih", "b_hh"):
                out.append((f"l{i}_{name}", getattr(layer, name)))
        out += [("head_w", self.head_w), ("head_b", self.head_b)]
        return out


# -- the TCN ------------------------------------------------------------------
@dataclass(frozen=True)
class ConvBlockWeights:
    """One residual block: two causal dilated convolutions and, when the width
    changes, a 1x1 shortcut. Kernels are ``(out, in, k)`` as PyTorch stores them,
    tap ``j`` reading the input ``(k - 1 - j) * dilation`` steps back."""

    w1: np.ndarray            # (H, n_in, k)
    b1: np.ndarray            # (H,)
    w2: np.ndarray            # (H, H, k)
    b2: np.ndarray            # (H,)
    res_w: np.ndarray | None  # (H, n_in, 1), only when n_in != H
    res_b: np.ndarray | None  # (H,)
    dilation: int

    @property
    def hidden(self) -> int:
        return self.w1.shape[0]

    @property
    def n_in(self) -> int:
        return self.w1.shape[1]

    @property
    def kernel(self) -> int:
        return self.w1.shape[2]

    @property
    def reach(self) -> int:
        """Steps of history this block adds to the receptive field."""
        return 2 * (self.kernel - 1) * self.dilation

    def validate(self) -> None:
        h, k = self.hidden, self.kernel
        if self.dilation < 1:
            raise ReferenceError(f"dilation must be positive, got {self.dilation}")
        for name, array, shape in (("b1", self.b1, (h,)), ("w2", self.w2, (h, h, k)),
                                   ("b2", self.b2, (h,))):
            if array.shape != shape:
                raise ReferenceError(f"{name} is {array.shape}, expected {shape}")
        if (self.res_w is None) != (self.n_in == h):
            raise ReferenceError(
                f"a block from {self.n_in} to {h} channels "
                f"{'needs' if self.n_in != h else 'must not carry'} a 1x1 shortcut"
            )
        if self.res_w is not None:
            if self.res_w.shape != (h, self.n_in, 1) or self.res_b is None or self.res_b.shape != (h,):
                raise ReferenceError(f"shortcut is {self.res_w.shape}, expected {(h, self.n_in, 1)}")


@dataclass(frozen=True)
class ConvWeights:
    """A TCN's payload. Plain arrays, and **no state**.

    The forecast at ``t`` is a function of the last ``receptive_field`` inputs
    and nothing else. That is the flight-code shape Objective.md section 8
    lists for this candidate -- identical compute every cycle, nothing to
    corrupt, a clean restart -- and it is a different contract from the two
    cells: :func:`forward` refuses a state for these weights and returns an
    empty one, and the harness's chunked scoring, which warms every chunk with
    ``window`` real steps and carries nothing across chunks, is exactly how a
    ring buffer of ``receptive_field - 1`` inputs behaves after a cold start.
    """

    blocks: tuple[ConvBlockWeights, ...]
    head_w: np.ndarray        # (n_predictions * n_channels, H)
    head_b: np.ndarray        # (n_predictions * n_channels,)
    n_channels: int
    window: int
    n_predictions: int
    n_exogenous: int = 0

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ReferenceError("a model with no blocks cannot forecast")
        for block in self.blocks:
            block.validate()
        if self.blocks[0].n_in != self.n_inputs:
            raise ReferenceError(
                f"block 0 takes {self.blocks[0].n_in} inputs but the model declares "
                f"{self.n_channels} channels + {self.n_exogenous} exogenous"
            )
        for a, b in zip(self.blocks, self.blocks[1:]):
            if b.n_in != a.hidden:
                raise ReferenceError(f"block widths do not chain: {a.hidden} -> {b.n_in}")
        expected = self.n_predictions * self.n_channels
        if self.head_w.shape != (expected, self.blocks[-1].hidden):
            raise ReferenceError(
                f"head is {self.head_w.shape}, expected {(expected, self.blocks[-1].hidden)}"
            )

    cell = "tcn"

    @property
    def n_inputs(self) -> int:
        return self.n_channels + self.n_exogenous

    @property
    def hidden(self) -> tuple[int, ...]:
        return tuple(block.hidden for block in self.blocks)

    @property
    def kernel(self) -> int:
        return self.blocks[0].kernel

    @property
    def receptive_field(self) -> int:
        return 1 + sum(block.reach for block in self.blocks)

    @property
    def layers(self) -> tuple[ConvBlockWeights, ...]:
        """The blocks, under the name the two cells use, for code that only counts."""
        return self.blocks

    def arrays(self) -> list[tuple[str, np.ndarray]]:
        out = []
        for i, block in enumerate(self.blocks):
            out += [(f"b{i}_w1", block.w1), (f"b{i}_b1", block.b1),
                    (f"b{i}_w2", block.w2), (f"b{i}_b2", block.b2)]
            if block.res_w is not None:
                out += [(f"b{i}_res_w", block.res_w), (f"b{i}_res_b", block.res_b)]
        out += [("head_w", self.head_w), ("head_b", self.head_b)]
        return out

    @property
    def n_parameters(self) -> int:
        return int(sum(a.size for _, a in self.arrays()))

    def nbytes(self) -> int:
        return self.n_parameters * 4

    def describe(self) -> str:
        commanded = f"+{self.n_exogenous}cmd" if self.n_exogenous else ""
        return (f"TCN {self.n_channels}ch{commanded} -> {len(self.blocks)}x{self.blocks[0].hidden} "
                f"k{self.kernel} rf{self.receptive_field} -> {self.n_predictions}x{self.n_channels}   "
                f"{self.n_parameters:,} parameters, {self.nbytes() / 1024:.1f} KiB float32")


def causal_conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray, dilation: int) -> np.ndarray:
    """``(batch, steps, n_in)`` -> ``(batch, steps, out)``, reading only the past.

    Tap ``j`` of the kernel multiplies the input ``(k - 1 - j) * dilation`` steps
    back; the ``(k - 1) * dilation`` steps before the sequence are zero -- the
    ring buffer at boot. Per tick this is ``k`` matrix-vector products and an
    add, on inputs the flight component already holds.
    """
    batch, steps, _ = x.shape
    out_channels, _, k = w.shape
    pad = (k - 1) * dilation
    padded = np.concatenate([np.zeros((batch, pad, x.shape[2]), dtype=DTYPE), x], axis=1)
    out = np.broadcast_to(b, (batch, steps, out_channels)).astype(DTYPE)
    for j in range(k):
        start = j * dilation
        out = out + padded[:, start:start + steps] @ w[:, :, j].T
    return out


def tcn_block(x: np.ndarray, block: ConvBlockWeights) -> np.ndarray:
    """One residual block: conv, ReLU, conv, ReLU, add the shortcut, ReLU.

    Dropout has no line here: identity at inference (see :func:`forward`).
    """
    y = np.maximum(causal_conv1d(x, block.w1, block.b1, block.dilation), 0)
    y = np.maximum(causal_conv1d(y, block.w2, block.b2, block.dilation), 0)
    shortcut = x if block.res_w is None else x @ block.res_w[:, :, 0].T + block.res_b
    return np.maximum(y + shortcut, 0)


#: State for one layer: ``(h, c)`` for an LSTM, ``(h,)`` for a GRU, each array
#: ``(batch, hidden)``. One tuple per layer, whatever its length -- a GRU does
#: not carry a dead cell vector to look like an LSTM, because the flight
#: component's state is exactly what is listed here and nothing more.
State = tuple[np.ndarray, ...]


def zero_state(weights: Weights, batch: int) -> list[State]:
    return [tuple(np.zeros((batch, layer.hidden), dtype=DTYPE)
                  for _ in range(layer.n_state))
            for layer in weights.layers]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic, in the branch form that does not overflow for large |x|."""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponent = np.exp(x[~positive])
    out[~positive] = exponent / (1.0 + exponent)
    return out


# -- the LSTM -----------------------------------------------------------------
def lstm_cell(projected: np.ndarray, h: np.ndarray, c: np.ndarray,
              w_hh: np.ndarray, bias: np.ndarray) -> State:
    """One tick. ``projected`` is ``x @ w_ih.T`` for this timestep.

    This is the whole of the flight component's per-cycle recurrent arithmetic:
    one matrix-vector product, one add, four elementwise nonlinearities, two
    state updates. Fixed shapes, fixed cost, no branch on the data --
    Objective.md 11 rule 5.
    """
    gates = projected + h @ w_hh.T + bias
    hidden = c.shape[1]
    i = _sigmoid(gates[:, 0:hidden])
    f = _sigmoid(gates[:, hidden:2 * hidden])
    g = np.tanh(gates[:, 2 * hidden:3 * hidden])
    o = _sigmoid(gates[:, 3 * hidden:4 * hidden])
    c_next = f * c + i * g
    return o * np.tanh(c_next), c_next


def lstm_layer(x: np.ndarray, layer: LayerWeights, state: State) -> tuple[np.ndarray, State]:
    """Run one layer over ``(batch, steps, n_in)``, returning every hidden state.

    The input projection is hoisted out of the loop as a single GEMM. That is a
    laptop optimisation and changes no arithmetic: the flight code computes the
    same product one tick at a time, because it only ever has one tick.

    Both bias vectors enter the same pre-activation, so for the LSTM they are
    summed once here. **That is not true of the GRU** -- see :func:`gru_layer`.
    """
    batch, steps, _ = x.shape
    projected = x.reshape(-1, layer.n_in) @ layer.w_ih.T
    projected = projected.reshape(batch, steps, layer.n_gates * layer.hidden)
    bias = layer.b_ih + layer.b_hh

    h, c = state
    out = np.empty((batch, steps, layer.hidden), dtype=DTYPE)
    for t in range(steps):
        h, c = lstm_cell(projected[:, t], h, c, layer.w_hh, bias)
        out[:, t] = h
    return out, (h, c)


# -- the GRU ------------------------------------------------------------------
def gru_cell(projected: np.ndarray, h: np.ndarray,
             w_hh: np.ndarray, b_hh: np.ndarray) -> State:
    """One tick. ``projected`` is ``x @ w_ih.T + b_ih`` for this timestep.

    PyTorch's GRU, which is what the weights were fitted as:

        r  = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
        z  = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
        n  = tanh   (W_in x + b_in + r * (W_hn h + b_hn))
        h' = (1 - z) * n + z * h

    **The recurrent bias of the third gate, ``b_hn``, sits inside the reset
    product.** It cannot be pre-summed with ``b_in`` the way an LSTM's two bias
    vectors can, and a loader that folds the biases together will be wrong on
    exactly this gate and nowhere else -- which is why the recurrent product
    below keeps its bias and the input projection keeps its own. The file
    format must store both bias vectors unsummed for a GRU (docs/MODELS.md
    section 3).

    One matrix-vector product, one add, three elementwise nonlinearities, one
    state update. Fixed shapes, fixed cost, no branch on the data.
    """
    hidden = h.shape[1]
    recurrent = h @ w_hh.T + b_hh
    r = _sigmoid(projected[:, 0:hidden] + recurrent[:, 0:hidden])
    z = _sigmoid(projected[:, hidden:2 * hidden] + recurrent[:, hidden:2 * hidden])
    n = np.tanh(projected[:, 2 * hidden:3 * hidden] + r * recurrent[:, 2 * hidden:3 * hidden])
    return ((h - n) * z + n,)            # ATen's form of (1 - z) * n + z * h


def gru_layer(x: np.ndarray, layer: LayerWeights, state: State) -> tuple[np.ndarray, State]:
    """Run one GRU layer over ``(batch, steps, n_in)``, returning every hidden state.

    The input projection **including ``b_ih``** is hoisted as one GEMM, as torch
    does; ``b_hh`` stays with the recurrent product inside the loop, because the
    third of it is multiplied by the reset gate (see :func:`gru_cell`).
    """
    batch, steps, _ = x.shape
    projected = x.reshape(-1, layer.n_in) @ layer.w_ih.T + layer.b_ih
    projected = projected.reshape(batch, steps, layer.n_gates * layer.hidden)

    (h,) = state
    out = np.empty((batch, steps, layer.hidden), dtype=DTYPE)
    for t in range(steps):
        (h,) = gru_cell(projected[:, t], h, layer.w_hh, layer.b_hh)
        out[:, t] = h
    return out, (h,)


_LAYER = {"lstm": lstm_layer, "gru": gru_layer}


def forward(weights: Weights | ConvWeights, x: np.ndarray, state: list[State] | None = None,
            *, last_only: bool = False) -> tuple[np.ndarray, list[State]]:
    """Forecast from ``(batch, steps, n_channels)``.

    Returns ``(batch, steps, n_predictions, n_channels)`` -- the forecast made
    *at* each timestep for the ``n_predictions`` that follow it -- together with
    the state after the final step, so the next chunk can continue from here.
    ``last_only`` collapses the time axis to the final step, which is the shape
    telemanom trains on: one window in, one block of future values out.

    **Three contracts for state, one per player.** An LSTM's is ``(h, c)`` per
    layer, a GRU's ``(h,)``, and a TCN's is **nothing**: `ConvWeights` refuse a
    state and return an empty list, because their history is the input itself.

    No dropout. Dropout exists only during training; a flight component that
    dropped a random 30% of its hidden units every cycle would violate
    Objective.md 11 rule 5, and PyTorch's eval mode does the same thing here by
    simply not being implemented.
    """
    x = np.ascontiguousarray(x, dtype=DTYPE)
    if x.ndim != 3:
        raise ReferenceError(f"input must be (batch, steps, channels), got {x.shape}")
    if x.shape[2] != weights.n_inputs:
        raise ReferenceError(
            f"input has {x.shape[2]} channels, the model was fitted on "
            f"{weights.n_inputs} ({weights.n_channels} telemetry + "
            f"{weights.n_exogenous} exogenous)"
        )

    if isinstance(weights, ConvWeights):
        if state is not None:
            raise ReferenceError(
                "a TCN carries no state: its history is the input window, and a state "
                "handed to it would be a claim the arithmetic cannot honour"
            )
        activations = x
        for block in weights.blocks:
            activations = tcn_block(activations, block)
        if last_only:
            activations = activations[:, -1:]
        batch, steps, hidden = activations.shape
        flat = activations.reshape(-1, hidden) @ weights.head_w.T + weights.head_b
        y = flat.reshape(batch, steps, weights.n_predictions, weights.n_channels)
        return (y[:, 0] if last_only else y), []

    state = zero_state(weights, x.shape[0]) if state is None else list(state)
    if len(state) != len(weights.layers):
        raise ReferenceError(
            f"state carries {len(state)} layers, the model has {len(weights.layers)}"
        )
    for i, (layer, layer_state) in enumerate(zip(weights.layers, state)):
        if len(layer_state) != layer.n_state:
            raise ReferenceError(
                f"layer {i} is a {layer.cell.upper()} and carries {layer.n_state} state "
                f"vector(s); the state handed in has {len(layer_state)}"
            )

    activations = x
    out_state: list[State] = []
    for layer, layer_state in zip(weights.layers, state):
        activations, next_state = _LAYER[layer.cell](activations, layer, layer_state)
        out_state.append(next_state)

    if last_only:
        activations = activations[:, -1:]
    batch, steps, hidden = activations.shape
    flat = activations.reshape(-1, hidden) @ weights.head_w.T + weights.head_b
    y = flat.reshape(batch, steps, weights.n_predictions, weights.n_channels)
    return (y[:, 0] if last_only else y), out_state
