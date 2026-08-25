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
depends on the data. The C++ is a transcription of it. :func:`lstm_layer` exists
only to batch that cell efficiently on a laptop, and hoists the input projection
out of the loop; the flight code has no batch and computes it per tick, which is
the same arithmetic in a different order of evaluation.

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
docs/MODELS.md and feeds that decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Gate order inside the stacked weight matrices, as PyTorch stacks them and as
#: the C++ loader will have to expect. Named here once so that a reader of the
#: file format never has to infer it from a slice index.
GATES = ("input", "forget", "cell", "output")

DTYPE = np.float32


class ReferenceError(RuntimeError):
    """The reference implementation was handed something it cannot execute.

    Local rather than imported: `tests/test_layering.py` allows this package to
    import `sentinel_eval.detector` and nothing else from the harness, and that
    one-way dependency is worth more than sharing an exception base class.
    """


@dataclass(frozen=True)
class LayerWeights:
    """One LSTM layer. Gates stacked in :data:`GATES` order along axis 0."""

    w_ih: np.ndarray          # (4H, n_in)
    w_hh: np.ndarray          # (4H, H)
    b_ih: np.ndarray          # (4H,)
    b_hh: np.ndarray          # (4H,)

    @property
    def hidden(self) -> int:
        return self.w_hh.shape[1]

    @property
    def n_in(self) -> int:
        return self.w_ih.shape[1]

    def validate(self) -> None:
        h = self.hidden
        if self.w_ih.shape != (4 * h, self.n_in):
            raise ReferenceError(f"w_ih is {self.w_ih.shape}, expected {(4 * h, self.n_in)}")
        for name, array, shape in (("w_hh", self.w_hh, (4 * h, h)),
                                   ("b_ih", self.b_ih, (4 * h,)),
                                   ("b_hh", self.b_hh, (4 * h,))):
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

    def __post_init__(self) -> None:
        if not self.layers:
            raise ReferenceError("a model with no layers cannot forecast")
        for layer in self.layers:
            layer.validate()
        if self.layers[0].n_in != self.n_channels:
            raise ReferenceError(
                f"layer 0 takes {self.layers[0].n_in} inputs but the model declares "
                f"{self.n_channels} channels"
            )
        expected = self.n_predictions * self.n_channels
        if self.head_w.shape != (expected, self.layers[-1].hidden):
            raise ReferenceError(
                f"head is {self.head_w.shape}, expected "
                f"{(expected, self.layers[-1].hidden)}"
            )

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
        return (f"LSTM {self.n_channels}ch -> {list(self.hidden)} -> "
                f"{self.n_predictions}x{self.n_channels}   "
                f"{self.n_parameters:,} parameters, {self.nbytes() / 1024:.1f} KiB float32")


#: Zero state for one layer: ``(h, c)``, each ``(batch, hidden)``.
State = tuple[np.ndarray, np.ndarray]


def zero_state(weights: Weights, batch: int) -> list[State]:
    return [(np.zeros((batch, layer.hidden), dtype=DTYPE),
             np.zeros((batch, layer.hidden), dtype=DTYPE))
            for layer in weights.layers]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic, in the branch form that does not overflow for large |x|."""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponent = np.exp(x[~positive])
    out[~positive] = exponent / (1.0 + exponent)
    return out


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
    """
    batch, steps, _ = x.shape
    projected = x.reshape(-1, layer.n_in) @ layer.w_ih.T
    projected = projected.reshape(batch, steps, 4 * layer.hidden)
    bias = layer.b_ih + layer.b_hh

    h, c = state
    out = np.empty((batch, steps, layer.hidden), dtype=DTYPE)
    for t in range(steps):
        h, c = lstm_cell(projected[:, t], h, c, layer.w_hh, bias)
        out[:, t] = h
    return out, (h, c)


def forward(weights: Weights, x: np.ndarray, state: list[State] | None = None,
            *, last_only: bool = False) -> tuple[np.ndarray, list[State]]:
    """Forecast from ``(batch, steps, n_channels)``.

    Returns ``(batch, steps, n_predictions, n_channels)`` -- the forecast made
    *at* each timestep for the ``n_predictions`` that follow it -- together with
    the state after the final step, so the next chunk can continue from here.
    ``last_only`` collapses the time axis to the final step, which is the shape
    telemanom trains on: one window in, one block of future values out.

    No dropout. Dropout exists only during training; a flight component that
    dropped a random 30% of its hidden units every cycle would violate
    Objective.md 11 rule 5, and PyTorch's eval mode does the same thing here by
    simply not being implemented.
    """
    x = np.ascontiguousarray(x, dtype=DTYPE)
    if x.ndim != 3:
        raise ReferenceError(f"input must be (batch, steps, channels), got {x.shape}")
    if x.shape[2] != weights.n_channels:
        raise ReferenceError(
            f"input has {x.shape[2]} channels, the model was fitted on {weights.n_channels}"
        )

    state = zero_state(weights, x.shape[0]) if state is None else list(state)
    if len(state) != len(weights.layers):
        raise ReferenceError(
            f"state carries {len(state)} layers, the model has {len(weights.layers)}"
        )

    activations = x
    out_state: list[State] = []
    for layer, layer_state in zip(weights.layers, state):
        activations, next_state = lstm_layer(activations, layer, layer_state)
        out_state.append(next_state)

    if last_only:
        activations = activations[:, -1:]
    batch, steps, hidden = activations.shape
    flat = activations.reshape(-1, hidden) @ weights.head_w.T + weights.head_b
    y = flat.reshape(batch, steps, weights.n_predictions, weights.n_channels)
    return (y[:, 0] if last_only else y), out_state
