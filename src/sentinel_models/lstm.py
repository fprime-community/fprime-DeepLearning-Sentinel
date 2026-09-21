"""telemanom's forecaster: two recurrent layers of 80 units, trained with PyTorch.

The cell is a field of :class:`Hyper` -- telemanom's LSTM by default, or the GRU
work item 5 adds as the second player at the architecture gate. One class,
:class:`TelemanomRNN`, builds both, so they differ by the cell and by nothing
else; :class:`TelemanomLSTM` and :class:`TelemanomGRU` are the two by name.

Hundman et al. (KDD 2018) is the reference method this whole project is built on
(Objective.md 3), and this module reproduces its forecaster. The published
configuration is transcribed in :class:`Hyper` and every departure from it is
named there or in docs/MODELS.md -- a claim of faithfulness that cannot be
audited is worth nothing, which is the lesson docs/HARNESS.md section 7 already
paid for once.

**Torch trains; it does not score.** A hand-written LSTM backward pass that is
subtly wrong yields a plausible bad number, and a plausible bad number is the
failure this project's documents are most afraid of, so gradients come from
autograd. Scoring then runs through :mod:`sentinel_models.reference`, plain
NumPy, which is what Phase 2's C++ is transcribed from.
:func:`to_weights` is the bridge, and `tests/test_reference_equivalence.py`
asserts the two agree to float32 noise. That assertion is the contract; if it
fails, nothing downstream means anything.

**Nothing is written to disk.** Weights live in memory for the life of the
process and are dropped on exit, because Rule 1 says the Mac is a pipe and
`tests/test_no_local_persistence.py` enforces it. Every run therefore retrains,
which is slower and is also the honest arrangement: there is no stale checkpoint
to score by accident.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import nn

from .reference import (DTYPE, ConvBlockWeights, ConvWeights, LayerWeights,
                        ReferenceError, Weights)
from .windows import (DEFAULT_DECAY_STEPS, SequenceSampler, split_runs,
                      usable_runs)

#: Where fitting happens. **CPU by default**, and never MPS -- Apple's backend is
#: not deterministic across releases.
#:
#: Set to ``"cuda"`` by `scripts/fit_folds.py` when fitting on a rented GPU, which
#: is 19x this laptop on the shape that matters and, measured on an A40, produces
#: **bit-identical weights across runs on the same box**. It does not produce the
#: same weights a different box would: reduction order differs with the hardware,
#: so "reproducible from cold" means on equivalent fitting hardware, and the
#: device is recorded in provenance for that reason.
#:
#: Scoring is unaffected either way -- it runs through
#: `sentinel_models.reference`, plain NumPy, on whatever machine scores.
DEVICE = "cpu"

#: Pinned so weights reproduce. CPU GEMM reduction order varies with the thread
#: count, so an unpinned count silently makes a run irreproducible on the same
#: machine. Recorded in the scorecard's provenance.
#:
#: Four rather than all ten, measured. Apple silicon pairs fast performance cores
#: with slow efficiency ones, and torch splits a GEMM evenly across whatever it is
#: given, so the fast cores end up waiting on the slow ones: 127 GFLOP/s on four
#: threads against 103 on ten, for this shape. More cores is slower here.
THREADS = 4

#: The architectures the trainer can build. telemanom's is the LSTM; the GRU is
#: work item 5's player and the TCN work item 6's. "cell" is kept as the field's
#: name although a TCN has no cell: it is the one field that says which player.
CELLS = ("lstm", "gru", "tcn")

#: Bai, Kolter and Koltun (2018): kernel 3, residual blocks at dilations 1, 2, 4,
#: ... -- the receptive field is ``1 + 2 (k - 1) (2^L - 1)``, so six blocks of
#: kernel 3 see 253 steps, the first shape past telemanom's ``l_s = 250``.
DEFAULT_KERNEL = 3


@dataclass(frozen=True)
class Hyper:
    """telemanom's published configuration, plus what scale forced us to change.

    Values marked *published* are transcribed from telemanom's `config.yaml`.
    The three that are not are called out, because ESA-ADB's training windows are
    three to four orders of magnitude larger than the SMAP/MSL channels the
    published settings were chosen for.
    """

    window: int = 250                    # published: l_s
    hidden: tuple[int, ...] = (80, 80)   # published: layers
    dropout: float = 0.3                 # published
    n_predictions: int = 10              # published: l_p
    batch_size: int = 70                 # published
    max_epochs: int = 35                 # published: epochs
    patience: int = 10                   # published

    #: **NOT published, and the published value is why.** telemanom's `min_delta`
    #: is 3e-4, applied as `current < best_loss - min_delta` -- an absolute
    #: quantity in the units of the loss. Validation MSE on ESA-ADB is ~1e-4, so
    #: after the first epoch the bar became `current < 1.7e-4 - 3e-4 = -1.3e-4`,
    #: which is negative and which no mean-squared error can satisfy. No epoch
    #: ever registered as an improvement, `best_epoch` stayed 0, `restore_best`
    #: restored epoch 1, and patience fired at epoch 11 -- for every fit in the
    #: project. Median 3.1x better weights discarded, worst case 8.8x.
    #:
    #: The replacement is a **fraction of the standing best**, not a smaller
    #: constant: picking another absolute number by eye reproduces the failure
    #: with different digits. A ratio has no units and cannot exceed the thing it
    #: is compared against, whatever the data's scale.
    #:
    #: Deriving it from the first epochs was the alternative and is worse: it
    #: would make the stopping rule depend on the noise of two or three samples
    #: and reintroduce a quantity that can silently go wrong.
    #:
    #: See docs/DECISIONS.md -- *dimensionless constants transfer, absolute ones
    #: in data units do not.*
    min_improvement: float = 0.001       # an epoch must beat the best by 0.1%
    validation_fraction: float = 0.2     # published as a *random* split; ours is the
                                         # chronological tail -- HARNESS.md section 3
    learning_rate: float = 1e-3          # Keras Adam default, which telemanom took
    seed: int = 0

    #: NOT published. telemanom trains 35 epochs over a channel's entire training
    #: set; ours holds 3.6M to 10.8M usable steps. The per-epoch sequence budget
    #: is instead **scaled to the fold**, one sequence per this many usable steps,
    #: so each fold trains in proportion to the history it actually has. A fixed
    #: budget would have made every fold see the same amount of data drawn from
    #: differently-sized pools, which destroys the data-sufficiency curve that
    #: forward chaining exists to produce (docs/HARNESS.md section 3, and
    #: Objective.md 10 -- the cold-start problem this project has to answer).
    sequence_budget_divisor: int = 180

    #: NOT published. Caps the pinned validation set so early stopping costs a
    #: fixed amount rather than growing with the fold.
    max_validation_sequences: int = 2_000

    #: NOT published. telemanom's Keras EarlyStopping stops but does not restore;
    #: we restore the best-validating weights, which is strictly the better
    #: estimator and is recorded rather than assumed.
    restore_best: bool = True

    #: **The published training configuration, `docs/MODELS.md` 28.1.** Every
    #: field below defaults to today's behaviour and is emitted by `as_dict` --
    #: and therefore into the weight-cache key -- **only when it is not the
    #: default**, which is D14's rule and the way `cell` was added for the GRU.
    #: `tests/test_published_training.py` pins that the banked LSTM digest and
    #: fingerprints are unmoved.

    #: T-b. `config.yaml` carries TWO batch sizes: `batch_size: 70` is the
    #: error-window batch and `lstm_batch_size: 64` is the training one
    #: (`modeling.py:99`). `batch_size` above was marked "published" and is the
    #: wrong one of the two. `None` keeps it; 64 is the published trainer.
    train_batch_size: int | None = None

    #: T-c. telemanom's absolute `min_delta`, applied as
    #: `current < best - min_delta`. D17 replaced it with a fraction because
    #: ESA-ADB validation MSE is ~1e-4 and the bar went negative; SMAP/MSL ships
    #: pre-scaled to (-1,1), where the constant may be sound. `None` keeps the
    #: fraction; 0.0003 is published.
    min_delta_absolute: float | None = None

    #: T-d. `"tail"` is the chronological hold-out (`docs/HARNESS.md` section 3).
    #: `"shuffled"` is telemanom's: windows are shuffled (`channel.py:62`) and
    #: Keras takes a fraction of them, so the split is a random subset of
    #: positions rather than the end of the stream.
    validation_split: str = "tail"

    #: T-e. `False` is the per-fold sequence budget, written for ESA-ADB's
    #: millions of steps. `True` is telemanom's epoch: every training window,
    #: once, per epoch.
    full_epochs: bool = False

    #: The output head. ``"point"`` is telemanom's: `l_p * C` values, one mean
    #: per prediction per channel, trained on MSE. ``"gaussian"`` emits
    #: `l_p * C * 2` -- a mean AND a log-variance per channel -- trained on
    #: Gaussian negative log-likelihood, so the detector predicts its own
    #: uncertainty and the detection statistic `(x - mu)/sigma` is dimensionless
    #: by construction. D56, `docs/MODELS.md` 31.
    #:
    #: Emitted by `as_dict` only when it is not the default, for `cell`'s reason:
    #: the banked fits were keyed before this field existed (D14).
    head: str = "point"

    #: Which recurrent cell. NOT a telemanom setting -- telemanom is an LSTM.
    #: Work item 5 adds the GRU as the second player at the architecture gate,
    #: and it must differ from the LSTM by the cell alone; so the cell is a field
    #: on the *same* `Hyper`, and every other value above is shared by
    #: construction rather than by copying.
    #:
    #: Emitted by `as_dict` only when it is not the default. The weight-cache key
    #: and every artifact fingerprint derive from that dict, and the twelve banked
    #: LSTM fits were keyed before this field existed: docs/DECISIONS.md D14 --
    #: an optional field at its null value must leave the hash unmoved.
    #: `tests/test_lstm_detector.py` pins the LSTM's dict, key and fingerprints.
    cell: str = "lstm"

    #: TCN only: the convolution kernel. ``hidden`` is reused as the width of
    #: each residual block, so ``(50,) * 6`` is six blocks of 50 channels at
    #: dilations 1 to 32. Emitted by `as_dict` only for a TCN, for the reason
    #: `cell` is emitted only when it is not the default.
    kernel: int = DEFAULT_KERNEL

    def __post_init__(self) -> None:
        if self.cell not in CELLS:
            raise ReferenceError(f"unknown cell {self.cell!r}; expected one of {CELLS}")
        if self.kernel < 2:
            raise ReferenceError(f"kernel must be at least 2, got {self.kernel}")
        if self.head not in ("point", "gaussian"):
            raise ReferenceError(
                f"unknown head {self.head!r}; expected 'point' or 'gaussian'")
        if self.validation_split not in ("tail", "shuffled"):
            raise ReferenceError(
                f"unknown validation_split {self.validation_split!r}; "
                f"expected 'tail' or 'shuffled'")

    @property
    def receptive_field(self) -> int:
        """How far back one forecast can see. The window for a recurrent cell."""
        if self.cell != "tcn":
            return self.window
        return 1 + 2 * (self.kernel - 1) * (2 ** len(self.hidden) - 1)

    def sequences_per_epoch(self, usable_steps: int) -> int:
        return max(self.batch_size, int(round(usable_steps / self.sequence_budget_divisor)))

    def as_dict_key(self) -> tuple:
        """Hashable identity, so a cache cannot serve weights fitted differently."""
        return tuple(sorted(
            (k, tuple(v) if isinstance(v, list) else v) for k, v in self.as_dict().items()
        ))

    def as_dict(self) -> dict:
        out = {
            "window": self.window, "hidden": list(self.hidden), "dropout": self.dropout,
            "n_predictions": self.n_predictions, "batch_size": self.batch_size,
            "max_epochs": self.max_epochs, "patience": self.patience,
            "min_improvement": self.min_improvement,
            "validation_fraction": self.validation_fraction,
            "learning_rate": self.learning_rate, "seed": self.seed,
            "sequence_budget_divisor": self.sequence_budget_divisor,
            "max_validation_sequences": self.max_validation_sequences,
            "restore_best": self.restore_best,
        }
        # Each absent at its default, so every banked fit keeps its key: D14.
        if self.train_batch_size is not None:
            out["train_batch_size"] = self.train_batch_size
        if self.min_delta_absolute is not None:
            out["min_delta_absolute"] = self.min_delta_absolute
        if self.validation_split != "tail":
            out["validation_split"] = self.validation_split
        if self.full_epochs:
            out["full_epochs"] = self.full_epochs
        if self.head != "point":
            out["head"] = self.head
        if self.cell != "lstm":
            out["cell"] = self.cell          # absent for the LSTM: D14, hash unmoved
        if self.cell == "tcn":
            out["kernel"] = self.kernel
        return out


@dataclass
class TrainingReport:
    """What the fit actually did. Goes into the detector's provenance."""

    epochs_run: int = 0
    sequences_per_epoch: int = 0
    train_positions: int = 0
    validation_positions: int = 0
    best_epoch: int = -1
    best_validation_mse: float = float("nan")
    history: list[float] = field(default_factory=list)
    stopped_early: bool = False
    threads: int = THREADS
    n_exogenous: int = 0
    #: What `best_validation_mse` and `validation_history` are in. `"mse"` for a
    #: point head, `"nll"` for a Gaussian one -- the field names predate D56 and
    #: are not renamed, so the unit is carried explicitly instead.
    objective: str = "mse"

    def as_dict(self) -> dict:
        return {
            "epochs_run": self.epochs_run,
            "sequences_per_epoch": self.sequences_per_epoch,
            "train_start_positions": self.train_positions,
            "validation_start_positions": self.validation_positions,
            "best_epoch": self.best_epoch,
            "best_validation_mse": self.best_validation_mse,
            "validation_history": [round(v, 8) for v in self.history],
            "stopped_early": self.stopped_early,
            "torch_threads": self.threads,
            "objective": self.objective,
            "exogenous_inputs": self.n_exogenous,
        }


class TelemanomRNN(nn.Module):
    """RNN(80) -> dropout -> RNN(80) -> dropout -> Dense(l_p * channels).

    The cell is ``hyper.cell``: telemanom's LSTM, or the GRU work item 5 adds
    beside it. **Everything else is this one class** -- the same head, the same
    dropout placement, the same forward -- so the two architectures differ by
    the cell alone, which is what lets the gate compare cells rather than
    pipelines (docs/MODELS.md section 1: "items 5 and 6 have to differ from
    this by architecture alone").

    telemanom's Keras model puts a dropout after *each* recurrent layer,
    including the last. ``nn.LSTM`` and ``nn.GRU`` only drop between layers, so
    the final one is explicit below -- a difference of one line that would
    otherwise have been a silent deviation.
    """

    #: torch's modules share the parameter naming (`weight_ih_l{i}`, ...) and the
    #: stacking convention; only the gate count differs, 4 against 3.
    CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, n_channels: int, hyper: Hyper, n_exogenous: int = 0) -> None:
        super().__init__()
        if len(set(hyper.hidden)) != 1:
            raise ReferenceError(
                f"nn.{hyper.cell.upper()} requires one hidden size for all layers, got "
                f"{hyper.hidden}. The reference implementation supports unequal layers; "
                f"the trainer does not."
            )
        self.hyper = hyper
        self.n_channels = n_channels
        self.n_exogenous = int(n_exogenous)
        self.rnn = self.CELLS[hyper.cell](
            input_size=n_channels + self.n_exogenous, hidden_size=hyper.hidden[0],
            num_layers=len(hyper.hidden), batch_first=True,
            dropout=hyper.dropout if len(hyper.hidden) > 1 else 0.0,
        )
        self.drop = nn.Dropout(hyper.dropout)
        #: `gaussian` doubles the width: a mean and a log-variance per channel
        #: per prediction (D56). `point` is unchanged, so every banked fit and
        #: every golden vector keeps its shape.
        self.outputs = 2 if hyper.head == "gaussian" else 1
        self.head = nn.Linear(hyper.hidden[-1],
                              hyper.n_predictions * n_channels * self.outputs)

    @property
    def cell(self) -> str:
        return self.hyper.cell

    def forward(self, x: torch.Tensor, *, last_only: bool = True) -> torch.Tensor:
        out, _ = self.rnn(x)
        if last_only:
            out = out[:, -1:]
        y = self.head(self.drop(out))
        shape = (y.shape[0], y.shape[1], self.hyper.n_predictions, self.n_channels)
        y = y.reshape(*shape, self.outputs) if self.outputs > 1 else y.reshape(*shape)
        return y[:, 0] if last_only else y


class TelemanomLSTM(TelemanomRNN):
    """telemanom's cell, by name. The class declares the cell; the `Hyper` follows."""

    def __init__(self, n_channels: int, hyper: Hyper, n_exogenous: int = 0) -> None:
        super().__init__(n_channels, replace(hyper, cell="lstm"), n_exogenous)

    @property
    def lstm(self) -> nn.LSTM:
        return self.rnn


class TelemanomGRU(TelemanomRNN):
    """Work item 5's cell, by name. Three gates, one state vector, the same everything else."""

    def __init__(self, n_channels: int, hyper: Hyper, n_exogenous: int = 0) -> None:
        super().__init__(n_channels, replace(hyper, cell="gru"), n_exogenous)

    @property
    def gru(self) -> nn.GRU:
        return self.rnn


class TelemanomTCN(nn.Module):
    """Six residual blocks of causal dilated convolutions, then Dense(l_p * channels).

    Bai, Kolter and Koltun, *An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling* (2018): each block is two
    causal convolutions of kernel ``k`` at dilation ``2^i``, each followed by
    ReLU and dropout, added to a 1x1 projection of the block's input and passed
    through a final ReLU. No weight normalisation -- it is an optimisation aid
    that is folded into plain weights at export, so the file format never sees
    it, and leaving it out keeps the parameter count exact.

    **Nothing is carried between timesteps.** The forecast at ``t`` is a
    function of inputs ``t - receptive_field + 1 .. t`` and nothing else; a
    flight component holds that many inputs in a ring buffer, zero-filled at
    boot, and there is no state to corrupt or restore (Objective.md section 8).
    Causal padding here is that zero-filled buffer: ``(k - 1) * d`` zeros on the
    left of every convolution, never on the right.

    The head, the final dropout and ``forward``'s contract are the recurrent
    module's, so the three players differ in the block that reads the history
    and in nothing after it.
    """

    def __init__(self, n_channels: int, hyper: Hyper, n_exogenous: int = 0) -> None:
        super().__init__()
        if hyper.cell != "tcn":
            raise ReferenceError(f"TelemanomTCN needs Hyper(cell='tcn'), got {hyper.cell!r}")
        if len(set(hyper.hidden)) != 1:
            raise ReferenceError(
                f"the TCN uses one width for all blocks, got {hyper.hidden}"
            )
        self.hyper = hyper
        self.n_channels = n_channels
        self.n_exogenous = int(n_exogenous)
        width, k = hyper.hidden[0], hyper.kernel
        blocks, c_in = [], n_channels + self.n_exogenous
        for i in range(len(hyper.hidden)):
            blocks.append(_TCNBlock(c_in, width, k, 2 ** i, hyper.dropout))
            c_in = width
        self.blocks = nn.ModuleList(blocks)
        self.drop = nn.Dropout(hyper.dropout)
        self.head = nn.Linear(width, hyper.n_predictions * n_channels)

    @property
    def cell(self) -> str:
        return "tcn"

    @property
    def receptive_field(self) -> int:
        return self.hyper.receptive_field

    def forward(self, x: torch.Tensor, *, last_only: bool = True) -> torch.Tensor:
        y = x.transpose(1, 2)                       # (batch, channels, steps)
        for block in self.blocks:
            y = block(y)
        y = y.transpose(1, 2)                       # (batch, steps, width)
        if last_only:
            y = y[:, -1:]
        y = self.head(self.drop(y))
        y = y.reshape(y.shape[0], y.shape[1], self.hyper.n_predictions, self.n_channels)
        return y[:, 0] if last_only else y


class _TCNBlock(nn.Module):
    """One residual block: two causal dilated convolutions and a 1x1 shortcut."""

    def __init__(self, c_in: int, width: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, width, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, kernel, dilation=dilation)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, width, 1) if c_in != width else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop1(torch.relu(self.conv1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop2(torch.relu(self.conv2(nn.functional.pad(y, (self.pad, 0)))))
        return torch.relu(y + (x if self.res is None else self.res(x)))


def gaussian_nll(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood, constant dropped. D56.

    `prediction[..., 0]` is the mean and `prediction[..., 1]` the log-variance.
    The log-variance is clamped because an unbounded one lets the model drive
    sigma to zero on an easy channel and take the loss to minus infinity, which
    is a degenerate optimum rather than a good forecast -- the standard failure
    of a heteroscedastic head, and the clamp is stated rather than tuned.
    """
    mu = prediction[..., 0]
    log_var = prediction[..., 1].clamp(-10.0, 10.0)
    return (0.5 * (log_var + (target - mu) ** 2 * torch.exp(-log_var))).mean()


def build_model(n_channels: int, hyper: Hyper, n_exogenous: int = 0) -> nn.Module:
    """The one place the architecture is chosen. Everything in `train` is shared."""
    if hyper.cell == "tcn":
        return TelemanomTCN(n_channels, hyper, n_exogenous)
    return TelemanomRNN(n_channels, hyper, n_exogenous)


def to_weights(model: nn.Module):
    """Extract plain float32 arrays. The whole of what a `model.bin` must carry.

    Dropout contributes nothing: it is identity at inference, so it has no
    parameters to export and no line in the file format. That is worth stating
    because it is the first concrete thing work item 4 establishes about
    Objective.md decision 14.2.
    """
    if isinstance(model, TelemanomTCN):
        return _conv_weights(model)
    return _rnn_weights(model)


def _conv_weights(model: TelemanomTCN) -> ConvWeights:
    def array(tensor: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=DTYPE)

    blocks = tuple(
        ConvBlockWeights(
            w1=array(b.conv1.weight), b1=array(b.conv1.bias),
            w2=array(b.conv2.weight), b2=array(b.conv2.bias),
            res_w=None if b.res is None else array(b.res.weight),
            res_b=None if b.res is None else array(b.res.bias),
            dilation=b.conv1.dilation[0],
        )
        for b in model.blocks
    )
    return ConvWeights(
        blocks=blocks, head_w=array(model.head.weight), head_b=array(model.head.bias),
        n_channels=model.n_channels, window=model.hyper.window,
        n_predictions=model.hyper.n_predictions, n_exogenous=model.n_exogenous,
    )


def _rnn_weights(model: TelemanomRNN) -> Weights:
    def array(tensor: torch.Tensor) -> np.ndarray:
        """Off the device and into a plain array. This is what makes a fit
        portable: what leaves here has no tie to the machine that produced it."""
        return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=DTYPE)

    layers = tuple(
        LayerWeights(
            w_ih=array(getattr(model.rnn, f"weight_ih_l{i}")),
            w_hh=array(getattr(model.rnn, f"weight_hh_l{i}")),
            b_ih=array(getattr(model.rnn, f"bias_ih_l{i}")),
            b_hh=array(getattr(model.rnn, f"bias_hh_l{i}")),
        )
        for i in range(model.rnn.num_layers)
    )
    return Weights(
        layers=layers,
        head_w=array(model.head.weight),
        head_b=array(model.head.bias),
        n_channels=model.n_channels,
        window=model.hyper.window,
        n_predictions=model.hyper.n_predictions,
        n_exogenous=model.n_exogenous,
        outputs=getattr(model, "outputs", 1),
    )


def load_rnn_weights(model: TelemanomRNN, weights: Weights) -> None:
    """The inverse of :func:`_rnn_weights`: put plain arrays back into a module.

    (!) THIS EXISTS FOR D76 AND FOR NOTHING ELSE. The flown retrainer initialises
    the shadow from the flying model's weights and fine-tunes -- "there is no
    random initialisation on the flight path" -- and that had no implementation
    anywhere in this stack. A cold fit is unaffected: every caller defaults to
    ``None`` and takes the path it always took.

    Shapes are checked rather than broadcast. A silent reshape here would warm
    start from something that is not the flying model, and the arm measuring how
    far two warm starts diverge would be measuring the reshape.
    """
    if model.rnn.num_layers != len(weights.layers):
        raise ValueError(
            f"model has {model.rnn.num_layers} layers, weights have "
            f"{len(weights.layers)}")
    with torch.no_grad():
        for i, layer in enumerate(weights.layers):
            for name, array in (("weight_ih_l", layer.w_ih), ("weight_hh_l", layer.w_hh),
                                ("bias_ih_l", layer.b_ih), ("bias_hh_l", layer.b_hh)):
                target = getattr(model.rnn, f"{name}{i}")
                source = np.asarray(array, dtype=DTYPE)
                if tuple(target.shape) != source.shape:
                    raise ValueError(
                        f"{name}{i} is {tuple(target.shape)} in the model and "
                        f"{source.shape} in the weights")
                target.copy_(torch.from_numpy(source).to(target.device))
        for target, source in ((model.head.weight, weights.head_w),
                               (model.head.bias, weights.head_b)):
            source = np.asarray(source, dtype=DTYPE)
            if tuple(target.shape) != source.shape:
                raise ValueError(
                    f"head is {tuple(target.shape)} in the model and {source.shape} "
                    f"in the weights")
            target.copy_(torch.from_numpy(source).to(target.device))


def configure_determinism(seed: int) -> None:
    """Same seed, same machine, same weights. Objective.md 11 rule 5 in spirit."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)


def train(values: np.ndarray, usable: np.ndarray, hyper: Hyper, *, fold: int = 0,
          impulses: np.ndarray | None = None,
          decay_steps: int = DEFAULT_DECAY_STEPS, log=None,
          init_weights: Weights | None = None
          ) -> tuple[Weights | ConvWeights, TrainingReport]:
    """Fit on nominal data only and return plain arrays plus what the fit did.

    ``values`` is the training window, ``usable`` the harness's mask over it:
    inside the window, not annotated anomalous, not a gap, observed everywhere.
    Objective.md 6.1 -- a spacecraft has no failure examples, so the model learns
    normality and everything else is suspicious by exclusion.
    """
    configure_determinism(hyper.seed + fold)
    rng = np.random.default_rng(hyper.seed + fold)

    span = hyper.window + hyper.n_predictions
    runs = usable_runs(usable, span)
    if not runs:
        raise ReferenceError(
            f"no usable run reaches {span} steps in a window of {len(usable):,}; "
            f"there is nothing to fit on"
        )
    values = np.ascontiguousarray(values, dtype=DTYPE)
    exogenous = dict(impulses=impulses, decay_steps=decay_steps)
    train_positions = None
    if hyper.validation_split == "shuffled":
        # 28.1 T-d. telemanom shuffles the windows (`channel.py:62`) and Keras
        # takes a fraction of the shuffled array, so the hold-out is a random
        # subset of POSITIONS. Splitting runs cannot express that, so one sampler
        # is built over everything and the positions are partitioned.
        trainer = validator = SequenceSampler(values, runs, window=hyper.window,
                                              n_predictions=hyper.n_predictions,
                                              **exogenous)
        if len(trainer) == 0:
            raise ReferenceError("the training side of the split holds no complete sequence")
        order = np.random.default_rng(hyper.seed + fold).permutation(trainer.n_positions)
        n_validation = int(round(trainer.n_positions * hyper.validation_fraction))
        validation_positions = trainer.resolve_positions(order[:n_validation])
        train_positions = trainer.resolve_positions(order[n_validation:])
        if train_positions.size == 0:
            raise ReferenceError("the training side of the split holds no complete sequence")
    else:
        train_runs, validation_runs = split_runs(runs, hyper.validation_fraction)
        trainer = SequenceSampler(values, train_runs, window=hyper.window,
                                  n_predictions=hyper.n_predictions, **exogenous)
        validator = SequenceSampler(values, validation_runs, window=hyper.window,
                                    n_predictions=hyper.n_predictions, **exogenous)
        if len(trainer) == 0:
            raise ReferenceError("the training side of the split holds no complete sequence")

    usable_steps = int(np.asarray(usable, dtype=bool).sum())
    batch_size = hyper.train_batch_size or hyper.batch_size      # 28.1 T-b
    n_train_positions = (len(train_positions) if train_positions is not None
                         else len(trainer))
    # 28.1 T-e. telemanom's epoch is every training window once; ours is a budget
    # scaled to the fold, because ESA-ADB folds hold millions of steps.
    per_epoch = (n_train_positions if hyper.full_epochs
                 else hyper.sequences_per_epoch(usable_steps))
    # `threads` is read here, at fit time, not at class definition: the module
    # global is rebound by scripts/fit_folds.py, and the six GRU fits of
    # 2026-08-28 recorded the default 4 while running on one thread because it
    # was not. On CUDA the count does not touch the arithmetic; the record
    # should still say what happened.
    report = TrainingReport(sequences_per_epoch=per_epoch, train_positions=len(trainer),
                            validation_positions=len(validator), threads=THREADS)

    # Two features per command -- the impulse, and how recently it fired.
    n_exogenous = 0 if impulses is None else 2 * impulses.shape[1]
    # Built here, not at import: `DEVICE` is a module global that
    # scripts/fit_folds.py rebinds at run time, and the cell is read from `hyper`
    # so the LSTM and the GRU share every line of this function.
    model = build_model(values.shape[1], hyper, n_exogenous).to(DEVICE)
    # D76: the shadow starts from the flying model's weights, not from a
    # distribution. `configure_determinism` above has already seeded everything
    # the fit uses -- batch order, dropout, validation sampling -- so a warm
    # start replaces only the INITIALISATION and leaves the rest reproducible.
    if init_weights is not None:
        if not isinstance(model, TelemanomRNN):
            raise TypeError("warm start is implemented for the GRU only (D28)")
        load_rnn_weights(model, init_weights)
    report.n_exogenous = n_exogenous
    optimiser = torch.optim.Adam(model.parameters(), lr=hyper.learning_rate)
    loss_fn = gaussian_nll if hyper.head == "gaussian" else nn.MSELoss()
    report.objective = "nll" if hyper.head == "gaussian" else "mse"

    validation = None
    if train_positions is not None:
        positions = validation_positions[:hyper.max_validation_sequences]
        if positions.size:
            validation = validator.gather(positions)
            report.validation_positions = len(positions)
    elif len(validator):
        positions = validator.sample_positions(hyper.max_validation_sequences,
                                               np.random.default_rng(hyper.seed + 9973))
        validation = validator.gather(positions)
        report.validation_positions = len(positions)

    best_state, best_loss, best_epoch, stale = None, float("inf"), -1, 0
    n_batches = max(1, per_epoch // batch_size)

    def _draw():
        if train_positions is None:
            return trainer.draw(batch_size, rng)
        picks = train_positions[rng.integers(0, train_positions.size, size=batch_size)]
        return trainer.gather(picks)

    for epoch in range(hyper.max_epochs):
        model.train()
        for _ in range(n_batches):
            inputs, targets = _draw()
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(torch.from_numpy(inputs).to(DEVICE)),
                           torch.from_numpy(targets).to(DEVICE))
            loss.backward()
            optimiser.step()

        report.epochs_run = epoch + 1
        current = _validate(model, validation, loss_fn) if validation else float(loss.item())
        report.history.append(current)
        if log:
            log(f"      epoch {epoch + 1:>2}/{hyper.max_epochs}  validation MSE {current:.6f}")

        # 28.1 T-c. The published bar is absolute; ours is a fraction of the
        # standing best, because D17 measured the absolute one going negative
        # against ESA-ADB's ~1e-4 loss. Which is right is a property of the
        # data's scale, so both are available and the arm says which it used.
        if not math.isfinite(best_loss):
            improved = True                      # the first epoch always counts
        elif hyper.min_delta_absolute is not None:
            improved = current < best_loss - hyper.min_delta_absolute
        else:
            # `best * (1 - m)` and `best - |best| * m` are IDENTICAL for a
            # non-negative loss, so every MSE fit in this repository is unmoved.
            # They differ once the loss can be negative, which a Gaussian NLL
            # routinely is: the multiplicative form then RAISES the bar as the
            # loss improves and training stops almost immediately. Caught by
            # `docs/MODELS.md` 31's two-channel smoke, before the full run.
            improved = current < best_loss - abs(best_loss) * hyper.min_improvement
        if improved:
            best_loss, best_epoch, stale = current, epoch, 0
            if hyper.restore_best:
                best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= hyper.patience:
                report.stopped_early = True
                break

    if hyper.restore_best and best_state is not None:
        model.load_state_dict(best_state)
    report.best_epoch, report.best_validation_mse = best_epoch, best_loss
    _refuse_a_stopping_rule_that_rejects_everything(report, hyper)

    model.eval()
    return to_weights(model), report


#: Epochs after which keeping the first one is a defect rather than a result. A
#: model that genuinely peaks at epoch 1 and never improves across this many more
#: is not a model anyone should ship either, so the tripwire is worth its rare
#: false positive.
SUSPICIOUS_AFTER_EPOCHS = 5


def _refuse_a_stopping_rule_that_rejects_everything(report: "TrainingReport",
                                                    hyper: Hyper) -> None:
    """Fail if training kept its first epoch. Not a warning -- a failure.

    This is the defect that disabled training for every model in the project, and
    it announced itself in no way at all: no error, no warning, just a fit that
    stopped at epoch 11 with `best_epoch = 0` and weights up to 8.8x worse than
    ones it had already computed and thrown away.

    A warning would not have been enough. Nothing was reading the training
    reports, which is precisely why it survived three work items; a line of output
    nobody looks at is indistinguishable from silence. So this raises, and the
    message names the parameter to look at first.
    """
    if report.best_epoch != 0 or report.epochs_run <= SUSPICIOUS_AFTER_EPOCHS:
        return
    history = report.history
    better = min(history) if history else float("nan")
    raise ReferenceError(
        f"training kept its FIRST epoch after running {report.epochs_run}: the "
        f"stopping rule is rejecting every improvement.\n"
        f"  best epoch 0 at {report.best_validation_mse:.3e}; best seen "
        f"{better:.3e} ({report.best_validation_mse / better:.1f}x better, discarded)\n"
        f"  Check `min_improvement` ({hyper.min_improvement}) first. It is a "
        f"FRACTION of the standing best; an absolute value here -- telemanom's "
        f"published min_delta=3e-4 against a ~1e-4 loss -- makes the bar negative "
        f"and no epoch can ever clear it. See docs/DECISIONS.md."
    )


def _validate(model: nn.Module, validation, loss_fn, chunk: int = 256) -> float:
    inputs, targets = validation
    model.eval()
    total, seen = 0.0, 0
    with torch.no_grad():
        for lo in range(0, inputs.shape[0], chunk):
            x = torch.from_numpy(inputs[lo:lo + chunk]).to(DEVICE)
            y = torch.from_numpy(targets[lo:lo + chunk]).to(DEVICE)
            total += float(loss_fn(model(x), y)) * x.shape[0]
            seen += x.shape[0]
    return total / max(seen, 1)
