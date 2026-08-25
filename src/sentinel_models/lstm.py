"""telemanom's forecaster: two LSTM layers of 80 units, trained with PyTorch.

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
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .reference import DTYPE, LayerWeights, ReferenceError, Weights
from .windows import SequenceSampler, split_runs, usable_runs

#: CPU only. MPS is not deterministic across releases and this model is far too
#: small to need a GPU; Objective.md 11 rule 5 wants reproducibility, not speed.
DEVICE = "cpu"

#: Pinned so weights reproduce. CPU GEMM reduction order varies with the thread
#: count, so an unpinned count silently makes a run irreproducible on the same
#: machine. Recorded in the scorecard's provenance.
THREADS = 8


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
    min_delta: float = 3e-4              # published
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

    def sequences_per_epoch(self, usable_steps: int) -> int:
        return max(self.batch_size, int(round(usable_steps / self.sequence_budget_divisor)))

    def as_dict_key(self) -> tuple:
        """Hashable identity, so a cache cannot serve weights fitted differently."""
        return tuple(sorted(
            (k, tuple(v) if isinstance(v, list) else v) for k, v in self.as_dict().items()
        ))

    def as_dict(self) -> dict:
        return {
            "window": self.window, "hidden": list(self.hidden), "dropout": self.dropout,
            "n_predictions": self.n_predictions, "batch_size": self.batch_size,
            "max_epochs": self.max_epochs, "patience": self.patience,
            "min_delta": self.min_delta, "validation_fraction": self.validation_fraction,
            "learning_rate": self.learning_rate, "seed": self.seed,
            "sequence_budget_divisor": self.sequence_budget_divisor,
            "max_validation_sequences": self.max_validation_sequences,
            "restore_best": self.restore_best,
        }


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
        }


class TelemanomLSTM(nn.Module):
    """LSTM(80) -> dropout -> LSTM(80) -> dropout -> Dense(l_p * channels).

    telemanom's Keras model puts a dropout after *each* LSTM, including the last.
    ``nn.LSTM`` only drops between layers, so the final one is explicit below --
    a difference of one line that would otherwise have been a silent deviation.
    """

    def __init__(self, n_channels: int, hyper: Hyper) -> None:
        super().__init__()
        if len(set(hyper.hidden)) != 1:
            raise ReferenceError(
                f"nn.LSTM requires one hidden size for all layers, got {hyper.hidden}. "
                f"The reference implementation supports unequal layers; the trainer does not."
            )
        self.hyper = hyper
        self.n_channels = n_channels
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=hyper.hidden[0],
            num_layers=len(hyper.hidden), batch_first=True,
            dropout=hyper.dropout if len(hyper.hidden) > 1 else 0.0,
        )
        self.drop = nn.Dropout(hyper.dropout)
        self.head = nn.Linear(hyper.hidden[-1], hyper.n_predictions * n_channels)

    def forward(self, x: torch.Tensor, *, last_only: bool = True) -> torch.Tensor:
        out, _ = self.lstm(x)
        if last_only:
            out = out[:, -1:]
        y = self.head(self.drop(out))
        y = y.reshape(y.shape[0], y.shape[1], self.hyper.n_predictions, self.n_channels)
        return y[:, 0] if last_only else y


def to_weights(model: TelemanomLSTM) -> Weights:
    """Extract plain float32 arrays. The whole of what a `model.bin` must carry.

    Dropout contributes nothing: it is identity at inference, so it has no
    parameters to export and no line in the file format. That is worth stating
    because it is the first concrete thing work item 4 establishes about
    Objective.md decision 14.2.
    """
    def array(tensor: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=DTYPE)

    layers = tuple(
        LayerWeights(
            w_ih=array(getattr(model.lstm, f"weight_ih_l{i}")),
            w_hh=array(getattr(model.lstm, f"weight_hh_l{i}")),
            b_ih=array(getattr(model.lstm, f"bias_ih_l{i}")),
            b_hh=array(getattr(model.lstm, f"bias_hh_l{i}")),
        )
        for i in range(model.lstm.num_layers)
    )
    return Weights(
        layers=layers,
        head_w=array(model.head.weight),
        head_b=array(model.head.bias),
        n_channels=model.n_channels,
        window=model.hyper.window,
        n_predictions=model.hyper.n_predictions,
    )


def configure_determinism(seed: int) -> None:
    """Same seed, same machine, same weights. Objective.md 11 rule 5 in spirit."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)


def train(values: np.ndarray, usable: np.ndarray, hyper: Hyper, *, fold: int = 0,
          log=None) -> tuple[Weights, TrainingReport]:
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
    train_runs, validation_runs = split_runs(runs, hyper.validation_fraction)

    values = np.ascontiguousarray(values, dtype=DTYPE)
    trainer = SequenceSampler(values, train_runs, window=hyper.window,
                              n_predictions=hyper.n_predictions)
    validator = SequenceSampler(values, validation_runs, window=hyper.window,
                                n_predictions=hyper.n_predictions)
    if len(trainer) == 0:
        raise ReferenceError("the training side of the split holds no complete sequence")

    usable_steps = int(np.asarray(usable, dtype=bool).sum())
    per_epoch = hyper.sequences_per_epoch(usable_steps)
    report = TrainingReport(sequences_per_epoch=per_epoch, train_positions=len(trainer),
                            validation_positions=len(validator))

    model = TelemanomLSTM(values.shape[1], hyper).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=hyper.learning_rate)
    loss_fn = nn.MSELoss()

    validation = None
    if len(validator):
        positions = validator.sample_positions(hyper.max_validation_sequences,
                                               np.random.default_rng(hyper.seed + 9973))
        validation = validator.gather(positions)
        report.validation_positions = len(positions)

    best_state, best_loss, best_epoch, stale = None, float("inf"), -1, 0
    n_batches = max(1, per_epoch // hyper.batch_size)

    for epoch in range(hyper.max_epochs):
        model.train()
        for _ in range(n_batches):
            inputs, targets = trainer.draw(hyper.batch_size, rng)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(torch.from_numpy(inputs)), torch.from_numpy(targets))
            loss.backward()
            optimiser.step()

        report.epochs_run = epoch + 1
        current = _validate(model, validation, loss_fn) if validation else float(loss.item())
        report.history.append(current)
        if log:
            log(f"      epoch {epoch + 1:>2}/{hyper.max_epochs}  validation MSE {current:.6f}")

        if current < best_loss - hyper.min_delta:
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

    model.eval()
    return to_weights(model), report


def _validate(model: TelemanomLSTM, validation, loss_fn, chunk: int = 256) -> float:
    inputs, targets = validation
    model.eval()
    total, seen = 0.0, 0
    with torch.no_grad():
        for lo in range(0, inputs.shape[0], chunk):
            x = torch.from_numpy(inputs[lo:lo + chunk])
            y = torch.from_numpy(targets[lo:lo + chunk])
            total += float(loss_fn(model(x), y)) * x.shape[0]
            seen += x.shape[0]
    return total / max(seen, 1)
