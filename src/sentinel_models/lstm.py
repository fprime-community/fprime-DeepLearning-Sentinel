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
from .windows import (DEFAULT_DECAY_STEPS, SequenceSampler, split_runs,
                      usable_runs)

#: CPU only. MPS is not deterministic across releases and this model is far too
#: small to need a GPU; Objective.md 11 rule 5 wants reproducibility, not speed.
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
            "min_improvement": self.min_improvement,
            "validation_fraction": self.validation_fraction,
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
    n_exogenous: int = 0

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
            "exogenous_inputs": self.n_exogenous,
        }


class TelemanomLSTM(nn.Module):
    """LSTM(80) -> dropout -> LSTM(80) -> dropout -> Dense(l_p * channels).

    telemanom's Keras model puts a dropout after *each* LSTM, including the last.
    ``nn.LSTM`` only drops between layers, so the final one is explicit below --
    a difference of one line that would otherwise have been a silent deviation.
    """

    def __init__(self, n_channels: int, hyper: Hyper, n_exogenous: int = 0) -> None:
        super().__init__()
        if len(set(hyper.hidden)) != 1:
            raise ReferenceError(
                f"nn.LSTM requires one hidden size for all layers, got {hyper.hidden}. "
                f"The reference implementation supports unequal layers; the trainer does not."
            )
        self.hyper = hyper
        self.n_channels = n_channels
        self.n_exogenous = int(n_exogenous)
        self.lstm = nn.LSTM(
            input_size=n_channels + self.n_exogenous, hidden_size=hyper.hidden[0],
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
        n_exogenous=model.n_exogenous,
    )


def configure_determinism(seed: int) -> None:
    """Same seed, same machine, same weights. Objective.md 11 rule 5 in spirit."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)


def train(values: np.ndarray, usable: np.ndarray, hyper: Hyper, *, fold: int = 0,
          impulses: np.ndarray | None = None,
          decay_steps: int = DEFAULT_DECAY_STEPS, log=None
          ) -> tuple[Weights, TrainingReport]:
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
    exogenous = dict(impulses=impulses, decay_steps=decay_steps)
    trainer = SequenceSampler(values, train_runs, window=hyper.window,
                              n_predictions=hyper.n_predictions, **exogenous)
    validator = SequenceSampler(values, validation_runs, window=hyper.window,
                                n_predictions=hyper.n_predictions, **exogenous)
    if len(trainer) == 0:
        raise ReferenceError("the training side of the split holds no complete sequence")

    usable_steps = int(np.asarray(usable, dtype=bool).sum())
    per_epoch = hyper.sequences_per_epoch(usable_steps)
    report = TrainingReport(sequences_per_epoch=per_epoch, train_positions=len(trainer),
                            validation_positions=len(validator))

    # Two features per command -- the impulse, and how recently it fired.
    n_exogenous = 0 if impulses is None else 2 * impulses.shape[1]
    model = TelemanomLSTM(values.shape[1], hyper, n_exogenous).to(DEVICE)
    report.n_exogenous = n_exogenous
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

        if current < best_loss * (1.0 - hyper.min_improvement):
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
