"""Command line for the ground toolkit. The composition root, and nothing more.

    fit --telemetry X.npy --out model.bin    train, calibrate, write, verify
    verify --model model.bin                 read it back and say what it is
    selftest                                 the whole path on the generated
                                             fixture, zero cloud operations

House form is `src/sentinel_eval/cli.py`'s: global flags on the top parser,
subcommands with `dest="command"` required, dict dispatch, `cmd_x(args) -> int`,
one `except` around dispatch that prints `REFUSED: ...` to stderr and exits 2.

**Nothing here reaches the network.** There is no R2 client, no credential read
and no bucket in this package.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .errors import ToolkitError
from .fit import DEFAULT_TRAIN_FRACTION, FLOWN, fit_model
from .limits import DEFAULT_EWMA_SPAN, FLIGHT_LIMITS


def _load_telemetry(path: Path) -> np.ndarray:
    if not path.exists():
        raise ToolkitError(f"no telemetry at {path}")
    try:
        values = np.load(path, allow_pickle=False)
    except Exception as exc:                       # noqa: BLE001 -- reported, not swallowed
        raise ToolkitError(
            f"{path} is not a readable .npy array: {type(exc).__name__}: {exc}") from exc
    return np.asarray(values)


def _verify(blob: bytes, log=print) -> int:
    """Read the file back through the loader that mirrors the flight one."""
    from sentinel_export import Status, read_model

    status, spec = read_model(blob, FLIGHT_LIMITS)
    if status is not Status.OK:
        raise ToolkitError(
            f"the model this toolkit just wrote does not load: {status.name}. "
            "That is a defect in the toolkit, not in your telemetry")
    params = spec["params"]
    log(f"  verified   {len(blob):,} bytes, {spec['n_channels']} channels, "
        f"param_version {params['param_version']}, tier {params['tier']}, "
        f"warmup {params['warmup_steps']:,}")
    if params["param_version"] != 2:
        raise ToolkitError(
            f"param_version is {params['param_version']} and the fused rule needs 2; "
            "a version-1 file carrying a version-2 statistic is the silent drift "
            "`docs/MODEL_FILE.md` 6.2 forbids")
    return 0


def cmd_fit(args) -> int:
    values = _load_telemetry(Path(args.telemetry))
    print(f"  telemetry  {Path(args.telemetry).name}: {values.shape[0]:,} timesteps "
          f"x {values.shape[1] if values.ndim == 2 else '?'} channels")
    result = fit_model(
        values, window=args.window, hidden=tuple(args.hidden),
        n_predictions=args.predictions, max_epochs=args.epochs, seed=args.seed,
        quantile=args.quantile, train_fraction=args.train_fraction,
        ewma_span=args.ewma_span, tier=args.tier, mission=args.mission, log=print)

    out = Path(args.out)
    out.write_bytes(result.model_bytes)
    print(f"  wrote      {out} ({len(result.model_bytes):,} bytes)")
    _verify(result.model_bytes)

    # D85: the training segment is kept with every model, so the ground can
    # reproduce D78's control for any candidate this model's retrainer produces.
    from . import segment
    side = segment.write(out, result.model_bytes, values[:result.train_steps], {
        "window": args.window, "hidden": list(args.hidden), "n_predictions": args.predictions,
        "max_epochs": args.epochs, "seed": args.seed, "train_fraction": args.train_fraction,
        "holdout_rate": result.calibration.holdout_rate,
        "fit_rate": result.calibration.fit_rate})
    print(f"  wrote      {side} (the training segment, {result.train_steps:,} timesteps; "
          "mission telemetry -- keep it with the model)")

    from .report import render
    print()
    print(render(result.calibration, channels=values.shape[1],
                 train_steps=result.train_steps, window=args.window,
                 hidden=args.hidden, n_predictions=args.predictions,
                 model_bytes=len(result.model_bytes), param_version=2, tier=args.tier))
    return 0


def cmd_verify(args) -> int:
    path = Path(args.model)
    if not path.exists():
        raise ToolkitError(f"no model at {path}")
    return _verify(path.read_bytes())


def cmd_selftest(args) -> int:
    """The acceptance ladder's first rung, and it costs nothing to run.

    The fixture is `src/sentinel_eval/synthetic.py`'s -- a generated bucket with
    real parquet bytes under real manifest keys, so the genuine load path runs.
    The toolkit is handed the **longest contiguous anomaly-free run** of it,
    which is what "a mission's healthy telemetry" means.
    """
    from .selftest import run
    return run(seed=args.seed, steps=args.steps, log=print)


def cmd_gate(args) -> int:
    """D85: judge a downlinked candidate. Writes the report; transmits nothing."""
    import json as _json
    from . import gate as gate_mod
    telemetry = np.load(args.telemetry) if args.telemetry.endswith(".npy") else \
        np.fromfile(args.telemetry, dtype=np.float32).reshape(-1, args.channels)
    limits = _json.loads(Path(args.limits).read_text())
    res = gate_mod.gate(
        flying=Path(args.flying), candidate=Path(args.candidate), telemetry=telemetry,
        names=limits["names"], first_data=args.first, last_data=args.last,
        windows=args.windows, budget=args.budget, tools=Path(args.tools),
        yellow_low=np.asarray(limits["low"], dtype=np.float64),
        yellow_high=np.asarray(limits["high"], dtype=np.float64),
        block=args.block, blocks=args.blocks, horizon=args.horizon,
        workdir=Path(args.workdir), cache=Path(args.cache) if args.cache else None,
        uplink_dest=args.dest, dictionary=args.dictionary, reload_command=args.reload_command,
        orbit=args.orbit,
        baseline_record=_json.loads(Path(args.baseline).read_text()) if args.baseline else None,
        need_threshold=args.need_threshold, shadow=args.shadow)
    out = Path(args.report)
    out.write_text(gate_mod.to_json(res))
    Path(str(out) + ".txt").write_text(gate_mod.render(res) + "\n")
    print(gate_mod.render(res))
    print(f"  report     {out} (and {out.name}.txt)")
    return 0 if res.verdict == "CERTIFY" else 1


def cmd_baseline(args) -> int:
    """D85b N1: the flying model's residual over its first whole orbits in flight."""
    import json as _json
    from . import gate as gate_mod
    telemetry = np.load(args.telemetry) if args.telemetry.endswith(".npy") else \
        np.fromfile(args.telemetry, dtype=np.float32).reshape(-1, args.channels)
    rec = gate_mod.baseline(flying=Path(args.flying), telemetry=telemetry, start=args.start,
                            orbit=args.orbit, tools=Path(args.tools), workdir=Path(args.workdir))
    Path(args.out).write_text(_json.dumps(rec, indent=2, sort_keys=True))
    print(f"  baseline   ticks {rec['window'][0]:,}..{rec['window'][1]:,}: residual "
          f"{rec['residual']:.6e} -> {args.out}")
    return 0


def cmd_approve(args) -> int:
    """D85: the human's one command. Refuses anything but a CERTIFY report."""
    from .approve import approve
    return approve(Path(args.report), Path(args.candidate),
                   Path(args.event_log) if args.event_log else None,
                   dry_run=args.dry_run, timeout=args.timeout)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="sentinel_toolkit", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="healthy telemetry in, model.bin out")
    fit.add_argument("--telemetry", required=True,
                     help=".npy array of shape (timesteps, channels), healthy only")
    fit.add_argument("--out", required=True)
    fit.add_argument("--mission", default="mission")
    fit.add_argument("--window", type=int, default=FLOWN["window"])
    fit.add_argument("--hidden", type=int, nargs="+", default=list(FLOWN["hidden"]))
    fit.add_argument("--predictions", type=int, default=FLOWN["n_predictions"])
    fit.add_argument("--epochs", type=int, default=FLOWN["max_epochs"])
    fit.add_argument("--ewma-span", dest="ewma_span", type=int, default=DEFAULT_EWMA_SPAN)
    fit.add_argument("--tier", type=int, default=3, choices=(1, 2, 3))
    fit.add_argument("--train-fraction", dest="train_fraction", type=float,
                     default=DEFAULT_TRAIN_FRACTION)
    fit.add_argument("--quantile", type=float, default=None,
                     help="the dimensionless cut. NOT an alarm rate: this is a "
                          "quantile of your own nominal statistic, and no target "
                          "rate is accepted anywhere in this toolkit")

    verify = sub.add_parser("verify", help="read a model.bin back and describe it")
    verify.add_argument("--model", required=True)

    selftest = sub.add_parser("selftest", help="the generated fixture, end to end")
    selftest.add_argument("--steps", type=int, default=120_000)

    gate = sub.add_parser("gate", help="D85: judge a downlinked candidate against a "
                                       "control reproduced on the ground")
    gate.add_argument("--flying", required=True, help="the flying model.bin (its "
                      ".segment.npz sidecar must sit beside it)")
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--telemetry", required=True,
                      help="downlinked telemetry: .npy, or .f32 with --channels")
    gate.add_argument("--channels", type=int, default=0)
    gate.add_argument("--first", type=int, required=True, help="first tick the candidate trained on")
    gate.add_argument("--last", type=int, required=True, help="last tick the candidate trained on")
    gate.add_argument("--windows", type=int, required=True, help="windows the candidate trained on")
    gate.add_argument("--budget", type=int, default=1, help="steps per window (EXPERIMENTAL)")
    gate.add_argument("--tools", required=True, help="ground_tools, built by oxcaml_shape.sh")
    gate.add_argument("--limits", required=True,
                      help='JSON: {"names": [...], "low": [...], "high": [...]} (yellow)')
    gate.add_argument("--block", type=int, required=True,
                      help="the trend's block length in ticks, e.g. one orbit")
    gate.add_argument("--blocks", type=int, default=4)
    gate.add_argument("--horizon", type=int, default=100_000)
    gate.add_argument("--workdir", required=True)
    gate.add_argument("--cache", default=None, help="where reproduced controls are kept")
    gate.add_argument("--dest", default="RetrainApproved.bin")
    gate.add_argument("--dictionary", default="<the deployment's dictionary.json>")
    gate.add_argument("--reload-command", dest="reload_command",
                      default="SentinelRef.sentinelMonitor.RELOAD_MODEL")
    gate.add_argument("--report", required=True, help="where the JSON report is written")
    gate.add_argument("--orbit", type=int, default=None,
                      help="D85b: the telemetry's orbit in ticks; switches on need first, "
                           "replication and whole-orbit windows (docs/MODELS.md 81.2)")
    gate.add_argument("--baseline", default=None, help="D85b: the flying model's baseline JSON")
    gate.add_argument("--need-threshold", dest="need_threshold", type=float, default=None,
                      help="D85b: T, calibrated by 81.2's rule")
    gate.add_argument("--shadow", action="store_true",
                      help="D85b: judge a NOT NEEDED candidate for the record; never certifies")

    base = sub.add_parser("baseline", help="D85b: a flying model's residual over its first "
                                           "whole orbits in flight")
    base.add_argument("--flying", required=True)
    base.add_argument("--telemetry", required=True)
    base.add_argument("--channels", type=int, default=0)
    base.add_argument("--start", type=int, required=True, help="the tick it began flying")
    base.add_argument("--orbit", type=int, required=True)
    base.add_argument("--tools", required=True)
    base.add_argument("--workdir", required=True)
    base.add_argument("--out", required=True)

    appr = sub.add_parser("approve", help="D85: the human's one command -- uplink a "
                                          "CERTIFIED candidate and reload it")
    appr.add_argument("--report", required=True)
    appr.add_argument("--candidate", required=True)
    appr.add_argument("--event-log", dest="event_log", default=None,
                      help="the detector's own event log; its ModelReloadAccepted is the evidence")
    appr.add_argument("--dry-run", dest="dry_run", action="store_true")
    appr.add_argument("--timeout", type=float, default=60.0)

    args = parser.parse_args(argv)
    if getattr(args, "quantile", None) is None:
        from .calibrate import DEFAULT_QUANTILE
        args.quantile = DEFAULT_QUANTILE

    handlers = {"fit": cmd_fit, "verify": cmd_verify, "selftest": cmd_selftest,
                "gate": cmd_gate, "baseline": cmd_baseline, "approve": cmd_approve}
    try:
        return handlers[args.command](args)
    except ToolkitError as exc:
        print(f"\n  REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
