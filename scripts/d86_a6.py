#!/usr/bin/env python3
"""D86.A6, the accumulated signed residual on SMAP/MSL, pre-registered in docs/MODELS.md 79.13.

    PYTHONPATH=src .venv/bin/python scripts/d86_a6.py tune
    PYTHONPATH=src .venv/bin/python scripts/d86_a6.py eval

SMAP/MSL is read from R2 INTO MEMORY in each phase and nothing but results is written
(D86.4: no dataset copy on the Mac). Every object is checked against its manifest sha256;
every operation is counted and committed to the ledger. The cached univariate fits are read
under their original key (79.2). `eval` refuses unless `runs/d86/a6_frozen.json` is
byte-identical to `tests/fixtures/d86/a6_frozen.json` at HEAD and the tracked tree is clean.
No wall-clock figure is printed (stop 35).
"""
from __future__ import annotations

import hashlib, importlib.util, io, json, subprocess, sys  # noqa: E401
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_data import config as C                                  # noqa: E402
from sentinel_eval import ops, read                                    # noqa: E402
from sentinel_models import registry, telemanom                        # noqa: E402

spec = importlib.util.spec_from_file_location("d86_arms", ROOT / "scripts" / "d86_arms.py")
D = importlib.util.module_from_spec(spec); sys.modules["d86_arms"] = D
spec.loader.exec_module(D)

K_GRID = (0.5, 1.0, 2.0)
OUT = ROOT / "runs" / "d86" / "a6_frozen.json"
FIXTURE = "tests/fixtures/d86/a6_frozen.json"
EVAL_OUT = ROOT / "runs" / "d86" / "a6_eval.json"


def cusum(u, k):
    """79.13: two-sided Page CUSUM; the floor at 0 is the only reset, none at an alarm."""
    sp = np.empty_like(u); sm = np.empty_like(u); a = b = 0.0
    for t, v in enumerate(u):
        a = max(0.0, a + v - k); b = max(0.0, b - v - k)
        sp[t], sm[t] = a, b
    return np.maximum(sp, sm)


class Memory:
    """R2, read once per object into memory; verified against the manifest."""

    def __init__(self, source):
        self.source, self.blobs = source, {}

    def get(self, key):
        if key not in self.blobs:
            self.blobs[key] = self.source.get(key)
        return self.blobs[key]


def load():
    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    mem = Memory(read.R2Source(client, cfg.bucket))
    man = json.loads(mem.get(D.SMAP_MANIFEST))
    D.smap_source = lambda: (mem, man)                    # D86's reader, now from memory
    return client, budget, ledger, cfg


def channel_terms(cid):
    train, test = D.smap_channel(cid)                       # sha256-checked in D86's reader
    uni = registry.build("gru-telemanom"); uni.config = D.SR.proportional_config(len(test))
    v_tr, v_te = train[:, :1].astype(np.float32), test[:, :1].astype(np.float32)
    D.load_cached(uni, v_tr, np.ones(len(v_tr), bool), D.SR.ctx_for(cid, len(v_tr)))
    cfg = uni.config
    x = v_te[:, 0].astype(np.float64)
    agg, _, _ = D.forecast_parts(uni, v_te)
    e = x - agg[:, 0].astype(np.float64)
    e_s = D.ewma(np.abs(e), cfg.smoothing_window)
    ref = np.asarray(uni._smoothed_errors(v_te, D.SR.ctx_for(cid, len(v_te))), np.float64)[:, 0]
    if not np.array_equal(ref, e_s):
        raise RuntimeError("e_s differs from detectors._smoothed_errors")
    # u's reference: the SAME model's residual on the channel's TRAIN split (79.13)
    xt = v_tr[:, 0].astype(np.float64)
    agt, _, _ = D.forecast_parts(uni, v_tr)
    et = (xt - agt[:, 0].astype(np.float64))[uni.hyper.window:]
    scale = D.channel_scale(train[:, 0], unit=1.0)
    u = (e - float(et.mean())) / max(float(et.std()), D.FLOOR_FRACTION * scale)
    span = cfg.error_window
    T = {"zr": D.zstat(e_s, span), "zd": D.zstat(D.derivative(x), span),
         "frozen": np.asarray(telemanom.channel_ratios(e_s, cfg), np.float64)}
    for k in K_GRID:
        T[f"S{k}"] = cusum(u, k)
    return dict(steps=len(x), warmup=int(uni.warmup_steps), T=T)


def build_per():
    spans = D.smap_labels()
    per, skipped = {}, {}
    for cid in D.smap_units():
        try:
            c = channel_terms(cid)
        except Exception as exc:
            skipped[cid] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
            continue
        anom = np.zeros(c["steps"], bool)
        for lo, hi in spans.get(cid, []):
            anom[lo:hi + 1] = True
        nominal = ~anom; nominal[:c["warmup"]] = False
        per[cid] = dict(steps=c["steps"], nominal=nominal, T=c["T"])
    return per, skipped


ARMS6 = {"frozen": ["frozen"], "A0": ["zr", "zd"], "A1": ["zd"]}


def score(d, arm, k=None):
    names = ["zr", "zd", f"S{k}"] if arm == "A6" else ARMS6[arm]
    return np.maximum.reduce([d["T"][n] for n in names]), names


def attribution(per, names, cut, windows):
    out = {}
    for c, lo, hi in windows:
        st = np.stack([per[c]["T"][n][lo:hi + 1] for n in names])
        hits = st >= cut
        if hits.any(axis=0).any():
            i = int(np.flatnonzero(hits.any(axis=0))[0])
            out = dict(first=[names[j] for j in range(len(names)) if hits[j, i]],
                       any=[names[j] for j in range(len(names)) if hits[j].any()])
    return out


def main(argv=None) -> int:
    phase = (argv or sys.argv[1:])[0]
    before = D.store_manifest()
    client, budget, ledger, cfg = load()
    target, frozen_mult = D.target_rate()
    per, skipped = build_per()
    targets = D.smap_targets()
    if phase == "tune":
        tune = [t for t in targets if t[0] in D.TUNE]                  # EVAL never loaded
        res = dict(section="docs/MODELS.md 79.13", target_rate=target, skipped=skipped,
                   channels=len(per), k_selection={})
        for arm in ("A0", "A1"):
            sc = {c: score(per[c], arm)[0] for c in per}
            cut, rt = D.solve(per, sc, target)
            res[arm] = dict(cut=cut, rate=rt,
                            tune=len(D.smap_caught({c: sc[c] >= cut for c in per}, tune, D.TUNE)))
        fz = {c: per[c]["T"]["frozen"] >= frozen_mult for c in per}
        res["frozen"] = dict(cut=frozen_mult, rate=D.rate(per, fz),
                             tune=len(D.smap_caught(fz, tune, D.TUNE)))
        res["reproduction"] = {a: bool(abs(res[a]["cut"] - D.REPRO[a][0]) <= D.CUT_RTOL * D.REPRO[a][0] + 5e-7
                                       and res[a]["tune"] == D.REPRO[a][1]) for a in ("A0", "A1", "frozen")}
        for k in K_GRID:
            sc = {c: score(per[c], "A6", k)[0] for c in per}
            cut, rt = D.solve(per, sc, target)
            got = D.smap_caught({c: sc[c] >= cut for c in per}, tune, D.TUNE)
            res["k_selection"][str(k)] = dict(cut=cut, rate=rt, tune=len(got), events=got)
        k = max(K_GRID, key=lambda k: (res["k_selection"][str(k)]["tune"], -k))
        res["k"] = k; res["A6"] = res["k_selection"][str(k)]
        res["status"] = ("FROZEN" if all(res["reproduction"].values())
                         and abs(res["A6"]["rate"] - target) / target <= D.RATE_TOL
                         else "STOP: a reproduction gate or the rate failed")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1) + "\n")
        print(f"  {res['status']}; k = {k}")
        for a in ("frozen", "A0", "A1"):
            print(f"    {a:<6} cut {res[a]['cut']:.6f} rate {100 * res[a]['rate']:.4f}% TUNE {res[a]['tune']}/19")
        for kk, v in res["k_selection"].items():
            print(f"    A6 k={kk:<4} cut {v['cut']:.6f} rate {100 * v['rate']:.4f}% TUNE {v['tune']}/19")
    else:
        committed = subprocess.run(["git", "show", f"HEAD:{FIXTURE}"], cwd=ROOT,
                                   capture_output=True).stdout
        if not committed or committed != OUT.read_bytes():
            raise SystemExit("  (!) STOP 63: a6_frozen.json is not the one committed at HEAD")
        if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip():
            raise SystemExit("  (!) STOP 63: the tracked tree is not clean")
        fz = json.loads(committed)
        k, cut = fz["k"], fz["A6"]["cut"]
        ev = [t for t in targets if t[0] in D.EVAL]
        res = dict(section="docs/MODELS.md 79.13", frozen_sha256=hashlib.sha256(committed).hexdigest(),
                   k=k, eval_read="at least the ninth full EVAL read", arms={})
        for arm, kk, c0 in (("A0", None, fz["A0"]["cut"]), ("A6", k, cut)):
            sc = {c: score(per[c], arm, kk) for c in per}
            masks = {c: sc[c][0] >= c0 for c in per}
            got = D.smap_caught(masks, ev, D.EVAL)
            att = {}
            for evn in got:
                c, lo = evn.split("["); lo = int(lo[:-1])
                hi = next(h for cc, ll, h in ev if cc == c and ll == lo)
                att[evn] = attribution(per, sc[c][1], c0, [(c, lo, hi)])
            res["arms"][arm] = dict(cut=c0, eval=len(got), of=len(ev), events=got,
                                    rate=D.rate(per, masks), attribution=att)
        a0, a6 = set(res["arms"]["A0"]["events"]), set(res["arms"]["A6"]["events"])
        res["P10_catches_A9"] = "A-9[4569]" in a6
        res["P11_holds_every_A0_catch"] = a0 <= a6
        res["lost"], res["gained"] = sorted(a0 - a6), sorted(a6 - a0)
        EVAL_OUT.write_text(json.dumps(res, indent=1) + "\n")
        for arm in ("A0", "A6"):
            a = res["arms"][arm]
            print(f"    {arm} EVAL {a['eval']}/{a['of']} rate {100 * a['rate']:.4f}%")
        print(f"  P10 (A-9 caught): {res['P10_catches_A9']}  P11 (holds every A0 catch): "
              f"{res['P11_holds_every_A0_catch']}  lost {res['lost']} gained {res['gained']}")
        print(f"  A-9 attribution: {res['arms']['A6']['attribution'].get('A-9[4569]')}")
    ops.commit(client, cfg.bucket, ledger, budget)
    print(budget.report())
    after = D.store_manifest()
    if after != before:
        print("  (!) STOP 61: the cached store moved"); return 61
    return 0


if __name__ == "__main__":
    sys.exit(main())
