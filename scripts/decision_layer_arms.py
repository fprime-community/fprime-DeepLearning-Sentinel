#!/usr/bin/env python3
"""Work item 9.21: the arms of `docs/MODELS.md` 38, and the stride item.

**Pre-registered in `docs/MODELS.md` 38 before this ran.** One lever per arm; the
multiplier is the rate-matching dial and never a second lever (38.3). Every arm is
brought to the frozen arm's own pooled nominal rate -- **0.6820%**, not 0.6838%,
which is the commanded arm's (26.18). Parameters are selected on TUNE and reported
on EVAL, the channel-disjoint split committed at 37.8.

**The forecast is computed once**; every arm scores the resident residual arrays, so
the stack costs one bundle load and no refit. The weight store is asserted unmoved.

Mechanisms, each transcribed from a source read at first hand (`docs/RESEARCH.md`):
  Arm 5   Siffer, KDD 2017: z_q = t + (sigma/gamma)*[(q n / N_t)^(-gamma) - 1] (eq.1);
          Grimshaw x* solves u(x)v(x)=1 on (-1/Y_M, inf), gamma=v(x*)-1, sigma=gamma/x*.
  Arm 4b  Basseville and Nikiforov 1993 ch.2 s.2.2: g_k=(g_{k-1}+s_k)^+ (2.2.9),
          alarm at g_k >= h (2.2.10).
  Arm 6   Gibbs and Candes 2021: alpha_{t+1} = alpha_t + gamma*(alpha - err_t).
"""
from __future__ import annotations

import argparse, ast, csv, importlib.util, io, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="Degrees of freedom")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sentinel_data import config as C                                   # noqa: E402
from sentinel_eval import ops                                           # noqa: E402
from sentinel_models import telemanom                                   # noqa: E402

def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
    return m
SR = _load("smap_rungs", "scripts/smap_rungs.py")
F38 = _load("f38", "scripts/smap_forensics_38.py")

TUNE = {"A-3","A-7","A-8","C-1","E-11","F-7","G-7","M-3","M-4","P-1","P-7","T-8"}
EVAL = {"A-2","A-4","A-9","D-16","E-1","E-10","E-12","E-13","F-3","F-8","T-1","T-12","T-13"}
P_GRID = [0.0, 0.02, 0.05, 0.08, 0.10, 0.13, 0.16, 0.20]
Q_GRID = [0.50, 0.75, 0.90, 0.95, 0.99]
GAMMA_GRID = [0.001, 0.005, 0.02]
PEAK_CAP = 512
CUSUM_K = 3.0
ACI_MIN_BUFFER = 1000
TARGET_RATE = [None]


def trailing_stats(x, span):
    x = np.asarray(x, dtype=np.float64); n = len(x)
    c1 = np.concatenate([[0.0], np.cumsum(x)]); c2 = np.concatenate([[0.0], np.cumsum(x*x)])
    lo = np.maximum(np.arange(n) + 1 - span, 0); cnt = np.arange(n) + 1 - lo
    s1 = c1[np.arange(n)+1] - c1[lo]; s2 = c2[np.arange(n)+1] - c2[lo]
    mu = s1/cnt; return mu, np.sqrt(np.maximum(s2/cnt - mu*mu, 0.0))

def zstat(x, span):
    mu, sd = trailing_stats(x, span)
    return (np.asarray(x, dtype=np.float64) - mu) / np.maximum(sd, 1e-12)

def raw_runs(mask):
    m = np.asarray(mask, dtype=np.int8); d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

def grimshaw(Y, ncand=10):
    Y = np.asarray(Y, float); Nt = len(Y)
    if Nt < 20: return None
    Ym = Y.max()
    if Ym <= 0: return None
    u = lambda x: np.mean(1.0/(1.0 + x*Y))
    v = lambda x: 1.0 + np.mean(np.log(np.maximum(1.0 + x*Y, 1e-300)))
    f = lambda x: u(x)*v(x) - 1.0
    lo, hi = -1.0/Ym + 1e-9, 10.0/max(float(np.mean(Y)), 1e-12)
    grid = np.concatenate([np.linspace(lo, -1e-8, ncand), np.linspace(1e-8, hi, ncand)])
    vals = [f(x) for x in grid]; best = None
    for i in range(len(grid)-1):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i+1]): continue
        if vals[i]*vals[i+1] < 0:
            a, b = grid[i], grid[i+1]
            for _ in range(50):
                m = 0.5*(a+b)
                if f(a)*f(m) <= 0: b = m
                else: a = m
            x = 0.5*(a+b)
            if abs(x) < 1e-12: continue
            g = v(x) - 1.0; s = g/x
            if s <= 0 or not np.isfinite(g): continue
            z = 1.0 + g*Y/s
            if np.any(z <= 0): continue
            ll = -Nt*np.log(s) - (1.0 + 1.0/g)*np.sum(np.log(z))
            if np.isfinite(ll) and (best is None or ll > best[0]): best = (ll, g, s)
    return None if best is None else (best[1], best[2])

def pot_fit(nominal_vals, tq=0.98, cap=None):
    v = np.asarray(nominal_vals, float); v = v[np.isfinite(v)]
    if v.size < 500: return None
    t = float(np.quantile(v, tq)); Y = v[v > t] - t
    if cap is not None and len(Y) > cap: Y = Y[-cap:]
    fit = grimshaw(Y)
    if fit is None: return None
    g, s = fit
    return dict(gamma=g, sigma=s, t=t, n=len(v), Nt=len(Y))

def pot_z(fit, q):
    if fit is None or abs(fit["gamma"]) < 1e-12: return np.inf
    return fit["t"] + (fit["sigma"]/fit["gamma"]) * ((q*fit["n"]/fit["Nt"])**(-fit["gamma"]) - 1.0)

def pooled_rate(masks, per):
    a = sum(int((masks[c] & per[c]["nominal"]).sum()) for c in masks)
    n = sum(int(per[c]["nominal"].sum()) for c in masks)
    return a/n if n else float("nan")

def solve_threshold(per, key, target, lo=None, hi=None, steps=None, floor=None):
    """The threshold is the (1 - target) quantile of the POOLED NOMINAL scores.

    Exact by construction, and immune to the step-shaped scores
    `telemanom.channel_ratios` produces -- its output is below 1.0 for a suppressed
    step and at or above 1.0 for a surviving one, with nothing in between, so a
    bisection lands on the discontinuity and reports whatever rate sits there. The
    2-channel smoke showed exactly that: the dial pinned at 1.0000 and 0.4990 for
    every `p`. A pooled quantile asks the question the matched-rate rule actually
    asks -- what cut admits this fraction of nominal time -- and answers it directly.
    """
    pool = np.concatenate([per[c]["S"][key][per[c]["nominal"]] for c in per])
    pool = pool[np.isfinite(pool)]
    if not pool.size: return np.inf, float("nan"), {c: np.zeros(per[c]["steps"], bool) for c in per}
    # Candidate cuts: the pooled quantile, and the distinct values around it. Ties
    # matter -- `channel_ratios` forces every surviving step to exactly 1.0, so the
    # achievable rates are DISCRETE and a target between two of them is unreachable.
    q0 = float(np.quantile(pool, 1.0 - target))
    if floor is not None: q0 = max(q0, floor)
    uniq = np.unique(pool)
    if floor is not None: uniq = uniq[uniq >= floor]
    if not uniq.size: uniq = np.array([floor])
    i = int(np.searchsorted(uniq, q0))
    cands = uniq[max(0, i - 3): i + 4]
    cands = np.unique(np.concatenate([cands, [q0, np.nextafter(q0, np.inf)]]))
    best = None
    for thr in cands:
        masks = {c: per[c]["S"][key] >= thr for c in per}
        r = pooled_rate(masks, per)
        if r != r: continue
        # prefer at-or-below target, then closest
        key_ = (0 if r <= target * 1.02 else 1, abs(r - target))
        if best is None or key_ < best[0]: best = (key_, float(thr), r, masks)
    return best[1], best[2], best[3]

def caught(masks, targets, which):
    return [f"{c}[{lo}]" for c, lo, hi in targets
            if c in which and c in masks and masks[c][lo:hi+1].any()]

def report(name, masks, per, targets, note=""):
    r = pooled_rate(masks, per)
    t_, e_ = caught(masks, targets, TUNE), caught(masks, targets, EVAL)
    a_ = caught(masks, targets, TUNE | EVAL)
    row = dict(arm=name, nominal_rate=r, tune=len(t_), tune_n=19, eval=len(e_), eval_n=19,
               all38=len(a_), eval_events=e_, note=note)
    off = abs(r - TARGET_RATE[0]) / TARGET_RATE[0] if TARGET_RATE[0] else 0.0
    flag = "" if off <= 0.02 else f"  (!) RATE NOT MATCHED, {100*off:.0f}% off target"
    row["rate_matched"] = off <= 0.02
    print(f"  {name:<26} rate {100*r:7.4f}%   TUNE {len(t_):2}/19   EVAL {len(e_):2}/19   "
          f"all {len(a_):2}/38  {note}{flag}")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visibility", required=True)
    ap.add_argument("--stage4", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="arms")
    a = ap.parse_args(argv)

    vis = json.loads(Path(a.visibility).read_text()); st4 = json.loads(Path(a.stage4).read_text())
    frozen_mult = float(st4["arms"]["gru"]["multiplier"])
    target = float(st4["arms"]["gru"]["nominal_rate"])
    TARGET_RATE[0] = target
    print(f"  frozen arm: mult {frozen_mult:.6f}, matched at {100*target:.4f}% "
          f"(0.6838% is the COMMANDED arm's, 26.18)")
    targets = [(v["channel"], v["start"], v["end"]) for v in vis["sequences"]
               if v["class"] == "contextual" and v["in_range"]]

    cfg_r2 = C.load_r2_config(); client, budget = ops.connect(cfg_r2)
    ledger = ops.load(client, cfg_r2.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    man = json.loads(SR.fetch(client, cfg_r2.bucket, SR.MANIFEST_KEY))
    labels = list(csv.DictReader(io.StringIO(SR.fetch(client, cfg_r2.bucket, man["labels_key"]).decode())))
    spans, seen = {}, set()
    for r in labels:
        cid = r["chan_id"]
        if cid in seen: continue
        seen.add(cid)
        spans[cid] = list(zip(ast.literal_eval(r["anomaly_sequences"]),
                              [x.strip() for x in r["class"].strip("[]").split(",")]))
    chans = list(man["channels"])
    if a.limit: chans = chans[: a.limit]
    store = SR.D.WEIGHT_STORE
    before = sum(1 for q in store.iterdir() if q.suffix == ".npz")

    per, skipped = {}, {}
    t0 = time.time()
    for i, ch in enumerate(chans, 1):
        cid = ch["channel_id"]; keys = {o["split"]: o["key"] for o in ch["objects"]}
        train = np.load(io.BytesIO(SR.fetch(client, cfg_r2.bucket, keys["train"])))
        test = np.load(io.BytesIO(SR.fetch(client, cfg_r2.bucket, keys["test"])))
        try:
            det = SR.registry.build("gru-telemanom"); det.config = SR.proportional_config(len(test))
            v_tr, v_te = train[:, :1].astype(np.float32), test[:, :1].astype(np.float32)
            det.fit(v_tr, np.ones(len(v_tr), dtype=bool), SR.ctx_for(cid, len(v_tr)))
            s_ctx = SR.ctx_for(cid, len(v_te))
            e_s = np.asarray(det._smoothed_errors(v_te, s_ctx), dtype=np.float64)[:, 0]
            hz = F38.horizon_predictions(det, v_te[:, 0])
        except Exception as exc:
            skipped[cid] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            print(f"    [{i}/{len(chans)}] {cid}: SKIP -- {skipped[cid][:60]}"); continue
        x = v_te[:, 0].astype(np.float64)
        anom = np.zeros(len(x), dtype=bool)
        for (lo_, hi_), _c in spans.get(cid, []): anom[lo_:hi_+1] = True
        nominal = ~anom; nominal[: int(det.warmup_steps)] = False
        cfg = det.config; span = cfg.error_window
        zr = zstat(e_s, span); zd = zstat(np.abs(np.diff(x, prepend=x[0])), span)
        zg = zstat(np.nan_to_num(np.nanstd(hz, axis=1), nan=0.0), span)
        # B&N's CUSUM needs E[s_k] < 0 in control. The trailing-standardised residual
        # is right-skewed, so a slack of 0.5 left positive drift and h solved to 67,026
        # with nothing firing (38.15). CUSUM_K = 3.0 sigma is fixed a priori and
        # dimensionless: no label, no data-unit constant (D55).
        g = np.zeros(len(zr)); acc = 0.0
        for k, sk in enumerate(zr - CUSUM_K):
            acc = max(0.0, acc + sk); g[k] = acc
        per[cid] = dict(e_s=e_s, nominal=nominal, cfg=cfg, steps=len(x), zr=zr,
                        S={"arm2": np.maximum.reduce([zr, zd, zg]),
                           "arm2_nodis": np.maximum(zr, zd), "arm4b": g},
                        pot=pot_fit(e_s[nominal], cap=None),
                        pot_cap=pot_fit(e_s[nominal], cap=PEAK_CAP))
        SR.D.clear_caches()
        if i % 10 == 0 or i == len(chans): print(f"    [{i}/{len(chans)}] {cid}  {time.time()-t0:6.1f}s")
    after = sum(1 for q in store.iterdir() if q.suffix == ".npz")
    assert after == before, f"weight store moved {before} -> {after}; this run must not fit"
    load_wall = time.time() - t0
    print(f"  loaded {len(per)} channels in {load_wall:.1f}s; skipped {list(skipped)}")

    rows, detail = [], {}

    # ---- frozen baseline, rebuilt (reproduction gate) --------------------
    base = {}
    for cid, d in per.items():
        out = telemanom.channel_ratios(d["e_s"], d["cfg"])
        d["S"]["frozen"] = np.asarray(out, float)
        base[cid] = d["S"]["frozen"] >= frozen_mult
    rows.append(report("frozen (stage 4)", base, per, targets, "reproduction gate"))
    detail["frozen_eval"] = caught(base, targets, EVAL)

    # ---- Arm 1: pruning swept, multiplier re-solved -----------------------
    arm1 = {}
    for p in P_GRID:
        cfg_p = None
        for cid, d in per.items():
            cfg_p = telemanom.Config(error_window=d["cfg"].error_window, stride=d["cfg"].stride,
                                     pruning_p=p)
            d["S"][f"p{p}"] = np.asarray(telemanom.channel_ratios(d["e_s"], cfg_p), float)
        # (!) The dial may only TIGHTEN. `channel_ratios` maps a suppressed step to
        # raw/(1+raw) < 1 and a surviving one to max(raw, 1) >= 1, so ANY cut below
        # 1.0 re-admits steps pruning deleted -- which is not rate-matching, it is a
        # different rule. 38.15 recorded that defect; the dial is now floored at 1.0.
        got = solve_threshold(per, f"p{p}", target, floor=1.0)
        arm1[p] = got
        rows.append(report(f"arm1 p={p:.2f}", got[2], per, targets, f"mult {got[0]:.4f} (dial >= 1)"))
    best_p = max(P_GRID, key=lambda p: len(caught(arm1[p][2], targets, TUNE)))
    detail["arm1_selected_p"] = best_p
    print(f"  ARM 1: p selected on TUNE = {best_p}")

    # ---- Arm 2 -------------------------------------------------------------
    for key, label in (("arm2", "arm2 fused"), ("arm2_nodis", "arm2 ablation (no disagree)")):
        got = solve_threshold(per, key, target, 0.0, 60.0)
        rows.append(report(label, got[2], per, targets, f"z {got[0]:.3f}"))
        detail[key] = dict(threshold=got[0], eval=caught(got[2], targets, EVAL))

    # ---- Arm 4a: run-length persistence -----------------------------------
    # Pruning is REPLACED by a run-length test on RAW, UNDILATED exceedance runs
    # (38.6: `sequences_at` returns runs already dilated by +/-99 and merged, so a
    # length taken from it is very nearly constant). Reference runs come from the
    # trailing cells, so the rule is causal, per channel and label-free.
    runlen = {}
    for cid, d in per.items():
        cfgc, e = d["cfg"], d["e_s"]
        span, stride = cfgc.error_window, max(1, cfgc.stride)
        score = np.zeros(d["steps"])
        for seg_lo in range(0, d["steps"], stride):
            seg_hi = min(seg_lo + stride, d["steps"])
            ref_lo = max(0, seg_lo - span)
            win = e[ref_lo:seg_hi]
            if win.size == 0: continue
            eps, _ = telemanom.dynamic_threshold(np.asarray(win, dtype=np.float32), cfgc)
            ref = e[ref_lo:seg_lo]
            rl = np.array([hi-lo for lo, hi in raw_runs(ref >= eps)]) if ref.size else np.array([])
            # Score a candidate run by its length RELATIVE to the reference cells' own
            # run lengths. Finite and unbounded above, so a cut exists; 38.15 recorded
            # that a percentile in [0,1] left the search with nothing to cut at.
            scale = float(np.median(rl)) if rl.size >= 5 else 1.0
            for lo, hi in raw_runs(e[seg_lo:seg_hi] >= eps):
                score[seg_lo+lo:seg_lo+hi] = (hi - lo) / max(scale, 1.0)
        runlen[cid] = score
        d["S"]["arm4a"] = score
    got = solve_threshold(per, "arm4a", target)
    rows.append(report("arm4a run-length", got[2], per, targets, f"q {got[0]:.4f}"))
    detail["arm4a"] = dict(q=got[0], eval=caught(got[2], targets, EVAL))
    lens_all = np.concatenate([np.array([hi-lo for lo, hi in raw_runs(
        per[c]["e_s"] >= np.quantile(per[c]["e_s"][per[c]["nominal"]], 1.0-target))])
        for c in per if per[c]["nominal"].any()] or [np.array([1])])
    detail["arm4a_runlen_iqr_over_median"] = float(
        (np.percentile(lens_all, 75) - np.percentile(lens_all, 25)) /
        max(np.median(lens_all), 1e-9))

    # ---- Arm 4b: CUSUM -----------------------------------------------------
    got = solve_threshold(per, "arm4b", target, 0.0, 5000.0)
    rows.append(report("arm4b CUSUM", got[2], per, targets, f"h {got[0]:.2f}"))
    detail["arm4b"] = dict(h=got[0], eval=caught(got[2], targets, EVAL))

    # ---- Arm 5: POT, unbounded and bounded --------------------------------
    for tag, key in (("arm5 POT (unbounded)", "pot"), ("arm5 POT (bounded)", "pot_cap")):
        lo_q, hi_q = 1e-9, 0.2
        best = None
        for _ in range(44):
            q = (lo_q*hi_q) ** 0.5
            masks = {c: per[c]["e_s"] >= pot_z(per[c][key], q) for c in per}
            r = pooled_rate(masks, per); best = (q, r, masks)
            if r > target: hi_q = q
            else: lo_q = q
            if r == r and abs(r - target) <= 0.02*target: break
        rows.append(report(tag, best[2], per, targets, f"q {best[0]:.3e}"))
        detail[key] = dict(q=best[0], eval=caught(best[2], targets, EVAL))
    z5 = {c: pot_z(per[c]["pot"], detail["pot"]["q"]) for c in per}
    fits = [per[c]["pot"] for c in per if per[c]["pot"]]
    detail["pot_gamma_ok"] = sum(1 for f in fits if f["gamma"] > -0.5)
    detail["pot_fitted"] = len(fits); detail["pot_channels"] = len(per)

    # ---- Arm 6: ACI --------------------------------------------------------
    def aci(gam, alpha0):
        """Gibbs and Candes: alpha_{t+1} = alpha_t + gamma*(alpha0 - err_t).

        The quantile is refreshed once per `stride`, not per sample -- the cadence
        the frozen arm already runs at, and O(T/stride) rather than O(T) quantiles.
        """
        masks, traj, ratio = {}, [], {}
        for cid, d in per.items():
            # PHASE2 5b: a 0.1%-tail quantile needs ~1,000 samples to be an interior
            # one. error_window is 0.05*len(test), a few dozen on short channels, and
            # 38.15 measured the consequence -- 24.77% realised against a 0.68% target.
            span = max(d["cfg"].error_window, ACI_MIN_BUFFER)
            stride = max(1, d["cfg"].stride)
            e, alpha = d["e_s"], alpha0
            m = np.zeros(d["steps"], dtype=bool); a_hist = []
            rr = np.zeros(d["steps"])
            for seg_lo in range(0, d["steps"], stride):
                seg_hi = min(seg_lo + stride, d["steps"])
                buf = e[max(0, seg_lo - span):seg_lo]
                thr = (float(np.quantile(buf, 1.0 - np.clip(alpha, 1e-6, 0.5)))
                       if buf.size >= ACI_MIN_BUFFER else np.inf)
                hit = e[seg_lo:seg_hi] >= thr
                m[seg_lo:seg_hi] = hit
                rr[seg_lo:seg_hi] = (e[seg_lo:seg_hi] / thr) if np.isfinite(thr) and thr > 0 else 0.0
                err = float(hit.mean()) if hit.size else 0.0
                alpha = float(np.clip(alpha + gam * (alpha0 - err), 1e-6, 0.5))
                a_hist.append(alpha)
            masks[cid] = m; traj.append(np.array(a_hist)); ratio[cid] = rr
        return masks, traj, ratio

    for gam in GAMMA_GRID:
        lo_a, hi_a, best = 1e-6, 0.2, None
        for _ in range(24):                       # rate-match by bisecting alpha0 (38.3)
            a0 = (lo_a * hi_a) ** 0.5
            masks, traj, ratio = aci(gam, a0)
            r = pooled_rate(masks, per); best = (a0, r, masks, traj, ratio)
            if r > target: hi_a = a0
            else: lo_a = a0
            if r == r and abs(r - target) <= 0.02 * target: break
        a0, r, masks, traj, ratio = best
        alla = np.concatenate([x for x in traj if x.size]) if traj else np.array([a0])
        iqr = float(np.percentile(alla, 75) - np.percentile(alla, 25))
        rows.append(report(f"arm6 ACI g={gam}", masks, per, targets,
                           f"alpha0 {a0:.3e} iqr/alpha {iqr/max(a0,1e-12):.2f}"))
        if gam == GAMMA_GRID[0]:
            a6_score = dict(ratio)      # continuous, so it scales with u
        detail[f"arm6_g{gam}"] = dict(alpha0=a0, eval=caught(masks, targets, EVAL),
                                      alpha_iqr_over_alpha=iqr / max(a0, 1e-12))

    # ---- Arm 7: the union, budget FITTED JOINTLY (38.3) --------------------
    # oscfar.py:80-84 records this project making the other half of this error:
    # calibrating each term to the target on its own "is wrong, and it is kept
    # because it is what ran". Here a single shared level `u` is bisected until the
    # UNION admits the target, exactly as oscfar._fit_jointly does.
    members = ["arm2_nodis", "arm4a", "arm4b", "arm5", "arm6"]
    for cid, d in per.items():
        d["S"]["arm5"] = np.where(np.isfinite(z5.get(cid, np.inf)) & (z5.get(cid, np.inf) > 0),
                                  d["e_s"] / max(z5.get(cid, np.inf), 1e-30), 0.0) \
            if cid in z5 else np.zeros(d["steps"])
        d["S"]["arm6"] = a6_score.get(cid, np.zeros(d["steps"]))

    def union_at(u):
        """Each member cut at its own pooled-nominal (1-u) quantile, then OR'd."""
        cuts, masks = {}, {}
        for k in members:
            pool = np.concatenate([per[c]["S"][k][per[c]["nominal"]] for c in per])
            pool = pool[np.isfinite(pool)]
            cuts[k] = float(np.quantile(pool, 1.0 - u)) if pool.size else np.inf
        for c in per:
            m = np.zeros(per[c]["steps"], dtype=bool)
            for k in members: m |= per[c]["S"][k] >= cuts[k]
            masks[c] = m
        return cuts, masks

    lo_u, hi_u, best7 = 1e-8, 0.05, None
    for _ in range(40):
        u = (lo_u * hi_u) ** 0.5
        cuts, masks = union_at(u)
        r = pooled_rate(masks, per); best7 = (u, r, masks, cuts)
        if r > target: hi_u = u
        else: lo_u = u
        if r == r and abs(r - target) <= 0.02 * target: break
    u, r7, masks7, cuts7 = best7
    rows.append(report("arm7 union (joint fit)", masks7, per, targets, f"shared level u {u:.3e}"))

    # per-event attribution: which member fired on each caught event
    attribution = {}
    for cid, lo, hi in targets:
        if cid not in per or not masks7[cid][lo:hi+1].any(): continue
        who = [k for k in members if (per[cid]["S"][k][lo:hi+1] >= cuts7[k]).any()]
        attribution[f"{cid}[{lo}]"] = who
    from collections import Counter
    share = Counter(k for v in attribution.values() for k in v)
    solo = Counter(v[0] for v in attribution.values() if len(v) == 1)
    detail["arm7"] = dict(shared_level=u, cuts=cuts7, attribution=attribution,
                          eval=caught(masks7, targets, EVAL),
                          member_share={k: share.get(k, 0) for k in members},
                          sole_credit={k: solo.get(k, 0) for k in members})
    n7 = max(1, len(attribution))
    top = max(share.values()) / n7 if share else 0.0
    print(f"    attribution over {len(attribution)} caught events: "
          + ", ".join(f"{k} {share.get(k,0)}" for k in members))
    print(f"    largest single-member share {100*top:.0f}%  (P7.2 band: under 70%)")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = dict(generated_utc=stamp, pre_registration="docs/MODELS.md 38",
               matched_rate=target, frozen_multiplier=frozen_mult,
               tune=sorted(TUNE), eval=sorted(EVAL), rows=rows, detail=detail,
               channels=len(per), skipped=skipped, load_wall_s=round(load_wall, 1),
               weight_store=dict(before=before, after=after), operations=budget.as_dict())
    path = ROOT / "runs" / "smap-msl" / "_forensics" / f"{stamp}-{a.tag}.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"\n  artifact {path.relative_to(ROOT)}\n{budget.report()}")
    try: ops.commit(client, cfg_r2.bucket, ledger, budget)
    except Exception as exc: print(f"  (!) LEDGER COMMIT FAILED -- artifact is written: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
