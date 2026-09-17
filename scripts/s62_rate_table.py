#!/usr/bin/env python3
"""Section 62 / E5-d: the detector's tick, at a rate, with and without a
retrainer process beside it.

(!) THE ORCHESTRATION IS PYTHON AND NOT SHELL, AND THAT IS NOT A STYLE CHOICE.
The first attempt captured the retrainer loop's pid with `$!` inside a command
substitution, which runs in a subshell -- so the parent could not wait on it, the
stop file was removed before the loop noticed it, and the loops accumulated. A
measurement whose load is uncontrolled is not a measurement.

(!) TWO CLOCKS. `compute` is steady_clock around Detector::step() alone, which is
what tick_rate.cpp reports. The PERIOD is separate and printed beside it. 42.9's
T7 worst of 326 us INCLUDED its trace write, so it bounds a smaller quantity and
the comparison is not like for like.
"""
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
TICK = ROOT / "oxcaml" / "_build" / "s62" / "tick_rate"
S61 = ROOT / "oxcaml" / "_build" / "s61" / "s61"
MODEL = ROOT / "runs" / "testbed" / "testbed.bin"

ARMS = [(1, 60), (10, 300), (100, 1500)]


def measure(hz, ticks):
    out = subprocess.run([str(TICK), str(MODEL), str(hz), str(ticks)],
                         capture_output=True, text=True, check=True)
    hz_s, n_s, med, p99, worst = out.stdout.split()
    return float(med), float(p99), float(worst)


class Retrainer:
    """The retrainer in its OWN OS PROCESS (D70 consequence 2), respawned so the
    load is continuous, and stopped deterministically."""

    def __init__(self):
        self.stop = False
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen([str(S61)], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        return self

    def poke(self):
        if self.proc.poll() is not None:
            self.proc = subprocess.Popen([str(S61)], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return False


def main() -> int:
    print(f"   instrument: steady_clock around Detector::step() alone; the period "
          f"is separate")
    print(f"   {'arm':<10} {'Hz':>5} {'ticks':>7} {'median':>9} {'p99':>9} {'worst':>9}")
    rows = []
    for hz, ticks in ARMS:
        c_med, c_p99, c_worst = measure(hz, ticks)
        print(f"   {'control':<10} {hz:>5} {ticks:>7} {c_med:>9.3f} {c_p99:>9.3f} "
              f"{c_worst:>9.3f}", flush=True)
        with Retrainer() as r:
            # one respawn check per second of the arm, so the load does not lapse
            deadline = time.time() + (ticks / hz) + 2.0
            import threading
            keep = [True]

            def pump():
                while keep[0] and time.time() < deadline:
                    r.poke()
                    time.sleep(0.2)
            th = threading.Thread(target=pump, daemon=True)
            th.start()
            w_med, w_p99, w_worst = measure(hz, ticks)
            keep[0] = False
            th.join(timeout=2)
        print(f"   {'retrainer':<10} {hz:>5} {ticks:>7} {w_med:>9.3f} {w_p99:>9.3f} "
              f"{w_worst:>9.3f}", flush=True)
        # (!) A SECOND CONTROL, AFTER. The first attempt ran control-then-retrainer
        # and the retrainer arm came out FASTER at every rate -- which load cannot
        # do. The arms differed in ORDER as well as in load: the first is cold, and
        # this host ramps its clock. A-B-A makes the order effect visible instead of
        # letting it be read as an effect of the retrainer. 42.9.1's rule, applied
        # to a confound it did not have.
        c2_med, c2_p99, c2_worst = measure(hz, ticks)
        print(f"   {'control-2':<10} {hz:>5} {ticks:>7} {c2_med:>9.3f} {c2_p99:>9.3f} "
              f"{c2_worst:>9.3f}", flush=True)
        rows.append((hz, c_worst, w_worst, c2_worst, c_med, w_med, c2_med))

    print("\n   HB4  paired difference, 42.9.1's method -- same detector, same seed,")
    print("        same drive, differing only in whether a retrainer process is alive:")
    for hz, cw, ww, c2w, cm, wm, c2m in rows:
        base = min(cw, c2w)
        print(f"     {hz:>5} Hz   controls {cw:>9.3f} / {c2w:>9.3f}   with retrainer "
              f"{ww:>9.3f}   vs the better control {ww - base:>+9.3f} us")
    print("\n   HB3  against 42.9's T7 worst of 326 us (which included its trace write):")
    for hz, cw, ww, c2w, cm, wm, c2m in rows:
        verdict = "inside" if ww <= 326.0 else ("<=1000" if ww <= 1000.0 else "OVER 1000")
        period_us = 1e6 / hz
        print(f"     {hz:>5} Hz   worst {ww:>9.3f} us   {verdict:>9}   "
              f"period {period_us:>10.0f} us   {100.0 * ww / period_us:>6.3f}% of it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
