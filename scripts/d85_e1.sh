#!/usr/bin/env bash
#
# D85 / docs/MODELS.md 78, E1: the whole retraining loop through the REAL deployments.
# Run as docs/MODELS.md 78.11 re-registers it while D85.1 stands: E1-dry.
#
#   PowerSim (-L: 78.3's balanced, spun-up plant) -> SampleTap -> Monitor, unchanged
#                                                  \-> hub (UDP) -> SentinelRetrain
#   SentinelRetrain: replica, 56's rule, guard band, warm start -> RetrainCandidate.bin
#   ground: the telemetry archive -> sentinel_toolkit gate -> report (CERTIFY/REFUSE)
#   human:  the approve / uplink / RELOAD_MODEL lines PRINTED, NEVER SENT (D85.1)
#   LC2:    the Monitor's emit ticks (EmitProbe) against the replica's, tick for tick
#
#     bash scripts/d85_e1.sh RUN_DIR [--dry-run-approve] [--calibrate SEQ]
#                                    [--age EMIS] [--tick-us US] [--hub PORT]
#
# (!) THIS SCRIPT SENDS NOTHING TO THE DETECTOR. D85.1: no shadow model may be swapped
# in, on the testbed or in any deployment, until a route holds C1 at zero. The step that
# used to run `sentinel_toolkit approve` for real is gone, not disabled: with
# --dry-run-approve the human's three lines are printed and `approve --dry-run` checks the
# report; without it, step 7 does not run. tests/test_e1_sends_nothing.py holds this.
#
# (!) TWO STAND-INS, STATED. The candidate reaches the ground by a file copy -- the
# retrainer's process has no downlink here -- and the ground's telemetry archive is the
# plant REPLAYED on the ground (loopsim, same seed and flags; the plant is
# deterministic). The replay is cross-checked against the channel values the ground
# actually received through fprime-gds: every downlinked value must equal its replayed
# row, bit for bit, or the run stops.
#
# (!) --tick-us is apparatus: an accelerated base tick for a host run, chosen by 78.11's
# ladder (--calibrate). Ticks, never hours: every wait is on the tap's own sequence
# number, never a wall clock, and no timing figure is produced (stop 35). Every process
# started here is stopped by the trap, and the run refuses to start if any deployment or
# GDS process is already running.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
F="${ROOT}/fprime"
SHAPE="${ROOT}/oxcaml/_build/shape-c8-p10"
# 78.11: the flying model is the committed test model, byte-identical to
# runs/d85/flying/flying.bin, so the controls cache key is unchanged.
FLYING="${ROOT}/flight/test/vectors/retrainer_ut_flying_c8_p10.bin"
FLYING_SHA256="8b22e32159455d03bccfc7c34fa2236a0bca69ddab32aa309b70326a3eb461a9"
# The gate reads the flying model's training segment from a sidecar BESIDE it
# (`sentinel_toolkit fit` writes it; D85 C10). A `.npz` may not be committed, so the gate
# reads the same bytes at their run path, sha-checked against the same pin. Found by E1.
GATE_FLYING="${ROOT}/runs/d85/flying/flying.bin"
CACHE="${ROOT}/runs/d85/controls"
PY="${ROOT}/.venv/bin/python"
LC2="${ROOT}/scripts/d85_lc2.py"
RUN="${1:?usage: bash scripts/d85_e1.sh RUN_DIR [--dry-run-approve] [--calibrate SEQ] [--age EMIS] [--tick-us US] [--hub PORT]}"
shift
AGE=""; TICK_US=100000; HUB=50100; DRY_RUN_APPROVE=0; CALIBRATE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --age) AGE="$2"; shift 2 ;;
        --tick-us) TICK_US="$2"; shift 2 ;;
        --hub) HUB="$2"; shift 2 ;;
        --dry-run-approve) DRY_RUN_APPROVE=1; shift ;;
        --calibrate) CALIBRATE="$2"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
# 78.4's constants, the same ones the arms use.
SPINUP=60000; GUARD=260; SPAN=260; HELD=9000; SCHEDULE=6550

rm -rf "${RUN}"; mkdir -p "${RUN}/ground"
# Absolute from here on: every process below starts inside ${RUN}, and a relative path
# handed to it lands in ${RUN}/${RUN} (found by E1: the GDS logs did exactly that).
RUN="$(cd "${RUN}" && pwd)"
PIDS=()
cleanup() {
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
    # fprime-gds re-parents its children when stopped; each carries this run's directory.
    pkill -f "${RUN}/gds" 2>/dev/null; sleep 1
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done
    pkill -9 -f "${RUN}/gds" 2>/dev/null
}
trap cleanup EXIT
orphans() {
    pgrep -fl 'SentinelRef|SentinelRetrain|fprime-gds|fprime_gds|fprime-cli|loopsim|ground_tools'
}
alive() {
    kill -0 "${PIDS[1]}" 2>/dev/null && kill -0 "${PIDS[2]}" 2>/dev/null
}
# Wait until both sides' heartbeats have reached tap sequence $1. The plant's own count,
# not a wall clock: a dropped cycle or a slow timer only makes this wait longer.
wait_for_seq() {
    while [ "$("${PY}" "${LC2}" lastseq "${RUN}")" -lt "$1" ]; do
        sleep 5
        alive || { echo "   (!) a deployment exited"; tail -5 "${RUN}/retrain.log" "${RUN}/ref.log"; exit 1; }
    done
}

echo "== orphan check before =="
if orphans; then
    echo "   (!) processes already running -- refusing to start"; trap - EXIT; exit 3
fi
echo "   none"

echo "== the flying model =="
for f in "${FLYING}" "${GATE_FLYING}"; do
    GOT="$(shasum -a 256 "${f}" | cut -d' ' -f1)"
    [ "${GOT}" = "${FLYING_SHA256}" ] || { echo "   (!) ${f} is ${GOT}, not ${FLYING_SHA256}"; exit 1; }
done
echo "   ${FLYING_SHA256}"

cp "${F}/build-artifacts/Darwin/SentinelRetrain/bin/SentinelRetrain" \
   "${F}/build-artifacts/Darwin/SentinelRef/bin/SentinelRef" "${RUN}/"
cp "${FLYING}" "${RUN}/SentinelModel.bin"      # the detector's model
cp "${FLYING}" "${RUN}/RetrainModel.bin"       # the same file: the retrainer's flying model
DICT="${F}/build-artifacts/Darwin/SentinelRef/dict/SentinelRefTopologyDictionary.json"
source "${F}/fprime-venv/bin/activate"

echo "== 1. fprime-gds, SentinelRetrain (hub client, -E), SentinelRef (-L -E${AGE:+ -g ${AGE}}), tick ${TICK_US} us (apparatus) =="
( cd "${RUN}" && exec fprime-gds -n -g none --dictionary "${DICT}" -l "${RUN}/gds" --log-directly \
      --file-storage-directory "${RUN}/gds/files" > gds.out 2>&1 ) &
PIDS+=($!)
sleep 6
( cd "${RUN}" && exec ./SentinelRetrain -a 127.0.0.1 -p "${HUB}" -t "${TICK_US}" -E > retrain.log 2>&1 ) &
PIDS+=($!)
sleep 2
AGE_ARGS=()
[ -n "${AGE}" ] && AGE_ARGS=(-g "${AGE}")
( cd "${RUN}" && exec ./SentinelRef -a 127.0.0.1 -p 50000 -H "${HUB}" -t "${TICK_US}" -L -E ${AGE_ARGS[@]+"${AGE_ARGS[@]}"} > ref.log 2>&1 ) &
PIDS+=($!)
sleep 5
grep -m1 "RetrainerReady" "${RUN}/retrain.log" | sed 's/^/   /' || { echo "   (!) the retrainer did not boot"; exit 1; }

if [ "${CALIBRATE}" -gt 0 ]; then
    echo "== calibration (78.11's ladder): the real pipeline to tap sequence ${CALIBRATE} =="
    wait_for_seq "${CALIBRATE}"
    cleanup; trap - EXIT
    sleep 1
    "${PY}" "${LC2}" calibrate "${RUN}"
    RUNG=$?
    echo "== orphan check after =="
    orphans || echo "   none"
    echo "CALIBRATE: tick ${TICK_US} us, rung rc=${RUNG} (0 PASS)"
    exit "${RUNG}"
fi

echo "== 2. the downlink, sampled: 60 s of PowerSim's channels through fprime-gds =="
# F' v4 names a channel by its fully qualified instance: `-c powerSim` matches nothing
# (found by E1). And the evidence is fprime-cli's OWN session log, which records every
# channel the ground client received, not its printed output, which is lossy (E1 found
# 388 ticks in the session log behind an empty printout). Run inside ${RUN}, so the
# session log lands in ${RUN}/logs and not in the repository.
( cd "${RUN}" && fprime-cli channels -c SentinelRef.powerSim -j --dictionary "${DICT}" -t 60 \
    > "${RUN}/downlink.jsonl" 2> "${RUN}/downlink.err" )
echo "   $(cat "${RUN}"/logs/fprime-cli-*/channel.log 2>/dev/null | grep -c 'SentinelRef.powerSim.SimTick') PowerSim ticks in the ground client's session log"

echo "== 3. waiting for the first candidate (the retrainer's own event) =="
while ! grep -q "Candidate written" "${RUN}/retrain.log"; do
    sleep 30
    alive || { echo "   (!) a deployment exited"; tail -5 "${RUN}/retrain.log" "${RUN}/ref.log"; exit 1; }
done
EVENT="$(grep -m1 "Candidate written" "${RUN}/retrain.log")"
echo "   $(echo "${EVENT}" | sed -E 's/.*(Candidate written)/\1/')"
cp "${RUN}/RetrainCandidate.bin" "${RUN}/ground/candidate.bin"   # stand-in for its downlink
read -r T_FIRST T_LAST <<< "$(echo "${EVENT}" | sed -E 's/.*on ticks ([0-9]+)\.\.([0-9]+).*/\1 \2/')"
FIRST=$(( T_FIRST - SPAN - GUARD )); LAST=$(( T_LAST - GUARD - 1 ))
echo "   trained on data ticks ${FIRST}..${LAST} (tap sequence numbers)"
grep -h "SamplesLost\|CallRefused" "${RUN}/retrain.log" | tail -3 | sed -E 's/^.*(SamplesLost|CallRefused)/   \1/'

echo "== 4. held out: the plant runs on until ${HELD} ticks after the last training tick =="
NEED=$(( LAST + 1 + HELD + 200 ))
wait_for_seq "${NEED}"
echo "   both sides past tap sequence ${NEED}"

echo "== 5. the ground: the archive, cross-checked against the downlink =="
AGE_REPLAY=()
[ -n "${AGE}" ] && AGE_REPLAY=("ageing=5000,8000,0.05,${AGE}")
"${SHAPE}/loopsim" out="${RUN}/ground" flying="${FLYING}" seed=1 ticks="${NEED}" spinup="${SPINUP}" \
    balance=0.60,0.92 retrain=0 ${AGE_REPLAY[@]+"${AGE_REPLAY[@]}"} > "${RUN}/ground/loopsim.out" 2>&1 \
    || { echo "   (!) the replay failed"; exit 1; }
"${PY}" - "${RUN}" "${SPINUP}" <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
run, spin = Path(sys.argv[1]), int(sys.argv[2])
names = ["SolarInput", "ChargeCurrent", "LoadCurrent", "BusVoltage", "CellTemp",
         "RadiatorTemp", "HeaterDuty", "StateOfCharge"]
trace = np.fromfile(run / "ground" / "trace.f32", dtype=np.float32).reshape(-1, 8)
import csv
pending, checked, bad = {}, 0, []
records = []
for log in sorted((run / "logs").glob("fprime-cli-*/channel.log")):
    records += [r for r in csv.reader(log.read_text().splitlines()) if len(r) >= 5]
for rec in records:
    if not rec[2].startswith("SentinelRef.powerSim."):
        continue
    name, val = rec[2].split(".")[-1], rec[4]
    if name in names:
        pending[name] = val
    elif name == "SimTick":
        row = int(val) - spin - 1
        if 0 <= row < len(trace):
            for n, v in pending.items():
                ok = np.float32(float(v)) == trace[row, names.index(n)]
                checked += 1
                if not ok:
                    bad.append((int(val), n, v, float(trace[row, names.index(n)])))
        pending = {}
print(f"   downlinked values checked against the replayed archive: {checked}, differing: {len(bad)}")
for b in bad[:5]:
    print("   (!) SimTick %d %s downlinked %r replayed %r" % b)
sys.exit(0 if checked > 0 and not bad else 1)
PYEOF
DOWNLINK_RC=$?
# E1d.2 is reported, and the run goes on: a downlink that delivers nothing must not stop
# the gate, the human's step or LC2 from being measured (found by E1).
echo "   downlink check rc=${DOWNLINK_RC} (0 every value checked equals the replay, 1 otherwise)"
cat > "${RUN}/ground/limits.json" <<'EOF'
{"names": ["SolarInput", "ChargeCurrent", "LoadCurrent", "BusVoltage", "CellTemp",
           "RadiatorTemp", "HeaterDuty", "StateOfCharge"],
 "low":  [-2.0, -10.0, 0.5, 27.0, -5.0, -35.0, 0.0, 0.30],
 "high": [300.0, 10.0, 9.5, 32.0, 40.0, 30.0, 1.0, 1.0]}
EOF

echo "== 6. the ground gate =="
DEST="RetrainApproved.bin"
RELOAD="SentinelRef.sentinelMonitor.RELOAD_MODEL"
( cd "${RUN}" && PYTHONPATH="${ROOT}/src" "${PY}" -m sentinel_toolkit gate \
    --flying "${GATE_FLYING}" --candidate "${RUN}/ground/candidate.bin" \
    --telemetry "${RUN}/ground/trace.f32" --channels 8 --first "${FIRST}" --last "${LAST}" \
    --windows "${SCHEDULE}" --budget 1 --tools "${SHAPE}/ground_tools" \
    --limits "${RUN}/ground/limits.json" --block 5400 --blocks 4 --horizon 100000 \
    --workdir "${RUN}/ground/gate" --cache "${CACHE}" --dest "${DEST}" \
    --dictionary "${DICT}" --reload-command "${RELOAD}" \
    --report "${RUN}/ground/report.json" ) | tee "${RUN}/ground/gate.out"
GATE_RC=${PIPESTATUS[0]}
echo "   gate rc=${GATE_RC} (0 CERTIFY, 1 REFUSE)"

echo "== 7. the human's step, shown and NOT SENT (D85.1) =="
APPROVE_RC="not run"
if [ "${DRY_RUN_APPROVE}" -eq 1 ]; then
    echo "   NOT SENT: python -m sentinel_toolkit approve --report ${RUN}/ground/report.json --candidate ${RUN}/ground/candidate.bin --event-log ${RUN}/ref.log"
    PYTHONPATH="${ROOT}/src" "${PY}" -c '
import sys
from sentinel_toolkit.gate import uplink_commands
for line in uplink_commands(*sys.argv[1:]):
    print("   NOT SENT: " + line)
' "${RUN}/ground/candidate.bin" "${DEST}" "${DICT}" "${RELOAD}"
    ( cd "${RUN}" && PYTHONPATH="${ROOT}/src" "${PY}" -m sentinel_toolkit approve --dry-run \
        --report "${RUN}/ground/report.json" --candidate "${RUN}/ground/candidate.bin" ) \
        > "${RUN}/approve.out" 2>&1
    APPROVE_RC=$?
    sed 's/^/   /' "${RUN}/approve.out"
    echo "   approve --dry-run rc=${APPROVE_RC} (0 on a CERTIFY report, 2 REFUSED on a REFUSE)"
else
    echo "   D85.1: no swap; step 7 not run (pass --dry-run-approve to print the lines)"
fi
if grep -q "ModelReloadAccepted" "${RUN}/ref.log"; then
    echo "   (!) ModelReloadAccepted is in the detector's log -- something was sent"; exit 1
fi
echo "   ModelReloadAccepted absent from the detector's log: nothing was sent"

echo "== 8. the rung conditions over the whole run, and LC2 =="
cleanup; trap - EXIT
sleep 1
"${PY}" "${LC2}" calibrate "${RUN}"
RUNG=$?
"${PY}" "${LC2}" lc2 "${RUN}"
LC2_RC=$?
echo "   LC2 rc=${LC2_RC} (0 HOLD, 1 FAIL, 3 NO VERDICT)"

echo "== orphan check after =="
orphans || echo "   none"
echo "E1-dry: downlink rc=${DOWNLINK_RC}, gate rc=${GATE_RC}, approve --dry-run rc=${APPROVE_RC}, rung rc=${RUNG}, LC2 rc=${LC2_RC}"
