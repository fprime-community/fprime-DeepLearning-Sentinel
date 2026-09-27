#!/usr/bin/env bash
#
# D85 / docs/MODELS.md 78, E1: the whole retraining loop through the REAL deployments.
#
#   PowerSim (-L: 78.3's balanced, spun-up plant) -> SampleTap -> Monitor, unchanged
#                                                  \-> hub (UDP) -> SentinelRetrain
#   SentinelRetrain: replica, 56's rule, guard band, warm start -> RetrainCandidate.bin
#   ground: the telemetry archive -> sentinel_toolkit gate -> report (CERTIFY/REFUSE)
#   human:  sentinel_toolkit approve -> file-uplink + RELOAD_MODEL -> the detector's log
#
#     bash scripts/d85_e1.sh RUN_DIR [--age EMIS] [--tick-us US] [--hub PORT]
#
# (!) TWO STAND-INS, STATED. The candidate reaches the ground by a file copy -- the
# retrainer's process has no downlink here -- and the ground's telemetry archive is the
# plant REPLAYED on the ground (loopsim, same seed and flags; the plant is
# deterministic). The replay is cross-checked against the channel values the ground
# actually received through fprime-gds: every downlinked value must equal its replayed
# row, bit for bit, or the run stops.
#
# (!) -t is apparatus: an accelerated base tick for a host run. Ticks, never hours: no
# timing figure is produced (stop 35). Every process started here is stopped by the trap,
# and the run refuses to start if any deployment or GDS process is already running.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
F="${ROOT}/fprime"
SHAPE="${ROOT}/oxcaml/_build/shape-c8-p10"
FLYING="${ROOT}/runs/d85/flying/flying.bin"
CACHE="${ROOT}/runs/d85/controls"
PY="${ROOT}/.venv/bin/python"
RUN="${1:?usage: bash scripts/d85_e1.sh RUN_DIR [--age EMIS] [--tick-us US] [--hub PORT]}"
shift
AGE=""; TICK_US=100000; HUB=50100
while [ $# -gt 0 ]; do
    case "$1" in
        --age) AGE="$2"; shift 2 ;;
        --tick-us) TICK_US="$2"; shift 2 ;;
        --hub) HUB="$2"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
# 78.4's constants, the same ones the arms use.
SPINUP=60000; GUARD=260; SPAN=260; HELD=9000; SCHEDULE=6550
RATE_HZ=$(( 1000000 / TICK_US ))

rm -rf "${RUN}"; mkdir -p "${RUN}/ground"
PIDS=()
cleanup() {
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
    # fprime-gds re-parents its children when stopped; each carries this run's directory.
    pkill -f "${RUN}/gds" 2>/dev/null; sleep 1
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done
    pkill -9 -f "${RUN}/gds" 2>/dev/null
}
trap cleanup EXIT

echo "== orphan check before =="
if pgrep -fl 'SentinelRef|SentinelRetrain|fprime-gds|fprime_gds|fprime-cli|loopsim|ground_tools'; then
    echo "   (!) processes already running -- refusing to start"; trap - EXIT; exit 3
fi
echo "   none"

cp "${F}/build-artifacts/Darwin/SentinelRetrain/bin/SentinelRetrain" \
   "${F}/build-artifacts/Darwin/SentinelRef/bin/SentinelRef" "${RUN}/"
cp "${FLYING}" "${RUN}/SentinelModel.bin"      # the detector's model
cp "${FLYING}" "${RUN}/RetrainModel.bin"       # the same file: the retrainer's flying model
DICT="${F}/build-artifacts/Darwin/SentinelRef/dict/SentinelRefTopologyDictionary.json"
source "${F}/fprime-venv/bin/activate"

echo "== 1. fprime-gds, SentinelRetrain (hub client), SentinelRef (-L${AGE:+ -g ${AGE}}), ${RATE_HZ} ticks a second =="
( cd "${RUN}" && exec fprime-gds -n -g none --dictionary "${DICT}" -l "${RUN}/gds" --log-directly \
      --file-storage-directory "${RUN}/gds/files" > gds.out 2>&1 ) &
PIDS+=($!)
sleep 6
( cd "${RUN}" && exec ./SentinelRetrain -a 127.0.0.1 -p "${HUB}" -t "${TICK_US}" > retrain.log 2>&1 ) &
PIDS+=($!)
sleep 2
AGE_ARGS=()
[ -n "${AGE}" ] && AGE_ARGS=(-g "${AGE}")
( cd "${RUN}" && exec ./SentinelRef -a 127.0.0.1 -p 50000 -H "${HUB}" -t "${TICK_US}" -L ${AGE_ARGS[@]+"${AGE_ARGS[@]}"} > ref.log 2>&1 ) &
PIDS+=($!)
START=$(date +%s)
sleep 5
grep -m1 "RetrainerReady" "${RUN}/retrain.log" | sed 's/^/   /' || { echo "   (!) the retrainer did not boot"; exit 1; }

echo "== 2. the downlink, sampled: 60 s of PowerSim's channels through fprime-gds =="
fprime-cli channels -c powerSim -j --dictionary "${DICT}" -t 60 > "${RUN}/downlink.jsonl" 2> "${RUN}/downlink.err"
echo "   $(wc -l < "${RUN}/downlink.jsonl" | tr -d ' ') channel updates received"

echo "== 3. waiting for the first candidate (the retrainer's own event) =="
while ! grep -q "Candidate written" "${RUN}/retrain.log"; do
    sleep 30
    if ! kill -0 "${PIDS[2]}" 2>/dev/null || ! kill -0 "${PIDS[1]}" 2>/dev/null; then
        echo "   (!) a deployment exited"; tail -5 "${RUN}/retrain.log" "${RUN}/ref.log"; exit 1
    fi
done
EVENT="$(grep -m1 "Candidate written" "${RUN}/retrain.log")"
echo "   ${EVENT}"
cp "${RUN}/RetrainCandidate.bin" "${RUN}/ground/candidate.bin"   # stand-in for its downlink
read -r T_FIRST T_LAST <<< "$(echo "${EVENT}" | sed -E 's/.*on ticks ([0-9]+)\.\.([0-9]+).*/\1 \2/')"
FIRST=$(( T_FIRST - SPAN - GUARD )); LAST=$(( T_LAST - GUARD - 1 ))
echo "   trained on data ticks ${FIRST}..${LAST} (tap sequence numbers)"
grep -h "SamplesLost\|CallRefused" "${RUN}/retrain.log" | tail -3 | sed 's/^/   /'

echo "== 4. held out: the plant runs on until ${HELD} ticks after the last training tick =="
NEED=$(( LAST + 1 + HELD + 200 ))
while [ $(( ($(date +%s) - START) * RATE_HZ )) -lt $(( NEED + NEED / 20 )) ]; do sleep 20; done

echo "== 5. the ground: the archive, cross-checked against the downlink =="
AGE_REPLAY=()
[ -n "${AGE}" ] && AGE_REPLAY=("ageing=5000,8000,0.05,${AGE}")
"${SHAPE}/loopsim" out="${RUN}/ground" flying="${FLYING}" seed=1 ticks="${NEED}" spinup="${SPINUP}" \
    balance=0.60,0.92 retrain=0 ${AGE_REPLAY[@]+"${AGE_REPLAY[@]}"} > "${RUN}/ground/loopsim.out" 2>&1 \
    || { echo "   (!) the replay failed"; exit 1; }
"${PY}" - "${RUN}" "${SPINUP}" <<'PYEOF' || exit 1
import json, sys
from pathlib import Path
import numpy as np
run, spin = Path(sys.argv[1]), int(sys.argv[2])
names = ["SolarInput", "ChargeCurrent", "LoadCurrent", "BusVoltage", "CellTemp",
         "RadiatorTemp", "HeaterDuty", "StateOfCharge"]
trace = np.fromfile(run / "ground" / "trace.f32", dtype=np.float32).reshape(-1, 8)
pending, checked, bad = {}, 0, []
for line in (run / "downlink.jsonl").read_text().splitlines():
    try:
        d = json.loads(line)
    except ValueError:
        continue
    name = str(d.get("template", {}).get("name", d.get("name", ""))).split(".")[-1]
    val = d.get("val", d.get("value"))
    if name in names:
        pending[name] = val
    elif name == "SimTick":
        row = int(val) - spin - 1
        if 0 <= row < len(trace):
            for n, v in pending.items():
                ok = np.float32(v) == trace[row, names.index(n)]
                checked += 1
                if not ok:
                    bad.append((int(val), n, v, float(trace[row, names.index(n)])))
        pending = {}
print(f"   downlinked values checked against the replayed archive: {checked}, differing: {len(bad)}")
for b in bad[:5]:
    print("   (!) SimTick %d %s downlinked %r replayed %r" % b)
sys.exit(0 if checked > 0 and not bad else 1)
PYEOF
cat > "${RUN}/ground/limits.json" <<'EOF'
{"names": ["SolarInput", "ChargeCurrent", "LoadCurrent", "BusVoltage", "CellTemp",
           "RadiatorTemp", "HeaterDuty", "StateOfCharge"],
 "low":  [-2.0, -10.0, 0.5, 27.0, -5.0, -35.0, 0.0, 0.30],
 "high": [300.0, 10.0, 9.5, 32.0, 40.0, 30.0, 1.0, 1.0]}
EOF

echo "== 6. the ground gate =="
( cd "${RUN}" && PYTHONPATH="${ROOT}/src" "${PY}" -m sentinel_toolkit gate \
    --flying "${FLYING}" --candidate "${RUN}/ground/candidate.bin" \
    --telemetry "${RUN}/ground/trace.f32" --channels 8 --first "${FIRST}" --last "${LAST}" \
    --windows "${SCHEDULE}" --budget 1 --tools "${SHAPE}/ground_tools" \
    --limits "${RUN}/ground/limits.json" --block 5400 --blocks 4 --horizon 100000 \
    --workdir "${RUN}/ground/gate" --cache "${CACHE}" --dest RetrainApproved.bin \
    --dictionary "${DICT}" --reload-command SentinelRef.sentinelMonitor.RELOAD_MODEL \
    --report "${RUN}/ground/report.json" ) | tee "${RUN}/ground/gate.out"
GATE_RC=${PIPESTATUS[0]}
echo "   gate rc=${GATE_RC} (0 CERTIFY, 1 REFUSE)"

echo "== 7. the human's one command =="
( cd "${RUN}" && PYTHONPATH="${ROOT}/src" "${PY}" -m sentinel_toolkit approve \
    --report "${RUN}/ground/report.json" --candidate "${RUN}/ground/candidate.bin" \
    --event-log "${RUN}/ref.log" --timeout 120 ) > "${RUN}/approve.out" 2>&1
APPROVE_RC=$?
sed 's/^/   /' "${RUN}/approve.out"
echo "   approve rc=${APPROVE_RC}"

echo "== the detector's own event log: reloads =="
grep -h -E "ModelReload|ModelLoaded|ModelRefused|DegradedToBaseline" "${RUN}/ref.log" | tail -5 | sed 's/^/   /' | cut -c1-200

cleanup; trap - EXIT
sleep 1
echo "== orphan check after =="
pgrep -fl 'SentinelRef|SentinelRetrain|fprime-gds|fprime_gds|fprime-cli|loopsim|ground_tools' || echo "   none"
echo "E1: gate rc=${GATE_RC}, approve rc=${APPROVE_RC}"
