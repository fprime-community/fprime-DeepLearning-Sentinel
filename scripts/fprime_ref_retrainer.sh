#!/usr/bin/env bash
#
# D84: the retrainer, OPT-IN, inside F's OWN reference project.
#
# scripts/fprime_ref_patch.sh proves a mission can adopt the detector with one line of
# library_locations. It is left exactly as it was, because it is also the proof that a
# mission which does NOT opt in sees no change. This script is the other half: the
# same Ref, with the retrainer switched on the way a mission would switch it on --
#
#   1. library_locations, as fprime_ref_patch.sh does, and the Monitor instanced;
#   2. the setting: SENTINEL_WITH_RETRAINER=ON and the mission's shape, in Ref's
#      settings.ini default_cmake_options;
#   3. a SECOND deployment, because the retrainer never shares the detector's process
#      (docs/DECISIONS.md D70 consequence 2). fprime/SentinelRetrain is used as it
#      stands -- a mission may equally copy it -- added to Ref's project with one
#      add_fprime_subdirectory line.
#
# and then it asserts what D84 promises: Ref still carries the Monitor and ZERO OCaml
# symbols; the second deployment carries the runtime.
#
# D85 adds the telemetry path's library half: Ref puts Sentinel.SampleTap in front of
# its Monitor, exactly as a mission would, and must STILL carry zero OCaml symbols. And
# since D85 the retrainer learns only from telemetry -- the synthetic drive is gone -- so
# run standalone it boots (RetrainerReady) and writes NO candidate. Ref has no channel
# source and no hub, so nothing reaches the tap; the live path, tap to hub to retrainer
# to gate to RELOAD_MODEL, is proven in fprime/SentinelRef (docs/MODELS.md 78, E1).
#
# The shape must already be generated: bash scripts/oxcaml_shape.sh --channels C
# --predictions P. The hub link to the detector is NOT made here -- Ref has no hub --
# and fprime/SentinelRef/Top shows the pattern. The Ref checkout is gitignored and
# every edit is reverted at the end. Re-runnable.
#
# (!) No timing figure is produced (stop 35). Status: the chosen retraining
# implementation, host-verified, not flight-qualified (D83).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${ROOT}/fprime"
REFPROJ="${PROJECT}/lib/fprime/TestDeploymentsProject"
VENV="${PROJECT}/fprime-venv"

C=8; P=10
while [ $# -gt 0 ]; do
    case "$1" in
        --channels) C="$2"; shift 2 ;;
        --predictions) P="$2"; shift 2 ;;
        *) echo "usage: bash scripts/fprime_ref_retrainer.sh [--channels C] [--predictions P]" >&2
           exit 2 ;;
    esac
done

if [ ! -d "${REFPROJ}/Ref" ]; then
    echo "No Ref deployment at ${REFPROJ}/Ref -- run scripts/fprime_setup.sh first." >&2
    exit 1
fi
SHAPE="${ROOT}/oxcaml/_build/shape-c${C}-p${P}"
if [ ! -f "${SHAPE}/cycle_complete.o" ]; then
    echo "No retrainer at ${C} channels, ${P} predictions -- run" >&2
    echo "  bash scripts/oxcaml_shape.sh --channels ${C} --predictions ${P}" >&2
    exit 1
fi
if [ ! -x "${ROOT}/.venv/bin/python" ]; then
    echo "No ${ROOT}/.venv -- the flying file is written with the ground toolkit." >&2
    exit 1
fi

export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"
RUN=""
RETRAIN_PID=""

cleanup() {
    if [ -n "${RETRAIN_PID}" ]; then kill "${RETRAIN_PID}" 2>/dev/null || true; fi
    echo "-- reverting the Ref checkout"
    git -C "${PROJECT}/lib/fprime" checkout -- \
        TestDeploymentsProject/settings.ini \
        TestDeploymentsProject/CMakeLists.txt \
        TestDeploymentsProject/Ref/Top/instances.fpp \
        TestDeploymentsProject/Ref/Top/topology.fpp \
        TestDeploymentsProject/Ref/Top/RefPackets.fppi 2>/dev/null || true
}
trap cleanup EXIT

echo "== the retrainer, opt-in, in F's own Ref project: ${C} channels, ${P} predictions =="

# 1 and 2. The library, the Monitor -- exactly fprime_ref_patch.sh's edits -- and the
#    setting that opts in.
python3 - "${REFPROJ}" "${PROJECT}" "${C}" "${P}" <<'PY'
import os, sys
from pathlib import Path
ref, project, c, p = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]

settings = ref / "settings.ini"
text = settings.read_text()
if "library_locations" not in text:
    rel = os.path.relpath(project, settings.parent)
    text = text.replace("[fprime]", f"[fprime]\nlibrary_locations: {rel}", 1)
    print(f"-- library_locations: {rel}")
marker = "    FPRIME_ENABLE_AUTOCODER_UTS=OFF\n"
assert text.count(marker) == 1, "Ref's settings.ini is not the shape this script expects"
text = text.replace(marker, marker + "    SENTINEL_WITH_RETRAINER=ON\n"
                    f"    SENTINEL_RETRAINER_CHANNELS={c}\n"
                    f"    SENTINEL_RETRAINER_PREDICTIONS={p}\n")
settings.write_text(text)
print(f"-- default_cmake_options: SENTINEL_WITH_RETRAINER=ON, channels {c}, predictions {p}")

top = ref / "Ref" / "Top"
inst = top / "instances.fpp"
s = inst.read_text()
m = "  instance comDriver: Drv.TcpClient base id 0x10025000\n"
assert s.count(m) == 1, "Ref's instances.fpp is not the shape this script expects"
inst.write_text(s.replace(m, m + "\n  instance sentinelMonitor: Sentinel.Monitor base id 0x20000000 \\\n"
                          "    queue size 10\n"
                          "\n  instance sampleTap: Sentinel.SampleTap base id 0x22000000\n"))
topo = top / "topology.fpp"
s = topo.read_text()
m = "    instance comDriver\n"
conn = "      rateGroup1Comp.RateGroupMemberOut[7] -> ComCcsds.Subtopology.aggregatorTimeout\n"
assert s.count(m) == 1 and s.count(conn) == 1, "Ref's topology.fpp is not the shape this script expects"
s = s.replace(m, m + "    instance sentinelMonitor\n    instance sampleTap\n")
s = s.replace(conn, conn + "      rateGroup1Comp.RateGroupMemberOut[8] -> sentinelMonitor.schedIn\n"
              "      sampleTap.sampleOut -> sentinelMonitor.channelsIn\n")
topo.write_text(s)
packets = top / "RefPackets.fppi"
s = packets.read_text()
m = "\n} omit {\n"
assert s.count(m) == 1, "Ref's packet set is not the shape this script expects"
packets.write_text(s.replace(m, """
  packet Sentinel id 100 group 2 {
    sentinelMonitor.Score
    sentinelMonitor.Threshold
    sentinelMonitor.ActiveMode
    sentinelMonitor.TicksSinceWarmup
    sentinelMonitor.LoadStatus
    sampleTap.Forwarded
  }
""" + m))
print("-- the Monitor instanced in Ref, on rate group 1, with the SampleTap in front of it")

# 3. The second deployment, beside Ref in the same project.
cmake = ref / "CMakeLists.txt"
s = cmake.read_text()
m = 'add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/Ref/")\n'
assert s.count(m) == 1, "Ref's CMakeLists.txt is not the shape this script expects"
cmake.write_text(s.replace(m, m + f'add_fprime_subdirectory("{project}/SentinelRetrain/")\n'))
print("-- a second deployment: fprime/SentinelRetrain, in Ref's project")
PY

echo "-- building Ref and the retrainer's deployment"
cd "${REFPROJ}/Ref"
rm -rf "${REFPROJ}/build-fprime-automatic-native" "${REFPROJ}/build-artifacts"
fprime-util generate
fprime-util build -j 8
cmake --build "${REFPROJ}/build-fprime-automatic-native" --target SentinelRetrain -j 8

REF_BIN="${REFPROJ}/build-artifacts/Darwin/Ref/bin/Ref"
RT_BIN="${REFPROJ}/build-artifacts/Darwin/SentinelRetrain/bin/SentinelRetrain"
for b in "${REF_BIN}" "${RT_BIN}"; do
    if [ ! -x "${b}" ]; then echo "FAIL: no binary at ${b}" >&2; exit 1; fi
done

# Symbol tables read into variables, never piped into grep -q (see fprime_ref_patch.sh).
ocaml_count() { nm -C "$1" 2>/dev/null | perl -ne '$c++ while /(?<![A-Za-z0-9])_?caml[A-Za-z_0-9]*/g; END{print $c+0}'; }
REF_SYMBOLS="$(nm -C "${REF_BIN}" 2>/dev/null || true)"
REF_OCAML="$(ocaml_count "${REF_BIN}")"
RT_OCAML="$(ocaml_count "${RT_BIN}")"
case "${REF_SYMBOLS}" in
    *"Sentinel::Monitor::schedIn_handler"*) echo "-- Ref carries the Monitor" ;;
    *) echo "FAIL: Ref carries no Monitor" >&2; exit 1 ;;
esac
case "${REF_SYMBOLS}" in
    *"Sentinel::Retrainer"*) echo "FAIL: Ref carries the Retrainer" >&2; exit 1 ;;
    *) ;;
esac
case "${REF_SYMBOLS}" in
    *"Sentinel::SampleTap::sampleIn_handler"*) echo "-- Ref carries the SampleTap, in front of its Monitor" ;;
    *) echo "FAIL: Ref carries no SampleTap" >&2; exit 1 ;;
esac
echo "-- OCaml symbols: Ref ${REF_OCAML}, SentinelRetrain ${RT_OCAML}"
if [ "${REF_OCAML}" != "0" ]; then echo "FAIL: the detector's process has an OCaml runtime" >&2; exit 1; fi
if [ "${RT_OCAML}" -le 100 ]; then echo "FAIL: the retrainer's deployment has no OCaml runtime" >&2; exit 1; fi

# The retrainer, run standalone: it boots at the shape asked for, and with no telemetry
# it writes no candidate (D85: the synthetic drive is gone). -t 100000 is ten ticks a
# second, so the check does not wait on a 1 Hz clock.
RUN="$(mktemp -d "${REFPROJ}/build-artifacts/retrain-run.XXXXXX")"
PYTHONPATH="${ROOT}/src" "${ROOT}/.venv/bin/python" "${ROOT}/scripts/s72_flying_file.py" \
    "${RUN}/RetrainModel.bin" --channels "${C}" --predictions "${P}"
( cd "${RUN}" && exec "${RT_BIN}" -a 127.0.0.1 -p 0 -t 100000 > retrain.log 2>&1 ) &
RETRAIN_PID=$!
READY=""
for _ in $(seq 1 20); do
    LOG="$(cat "${RUN}/retrain.log" 2>/dev/null || true)"
    case "${LOG}" in *RetrainerReady*) READY=1; break ;; esac
    sleep 1
done
sleep 3
kill "${RETRAIN_PID}" 2>/dev/null || true
wait "${RETRAIN_PID}" 2>/dev/null || true
RETRAIN_PID=""
if [ -z "${READY}" ]; then
    echo "FAIL: the retrainer did not report RetrainerReady" >&2; cat "${RUN}/retrain.log" >&2; exit 1
fi
echo "-- the retrainer booted at ${C} channels (RetrainerReady)"
if [ -e "${RUN}/RetrainCandidate.bin" ]; then
    echo "FAIL: the retrainer wrote a candidate with no telemetry" >&2; exit 1
fi
echo "-- and, with no telemetry, wrote no candidate"

echo
echo "PASS: with SENTINEL_WITH_RETRAINER=ON, F's own Ref project builds the retrainer as a"
echo "      second deployment at ${C} channels; Ref carries the Monitor behind the SampleTap"
echo "      and no OCaml runtime; the retrainer boots and, with no telemetry, trains nothing."
