#!/usr/bin/env bash
#
# Build Sentinel inside F's OWN reference deployment.
#
# docs/STATUS.md's definition of done for work item 9 is "done when it builds in
# an F' Ref deployment". At v4.3.0 that deployment is not where the brief expects
# it -- Ref moved out of the repository root into TestDeploymentsProject/Ref
# (docs/MODELS.md 20.2 correction 3) -- so this script goes and finds it.
#
# It copies nothing. The Sentinel component is an F' library
# (fprime/library.cmake), so Ref consumes it exactly the way a mission would: one
# line of library_locations in settings.ini, one instance, one connection. If
# this script works, adoption works.
#
# The checkout under fprime/lib/fprime is gitignored, so the edits below are made
# to a disposable tree and reverted at the end, leaving it as `git status` clean
# as it started. Re-runnable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${ROOT}/fprime"
REFPROJ="${PROJECT}/lib/fprime/TestDeploymentsProject"
VENV="${PROJECT}/fprime-venv"

if [ ! -d "${REFPROJ}/Ref" ]; then
    echo "No Ref deployment at ${REFPROJ}/Ref -- run scripts/fprime_setup.sh first." >&2
    exit 1
fi

export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"

cleanup() {
    echo "-- reverting the Ref checkout"
    git -C "${PROJECT}/lib/fprime" checkout -- \
        TestDeploymentsProject/settings.ini \
        TestDeploymentsProject/Ref/Top/instances.fpp \
        TestDeploymentsProject/Ref/Top/topology.fpp \
        TestDeploymentsProject/Ref/Top/RefPackets.fppi 2>/dev/null || true
}
trap cleanup EXIT

echo "== Sentinel in F's own Ref deployment =="
echo "-- Ref is at ${REFPROJ}/Ref"

# 1. Consume fprime-sentinel as an F' library. One line, and it is the whole
#    integration: no copied sources, no path into our tree beyond this.
python3 - "${REFPROJ}/settings.ini" "${PROJECT}" <<'PY'
import sys
from pathlib import Path
settings, project = Path(sys.argv[1]), Path(sys.argv[2])
text = settings.read_text()
if "library_locations" not in text:
    rel = __import__("os").path.relpath(project, settings.parent)
    text = text.replace("[fprime]", f"[fprime]\nlibrary_locations: {rel}", 1)
    settings.write_text(text)
    print(f"-- library_locations: {rel}")
else:
    print("-- library_locations already set")
PY

# 2. One instance, and one connection to the 1 Hz rate group. Ref's
#    RateGroupMemberOut is used up to index 7, so Sentinel takes 8.
python3 - "${REFPROJ}/Ref/Top" <<'PY'
import sys
from pathlib import Path
top = Path(sys.argv[1])

inst = top / "instances.fpp"
s = inst.read_text()
if "sentinelMonitor" not in s:
    marker = "  instance comDriver: Drv.TcpClient base id 0x10025000\n"
    assert s.count(marker) == 1, "Ref's instances.fpp is not the shape this script expects"
    s = s.replace(marker, marker + """
  @ Sentinel, consumed from the fprime-sentinel library
  instance sentinelMonitor: Sentinel.Monitor base id 0x20000000
""")
    inst.write_text(s)
    print("-- instance added")

topo = top / "topology.fpp"
s = topo.read_text()
if "sentinelMonitor" not in s:
    marker = "    instance comDriver\n"
    assert s.count(marker) == 1, "Ref's topology.fpp is not the shape this script expects"
    s = s.replace(marker, marker + "    instance sentinelMonitor\n")
    conn = "      rateGroup1Comp.RateGroupMemberOut[7] -> ComCcsds.Subtopology.aggregatorTimeout\n"
    assert s.count(conn) == 1, "Ref's rate group block is not the shape this script expects"
    s = s.replace(conn, conn + "      rateGroup1Comp.RateGroupMemberOut[8] -> sentinelMonitor.schedIn\n")
    topo.write_text(s)
    print("-- instance and rate-group connection added")

# Ref downlinks through a telemetry packet set, which requires every channel of
# every instance to be packetized or explicitly omitted -- so adding Sentinel to
# Ref means saying what happens to its five channels. They go into a packet
# rather than the omit list: a proof that drops the telemetry on the floor is a
# weaker proof than one that carries it.
packets = top / "RefPackets.fppi"
s = packets.read_text()
if "sentinelMonitor" not in s:
    marker = "\n} omit {\n"
    assert s.count(marker) == 1, "Ref's packet set is not the shape this script expects"
    s = s.replace(marker, """
  packet Sentinel id 100 group 2 {
    sentinelMonitor.Score
    sentinelMonitor.Threshold
    sentinelMonitor.ActiveMode
    sentinelMonitor.TicksSinceWarmup
    sentinelMonitor.LoadStatus
  }
""" + marker)
    packets.write_text(s)
    print("-- telemetry packet added")
PY

# 3. Build it.
echo "-- building Ref with Sentinel in its topology"
cd "${REFPROJ}/Ref"
# settings.ini changed, so a cache generated before the library was added would
# not know about it. Removed rather than reused; it is disposable either way.
rm -rf "${REFPROJ}/build-fprime-automatic-native" "${REFPROJ}/build-artifacts"
fprime-util generate
fprime-util build -j 8

BIN="${REFPROJ}/build-artifacts/Darwin/Ref/bin/Ref"
if [ ! -x "${BIN}" ]; then
    echo "FAIL: no Ref binary at ${BIN}" >&2
    exit 1
fi
echo "-- built: ${BIN} ($(wc -c < "${BIN}" | tr -d ' ') bytes)"

# 4. Prove Sentinel is actually in it rather than merely compiled beside it. A
#    symbol rather than a string: text logging can be configured off, and then a
#    string search would report a failure that is not one.
# The symbol table is read once into a variable rather than piped into `grep -q`:
# under `set -o pipefail`, grep -q closes the pipe as soon as it matches, nm dies
# of SIGPIPE, and the pipeline reports a failure that did not happen.
SYMBOLS="$(nm -C "${BIN}" 2>/dev/null || true)"
COUNT="$(printf '%s\n' "${SYMBOLS}" | grep -c 'Sentinel::' || true)"
case "${SYMBOLS}" in
    *"Sentinel::Monitor::schedIn_handler"*)
        echo "-- Sentinel is linked into Ref: ${COUNT} symbols, including the rate-group handler"
        ;;
    *)
        echo "FAIL: Ref built but carries no Sentinel symbols" >&2
        exit 1
        ;;
esac

TEXT="$(strings -a "${BIN}" 2>/dev/null || true)"
case "${TEXT}" in
    *"Sentinel degraded to the Level 1 baseline"*)
        echo "-- and its Level 1 event text is in the binary" ;;
    *) ;;
esac

echo
echo "PASS: Sentinel builds and links inside F' v4.3.0's own Ref deployment,"
echo "      consumed as a library, with nothing copied."
