#!/usr/bin/env bash
# D85 / docs/MODELS.md 78: build the loop simulator for the testbed at a generated shape.
#
#   bash scripts/loopsim_build.sh --channels 8 --predictions 10
#
# `loopsim` (fprime/SentinelRef/PowerSim/LoopSim.cpp) links the testbed's physics, the
# flight core the detector runs, and the retrainer's RetrainLoop over the cycle object
# `scripts/oxcaml_shape.sh` built for that shape. It is apparatus: a mission does not
# need it. Output: oxcaml/_build/shape-c<C>-p<P>/loopsim.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C=""; P=""
while [ $# -gt 0 ]; do
    case "$1" in
        --channels) C="${2:-}"; shift 2 ;;
        --predictions) P="${2:-}"; shift 2 ;;
        *) echo "usage: bash scripts/loopsim_build.sh --channels C --predictions P" >&2; exit 2 ;;
    esac
done
[ -n "${C}" ] && [ -n "${P}" ] || { echo "usage: bash scripts/loopsim_build.sh --channels C --predictions P" >&2; exit 2; }
B="${ROOT}/oxcaml/_build/shape-c${C}-p${P}"
if [ ! -f "${B}/cycle_complete.o" ] || [ ! -d "${B}/flightobj" ]; then
    echo "loopsim_build: no shape at ${B}; run bash scripts/oxcaml_shape.sh --channels ${C} --predictions ${P}" >&2
    exit 1
fi
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
FFLAGS="-Wold-style-cast -pedantic -Wall -Wextra -Wconversion -Wdouble-promotion -Wshadow -Werror"
cd "${B}"
clang++ -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off ${FFLAGS} -O2 \
    -I. -I"${ROOT}/flight/include" -I"${ROOT}/fprime/Sentinel/Retrainer" \
    -I"${ROOT}/fprime/SentinelRef/PowerSim" -I"$(ocamlopt -where)" \
    -c "${ROOT}/fprime/Sentinel/Retrainer/RetrainLoop.cpp" -o retrainloop.o
clang++ -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off -O2 -Wall -Wextra -Werror \
    -I"${ROOT}/fprime/SentinelRef/PowerSim" \
    -c "${ROOT}/fprime/SentinelRef/PowerSim/PowerPlant.cpp" -o powerplant.o
clang++ -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off -O2 -Wall -Wextra -Werror \
    -I. -I"${ROOT}/flight/include" -I"${ROOT}/fprime/Sentinel/Retrainer" \
    -I"${ROOT}/fprime/SentinelRef/PowerSim" \
    -c "${ROOT}/fprime/SentinelRef/PowerSim/LoopSim.cpp" -o loopsim.o
clang++ -o loopsim loopsim.o retrainloop.o powerplant.o cycle_complete.o flightobj/*.o -lm
echo "loopsim built: ${B}/loopsim"
