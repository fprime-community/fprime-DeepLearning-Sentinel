#!/usr/bin/env bash
# Section 62 / E5-d: the detector's tick with a retrainer process beside it.
# (!) STOP 35: every wall-clock figure is stated with its instrument.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s62"
mkdir -p "${B}"; cd "${B}"

START=$(date +%s)
echo "== Section 62 / E5-d =="
echo "   host: $(sysctl -n hw.ncpu) cores, $(( $(sysctl -n hw.memsize) / 1073741824 )) GiB"
echo "   instrument: steady_clock around Detector::step() alone; the PERIOD is"
echo "   separate and reported beside it. 42.9's T7 (worst 326 us) INCLUDED its"
echo "   trace write, so it is an upper bound on a smaller quantity."

if [ ! -x tick_rate ]; then
  clang++ -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off \
      -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -O2 \
      -I"${ROOT}/flight/include" "${SRC}/tick_rate.cpp" "${ROOT}"/flight/src/*.cpp -o tick_rate
fi

MODEL="${ROOT}/runs/testbed/testbed.bin"
run_rate () {   # hz ticks label
  ./tick_rate "${MODEL}" "$1" "$2"
}

# The retrainer, in its OWN PROCESS, looping cycles. D70 c.2's separate process.
#
# (!) STOPPED BY A SENTINEL FILE, NOT BY kill. Killing the subshell leaves the
# loop free to respawn the child and leaves the script waiting on a pid that is
# gone -- which is exactly what happened the first time this was run.
start_retrainer () {
  rm -f "${B}/stop"
  ( while [ ! -f "${B}/stop" ]; do
      "${ROOT}/oxcaml/_build/s61/s61" >/dev/null 2>&1 || true
    done ) &
  echo $!
}
stop_retrainer () {
  touch "${B}/stop"
  wait "$1" 2>/dev/null || true
  rm -f "${B}/stop"
}

echo
printf "   %-10s %6s %8s %9s %9s %9s\n" arm Hz ticks median p99 worst
declare -a ROWS
for spec in "1 60" "10 300" "100 1500"; do
  set -- $spec
  HZ=$1; N=$2
  CTRL=$(run_rate "${HZ}" "${N}")
  RP=$(start_retrainer); sleep 1
  WITH=$(run_rate "${HZ}" "${N}")
  stop_retrainer "${RP}"
  set -- ${CTRL}; printf "   %-10s %6s %8s %9s %9s %9s\n" "control" "$1" "$2" "$3" "$4" "$5"
  CW=$5
  set -- ${WITH}; printf "   %-10s %6s %8s %9s %9s %9s\n" "retrainer" "$1" "$2" "$3" "$4" "$5"
  ROWS+=("${HZ} ${CW} $5")
done

echo
echo "   HB4  paired difference, worst tick (retrainer minus control), us:"
for r in "${ROWS[@]}"; do
  set -- $r
  python3 -c "print(f'     {$1:>5.0f} Hz   control {$2:>9.3f}   with {$3:>9.3f}   delta {$3-$2:>+9.3f}')"
done

echo
echo "   HB6  C2's condition, stated as a number: 2 of $(sysctl -n hw.ncpu) cores busy"
echo "        (one detector thread, one retrainer process). C2 is UNVERIFIED and"
echo "        CONDITIONAL on core headroom at 1 Hz -- neither condition is optional."

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 62 measurement done =="
