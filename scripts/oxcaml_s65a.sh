#!/usr/bin/env bash
# Section 65.10 / R6: the domain lock, reproduced before the fix is believed.
#
# docs/MODELS.md 65.6 recorded "Fatal error: no domain lock held" and recorded
# the fix. Nothing reproduced the failure. This builds a probe that does, and
# runs it BOTH ways: the cross-thread call must abort, the same-thread call
# must succeed. A guard that has only ever passed is not known to work.
#
# (!) STOP 41: nothing here links ${OX_OBJ} into a detector binary.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s65a"
# D82: the checked accessor, built ONCE here so every -I "${SRC}" sees the same
# one. (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc
# on its include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL
# COMPILER ERROR ("Cannot create parameter Acc.next_depth ... Misc.Fatal_error"),
# not a diagnostic. Recorded in D82 as a rule-19 finding.
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
rm -rf "${B}"; mkdir -p "${B}"; cd "${B}"
cp "${SRC}/deep_f32.ml" "${SRC}/cycle_c.ml" "${SRC}/cycle_stubs.c" "${SRC}/sentinel_cycle.h" .

# F's validation set, cmake/flags.cmake, the same set 61 built the boundary at.
FFLAGS="-Wold-style-cast -pedantic -Wall -Wextra -Wconversion -Wdouble-promotion -Wshadow -Werror"

echo "== Section 65.10 / R6: the domain lock, both directions =="
echo "   ocamlopt -I "${SRC}" $(ocamlopt -version)"

set -e
ocamlopt -I "${SRC}" -output-complete-obj -O3 -o cycle_ml.o -I . "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" deep_f32.ml cycle_c.ml
clang++ -std=c++14 -fno-exceptions -fno-rtti ${FFLAGS} -O2 -I. -I"$(ocamlopt -where)" \
    -c "${SRC}/domain_lock_probe.cpp" -o probe.o
clang -std=c11 -pedantic -Wall -Wextra -Wconversion -Wshadow -Werror -O2 \
    -I. -I"$(ocamlopt -where)" -c cycle_stubs.c -o cycle_stubs.o
clang++ -o domain_lock_probe probe.o cycle_stubs.o cycle_ml.o -lm -lpthread
echo "   probe built clean at -Werror"
set +e

echo
echo "-- ARM 1: boot on the main thread, call from a SECOND thread"
./domain_lock_probe cross > cross.out 2> cross.err
CROSS_RC=$?
sed 's/^/     /' cross.out; sed 's/^/     /' cross.err
echo "   exit ${CROSS_RC}"

echo
echo "-- ARM 2: boot on the main thread, call on the SAME thread"
./domain_lock_probe same > same.out 2> same.err
SAME_RC=$?
sed 's/^/     /' same.out; sed 's/^/     /' same.err
echo "   exit ${SAME_RC}"

echo
FAIL=0
if [ ${CROSS_RC} -eq 0 ] || [ ${CROSS_RC} -eq 4 ]; then
    echo "   DL1 FAIL: the cross-thread call did not abort (exit ${CROSS_RC})"
    FAIL=1
elif grep -qi "no domain lock held" cross.err; then
    echo "   DL1 HOLD: cross-thread died with 'no domain lock held', exit ${CROSS_RC}"
else
    echo "   DL1 NO VERDICT: cross-thread died (exit ${CROSS_RC}) without that message"
    FAIL=1
fi
if [ ${SAME_RC} -eq 0 ]; then
    echo "   DL2 HOLD: same-thread call succeeded, exit 0"
else
    echo "   DL2 FAIL: same-thread call did not succeed (exit ${SAME_RC})"
    FAIL=1
fi
exit ${FAIL}
