#!/usr/bin/env bash
# E2: the isolation gate. docs/MODELS.md 47.9 X4/X5/X6, 47.9.1, 47.9.2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"

OCAML_LIB="$(ocamlopt -where)"
SRC="${ROOT}/oxcaml/retrainer"
# D82: the checked accessor, built ONCE here so every -I "${SRC}" sees the same
# one. (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc
# on its include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL
# COMPILER ERROR ("Cannot create parameter Acc.next_depth ... Misc.Fatal_error"),
# not a diagnostic. Recorded in D82 as a rule-19 finding.
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
BUILD="${ROOT}/oxcaml/_build"
STD="-std=c++14"; SAFETY="-fno-exceptions -fno-rtti"; DETERM="-ffp-contract=off"
WARN="-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror"

mkdir -p "${BUILD}"
cd "${BUILD}"

echo "-- ocamlopt: retrainer + thrash + runtime, one object (unix for the timed slices)"
ocamlopt -I "${SRC}" -output-complete-obj -o "${BUILD}/e2_complete.o" \
    -warn-error +a -alert @all \
    -I "${OCAML_LIB}/unix" "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" unix.cmxa \
    -I "${SRC}" "${SRC}/retrainer.ml" "${SRC}/thrash.ml" \
    "${SRC}/retrainer_stubs.c" "${SRC}/thrash_stubs.c"
echo "   $(wc -c < "${BUILD}/e2_complete.o") bytes"

echo "-- the thrash process (arms B, B2)"
# shellcheck disable=SC2086
clang++ ${STD} ${SAFETY} ${DETERM} ${WARN} -I"${SRC}" -I"${OCAML_LIB}" \
    "${SRC}/e2_thrash_main.cpp" "${BUILD}/e2_complete.o" -o "${BUILD}/e2_thrash"

echo "-- the detector harness, linking the FLIGHT CORE"
# shellcheck disable=SC2086
clang++ ${STD} ${SAFETY} ${DETERM} ${WARN} \
    -O2 -I"${ROOT}/flight/include" -I"${SRC}" -I"${OCAML_LIB}" \
    "${SRC}/e2_harness.cpp" \
    "${ROOT}"/flight/src/*.cpp \
    "${BUILD}/e2_complete.o" -o "${BUILD}/e2_harness"
echo "   built"
