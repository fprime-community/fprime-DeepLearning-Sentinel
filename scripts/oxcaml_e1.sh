#!/usr/bin/env bash
# E1 stage 1: build the OxCaml library and link it into C++ under the FLIGHT FLAG SET.
#
# docs/MODELS.md 47.9, predictions X1, X2 and X3. The flag list below is copied from
# flight/Makefile:22-25 and is IDENTICAL to it by intent -- X2 predicts the boundary
# survives contact with the OCaml runtime without a single flag being relaxed, and a
# harness built at laxer flags could not test that.
#
# (!) THE LINK NEEDS -output-complete-obj, AND THE FIRST ATTEMPT AT E1 DID NOT KNOW IT.
# Linking dune's `retrainer.a` plus `libasmrun.a` fails with about thirty undefined
# symbols -- caml_program, caml_globals, caml_frametable, caml_code_segments,
# caml_unit_deps_table and the whole caml_exn_* set. Those are not IN either archive:
# ocamlopt SYNTHESISES them at link time, because they describe the particular set of
# modules being linked. `-output-complete-obj` is the documented way to get them, and it
# emits one object carrying the compiled OCaml, the C stubs and the runtime together.
# Recorded here rather than in a commit message because the next person to embed OCaml in
# this repository will hit it in the first five minutes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
SWITCH="5.2.0+ox"

eval "$(opam env --switch="${SWITCH}" --set-switch)"

OCAML_LIB="$(ocamlopt -where)"
SRC="${ROOT}/oxcaml/retrainer"
# D82: the checked accessor, built ONCE here so every -I "${SRC}" sees the same
# one. (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc
# on its include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL
# COMPILER ERROR ("Cannot create parameter Acc.next_depth ... Misc.Fatal_error"),
# not a diagnostic. Recorded in D82 as a rule-19 finding.
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
BUILD="${ROOT}/oxcaml/_build"
OBJ="${BUILD}/retrainer_complete.o"
OUT="${BUILD}/e1_harness"

# flight/Makefile:22-25, verbatim.
STD="-std=c++14"
SAFETY="-fno-exceptions -fno-rtti"
DETERM="-ffp-contract=off"
WARN="-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror"

echo "== E1 stage 1 =="
echo "   switch     ${SWITCH}   ocamlopt -I "${SRC}" $(ocamlopt -version)"
echo "   dune       $(dune --version)"
echo "   ocaml lib  ${OCAML_LIB}"
echo "   flags      ${STD} ${SAFETY} ${DETERM} ${WARN}"
echo

echo "-- dune build: the library and the stubs compile as a library"
cd "${ROOT}/oxcaml"
dune build @all 2>&1 | sed 's/^/   /'
echo "   ok"
echo

echo "-- ocamlopt -I "${SRC}" -output-complete-obj: OCaml + stubs + runtime, one object"
mkdir -p "${BUILD}"
cd "${BUILD}"
# -warn-error +a -alert @all: strict by default, the same posture flight/Makefile takes
# with -Werror. retrainer.ml carries exactly one narrow, documented alert opt-out.
ocamlopt -I "${SRC}" -output-complete-obj -o "${OBJ}" \
    -warn-error +a -alert @all \
    -I "${SRC}" "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" \
    "${SRC}/retrainer.ml" "${SRC}/retrainer_stubs.c"
echo "   ${OBJ} ($(wc -c < "${OBJ}") bytes)"
echo

echo "-- c++ link, at the flight flag set and nothing weaker"
# shellcheck disable=SC2086
clang++ ${STD} ${SAFETY} ${DETERM} ${WARN} \
    -I"${SRC}" -I"${OCAML_LIB}" \
    "${SRC}/e1_harness.cpp" "${OBJ}" \
    -o "${OUT}"
echo "   linked: ${OUT} ($(wc -c < "${OUT}") bytes)"
echo

echo "-- run"
"${OUT}"
