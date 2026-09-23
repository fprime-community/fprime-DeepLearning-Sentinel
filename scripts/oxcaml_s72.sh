#!/usr/bin/env bash
# Section 72 / E5-e, HO1: the candidate model file leaves the OCaml process.
# (!) STOP 29: the shape fields are never written. (!) STOP 30: format_version stays 1.
# (!) shadow59.ml IS NOT EDITED. 59's [@zero_alloc strict] figures stand against the
#     file they were measured on; the C surface is a separate module, shadow_c.ml.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s72"
# D82: the checked accessor, built ONCE here so every -I "${SRC}" sees the same
# one. (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc
# on its include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL
# COMPILER ERROR ("Cannot create parameter Acc.next_depth ... Misc.Fatal_error"),
# not a diagnostic. Recorded in D82 as a rule-19 finding.
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
rm -rf "${B}"; mkdir -p "${B}"; cd "${B}"
cp "${SRC}/shadow59.ml" "${SRC}/shadow_c.ml" "${SRC}/shadow_stubs.c" \
   "${SRC}/sentinel_shadow.h" "${SRC}/sentinel_cycle.h" "${SRC}/cycle_stubs.c" \
   "${SRC}/deep_f32.ml" "${SRC}/cycle_c.ml" .

# F's validation set, cmake/flags.cmake:46-60, verbatim and opted into.
FFLAGS="-Wold-style-cast -pedantic -Wall -Wextra -Wconversion -Wdouble-promotion -Wshadow -Werror"

START=$(date +%s)
echo "== Section 72 / E5-e, HO1 =="
echo "   ocamlopt -I "${SRC}" $(ocamlopt -version)"
echo "   HZ1  assume annotations in shadow_c.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' shadow_c.ml)"
echo "   HZ1  assume annotations in shadow59.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' shadow59.ml)"
echo "   HZ2  entry points in sentinel_shadow.h: $(grep -c '^int32_t sentinel_shadow' sentinel_shadow.h) (all int32_t)"
echo "   HZ2  bare float/double in signatures: $(grep -cE '^\s*(float|double) [a-z_]+\(' sentinel_shadow.h) (stop 33)"
echo "   HZ2  caller-owned Bigarray crossings: $(grep -c caml_ba_alloc shadow_stubs.c)"
echo "   HZ3  exception guards in shadow_c.ml: $(grep -c 'guard (fun' shadow_c.ml) of 4 entry points"

echo
echo "-- HZ1: the OCaml surface under -zero-alloc-check all"
ocamlopt -I "${SRC}" -c -g -zero-alloc-check all -warn-error +a -alert @all -O3 -I . deep_f32.ml
ocamlopt -I "${SRC}" -c -g -zero-alloc-check all -warn-error +a -alert @all -O3 -I . shadow59.ml
ocamlopt -I "${SRC}" -c -g -zero-alloc-check all -warn-error +a -alert @all -O3 -I . shadow_c.ml
echo "   clean: copy_in, copy_out, weights_end and copy_loss hold [@zero_alloc strict]"

echo
echo "-- the OCaml side as one linkable object (-output-complete-obj, 47.6)"
ocamlopt -I "${SRC}" -output-complete-obj -O3 -o shadow_ml.o \
    -warn-error +a -alert @all "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" deep_f32.ml cycle_c.ml shadow59.ml shadow_c.ml
echo "   shadow_ml.o built"

echo
echo "-- HZ2: the harness at F's FULL validation set"
echo "   ${FFLAGS}"
clang++ -std=c++14 -fno-exceptions -fno-rtti ${FFLAGS} -O2 \
    -I. -I"${ROOT}/flight/include" -I"$(ocamlopt -where)" \
    -c "${SRC}/shadow_c_harness.cpp" -o harness.o
clang -std=c11 -pedantic -Wall -Wextra -Wconversion -Wshadow -Werror -O2 \
    -I. -I"$(ocamlopt -where)" -c shadow_stubs.c -o shadow_stubs.o
clang -std=c11 -pedantic -Wall -Wextra -Wconversion -Wshadow -Werror -O2 \
    -I. -I"$(ocamlopt -where)" -c cycle_stubs.c -o cycle_stubs.o
# flight/'s own sources, in their own directory so the link line is explicit
# rather than a glob that could pick up the objects above.
mkdir -p flightobj
( cd flightobj && clang++ -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off \
    ${FFLAGS} -O2 -I"${ROOT}/flight/include" -c "${ROOT}"/flight/src/*.cpp )
clang++ -o s72 harness.o shadow_stubs.o cycle_stubs.o shadow_ml.o flightobj/*.o -lm
echo "   built clean at -Werror with -Wold-style-cast and -Wdouble-promotion"

FLY="${ROOT}/flight/test/vectors/p1.bin"
echo
echo "-- HO1: load the flying bytes, write new weights, export, and read it back"
./s72 "${FLY}" "${B}/candidate.bin"

echo
echo "-- HO1: and the candidate is a file flight/'s own reader accepts from disk too"
clang++ -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off \
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -O2 \
    -I"${ROOT}/flight/include" "${SRC}/shadow59_load.cpp" \
    "${ROOT}"/flight/src/*.cpp -o s72_load
./s72_load "${FLY}" "${B}/candidate.bin"

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 72 HO1 done =="

# -- the component's object: E1's five entry points, 61's five, and 72's three.
#
# (!) THIS SUPERSEDES THE OBJECT scripts/oxcaml_s61.sh's tail BUILDS, at the same
# path, and s61.sh's tail was changed in the same series to build the identical
# set so the two runners cannot disagree about what cycle_complete.o contains.
# docs/MODELS.md 61 describes the object AS IT WAS AT 61; 72.4 is the rider.
# Two -output-complete-obj objects cannot be linked into one binary -- each
# carries its own runtime -- so there is one object and it is a superset.
echo
echo "-- the component's object: E1's five entry points, 61's five, and 72's three"
cd "${ROOT}/oxcaml/_build"
ocamlopt -I "${SRC}" -output-complete-obj -O3 -o cycle_complete.o \
    -warn-error +a -alert @all \
    -I "${SRC}" "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" \
    "${SRC}/retrainer.ml" "${SRC}/retrainer_stubs.c" \
    "${SRC}/deep_f32.ml" "${SRC}/cycle_c.ml" "${SRC}/cycle_stubs.c" \
    "${SRC}/shadow59.ml" "${SRC}/shadow_c.ml" "${SRC}/shadow_stubs.c"
echo "   cycle_complete.o ($(wc -c < cycle_complete.o) bytes)"
