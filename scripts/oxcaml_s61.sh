#!/usr/bin/env bash
# Section 61 / E5-c: the cycle driven across the C boundary, at F's own flag set.
# (!) STOP 32: Retrainer stays uninstanced. (!) STOP 33: no float/double in a signature.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s61"
# D82: the checked accessor, built ONCE here so every -I "${SRC}" sees the same
# one. (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc
# on its include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL
# COMPILER ERROR ("Cannot create parameter Acc.next_depth ... Misc.Fatal_error"),
# not a diagnostic. Recorded in D82 as a rule-19 finding.
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
rm -rf "${B}"; mkdir -p "${B}"; cd "${B}"
cp "${SRC}/deep_f32.ml" "${SRC}/cycle_c.ml" "${SRC}/cycle_stubs.c" "${SRC}/sentinel_cycle.h" .

# F's validation set, cmake/flags.cmake:46-60, verbatim and opted into.
FFLAGS="-Wold-style-cast -pedantic -Wall -Wextra -Wconversion -Wdouble-promotion -Wshadow -Werror"

START=$(date +%s)
echo "== Section 61 / E5-c =="
echo "   ocamlopt -I "${SRC}" $(ocamlopt -version)"
echo "   FC1  assume annotations in cycle_c.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' cycle_c.ml)"
echo "   FC2  entry points in sentinel_cycle.h: $(grep -c '^int32_t sentinel_cycle' sentinel_cycle.h) (all int32_t)"
echo "   FC2  bare float/double in signatures: $(grep -cE '^\s*(float|double) [a-z_]+\(' sentinel_cycle.h) (stop 33)"
echo "   FC2  CAML_BA_EXTERNAL crossings: $(grep -c CAML_BA_EXTERNAL cycle_stubs.c)"

echo
echo "-- FC1: the OCaml surface under -zero-alloc-check all"
ocamlopt -I "${SRC}" -c -g -zero-alloc-check all -warn-error +a -alert @all -O3 -I . deep_f32.ml
ocamlopt -I "${SRC}" -c -g -zero-alloc-check all -warn-error +a -alert @all -O3 -I . cycle_c.ml
echo "   clean: seed_from, copy_window and copy_best hold [@zero_alloc strict]"

echo
echo "-- the OCaml side as one linkable object (-output-complete-obj, 47.6)"
ocamlopt -I "${SRC}" -output-complete-obj -O3 -o cycle_ml.o -I . "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" deep_f32.ml cycle_c.ml
echo "   cycle_ml.o built"

echo
echo "-- FC3: the harness at F's FULL validation set"
echo "   ${FFLAGS}"
# The C++ translation unit is what FC3 claims, so it is compiled ALONE at the full
# set. The C stubs are C and are compiled as C; mixing them in one clang++ call
# would only produce a -Wdeprecated complaint about the input language.
clang++ -std=c++14 -fno-exceptions -fno-rtti ${FFLAGS} -O2 -I. -I"$(ocamlopt -where)" \
    -c "${SRC}/cycle_harness.cpp" -o harness.o
clang -std=c11 -pedantic -Wall -Wextra -Wconversion -Wshadow -Werror -O2 \
    -I. -I"$(ocamlopt -where)" -c cycle_stubs.c -o cycle_stubs.o
clang++ -o s61 harness.o cycle_stubs.o cycle_ml.o -lm
echo "   built clean at -Werror with -Wold-style-cast and -Wdouble-promotion"

echo
echo "-- FC5, FC7: the cycle across the boundary"
./s61

echo
echo "-- FC4: ASan and UBSan over the same cycle"
clang++ -std=c++14 -fno-exceptions -fno-rtti -O1 -g -fsanitize=address,undefined \
    -fno-omit-frame-pointer -I. -I"$(ocamlopt -where)" \
    "${SRC}/cycle_harness.cpp" -c -o harness_san.o
clang -I"$(ocamlopt -where)" -O1 -g -fsanitize=address,undefined -c cycle_stubs.c -o stubs_san.o
clang++ -fsanitize=address,undefined -o s61_san harness_san.o stubs_san.o cycle_ml.o -lm
set +e
ASAN_OPTIONS=detect_leaks=0 ./s61_san > san.log 2>&1; SAN=$?
set -e
if grep -qE "runtime error|AddressSanitizer|SUMMARY:" san.log; then
    echo "   (!) SANITISER REPORT:"; grep -m6 -E "runtime error|AddressSanitizer|SUMMARY:" san.log
else
    echo "   clean: no ASan or UBSan report (exit ${SAN})"
fi

echo
echo "-- FC6: Retrainer must stay uninstanced"
# (!) THE CHECK IS AN INSTANCING, NOT THE WORD, AND THE DIFFERENCE COST A BUILD.
# This read `grep -rqi retrainer`, which claimed "the Retrainer is not instanced"
# but tested "the word does not appear in Top/". The two parted company the
# moment the hub arrangement was documented: three comments in `topology.fpp`
# and `instances.fpp` name the retrainer to say which side connects, so FC6
# failed a deployment that instances nothing of the kind -- and took the
# `-output-complete-obj` build below down with it, because the script exits here.
# A guard's scope is part of its claim, so this matches an FPP instancing --
# `instance retrainer`, or the `Retrain.Retrainer` type -- and nothing else.
# 61.5a narrowed stop 32 to permit the separate deployment; this is that
# narrowing enforced rather than remembered. It still exits 1 on a real
# instancing: `fprime/SentinelRetrain/Top/` matches both alternatives.
if grep -rqE '^[[:space:]]*instance[[:space:]]+retrainer\b|Retrain\.Retrainer' \
        "${ROOT}/fprime/SentinelRef/Top/"; then
    echo "   (!) INSTANCED -- stop 32."; exit 1
else
    echo "   uninstanced: no Retrainer instance and no Retrain.Retrainer in Top/"
fi

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 61 done =="

# -- the component's object: E1's surface PLUS 61's cycle, one -output-complete-obj.
#
# (!) E1's OWN OBJECT IS NOT TOUCHED. scripts/oxcaml_e1.sh still builds
# oxcaml/_build/retrainer_complete.o from retrainer.ml and retrainer_stubs.c alone,
# and 47.13.1's figures still refer to it. This is a SUPERSET at a different path,
# because two -output-complete-obj objects cannot be linked into one binary -- each
# carries its own runtime.
#
# (!) SECTION 72 ADDED THE SHADOW SURFACE TO THIS OBJECT, and this tail builds the
# SAME set as scripts/oxcaml_s72.sh's tail so the two runners cannot disagree about
# what cycle_complete.o contains -- whichever runs last, the object is identical.
# docs/MODELS.md 61 describes the object AS IT WAS AT 61, five entry points plus
# five; 72.4 is the rider that records the third set and the symbol count it moved.
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
