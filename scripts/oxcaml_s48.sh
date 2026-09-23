#!/usr/bin/env bash
# Section 48: one GRU cell, forward and backward, under [@zero_alloc strict].
# G1-G6. -g and backtraces enabled are required; see 47.13.4 for why.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s48"
# D82: the checked accessor, built ONCE here so every -I "${SRC}" sees the same
# one. (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc
# on its include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL
# COMPILER ERROR ("Cannot create parameter Acc.next_depth ... Misc.Fatal_error"),
# not a diagnostic. Recorded in D82 as a rule-19 finding.
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
mkdir -p "${B}/variants"; cd "${B}"

echo "== Section 48 =="
echo "   ocamlopt -I "${SRC}" $(ocamlopt -version)"
echo "   assume annotations: $(grep -cE '\[@+zero_alloc[^]]*assume' "${SRC}/gru_cell.ml")   (G4)"

echo "-- G1/G2: the annotated cell must compile clean"
ocamlopt -I . -g -zero-alloc-check all -warn-error +a -alert @all -I "${SRC}" \
    -o s48_check "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" "${SRC}/gru_cell.ml" "${SRC}/gru_check.ml"
echo "   clean"

echo "-- G2 isolated: does backward hold WITHOUT the inline hint on sigmoid?"
sed 's/^let\[@inline\] sigmoid v =/let sigmoid v =/' "${SRC}/gru_cell.ml" > variants/nohint.ml
ocamlopt -I "${SRC}" -I . -g -zero-alloc-check all -c -I variants variants/nohint.ml >variants/nohint.log 2>&1 || true
if grep -q "failed on function Nohint.backward" variants/nohint.log; then
    echo "   (!) backward FAILS without the hint -- G2 is not clean"; exit 1
else
    echo "   backward passes unhinted; only forward needs it (G1's accommodation)"
fi

echo "-- G3: central finite differences over every parameter"
./s48_check

echo "-- G5: a deliberate allocation in backward must FAIL"
sed 's|    Array.unsafe_set g_h i ((Array.unsafe_get g_h i) +. d_h_direct)|    Array.unsafe_set g_h i ((Array.unsafe_get g_h i) +. d_h_direct);\
    ignore (Sys.opaque_identity (Array.make 4 0.0))|' "${SRC}/gru_cell.ml" > variants/v5.ml
if ocamlopt -I "${SRC}" -I . -g -zero-alloc-check all -c -I variants variants/v5.ml >variants/v5.log 2>&1; then
    echo "   (!) BUILT -- the gate cannot fail. G5 FAILED."; exit 1
else
    echo "   rejected in backward: $(grep -m1 'Error: called' variants/v5.log | sed 's/^ *Error: //')"
fi

echo "-- G6: against flight/'s own Gru::step"
clang++ -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off \
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -O2 \
    -I"${ROOT}/flight/include" "${SRC}/g6_flight_cell.cpp" \
    "${ROOT}"/flight/src/Gru.cpp "${ROOT}"/flight/src/ModelFile.cpp \
    "${ROOT}"/flight/src/Crc32.cpp "${ROOT}"/flight/src/Status.cpp -o s48_g6
./s48_g6
echo "== Section 48 done =="
