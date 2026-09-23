#!/usr/bin/env bash
# Section 57: the tolerance model for a head-only loss. HL1-HL6.
# (!) STOP 26: h_rel, rtol and K are 49's and 51's. (!) STOP 27: coeff_h is zero.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s57"
# D82: the checked accessor, built ONCE here so every -I "${SRC}" sees the same
# one. (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc
# on its include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL
# COMPILER ERROR ("Cannot create parameter Acc.next_depth ... Misc.Fatal_error"),
# not a diagnostic. Recorded in D82 as a rule-19 finding.
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
rm -rf "${B}"; mkdir -p "${B}"; cd "${B}"
cp "${SRC}/head57.ml" "${SRC}/head57_check.ml" .

START=$(date +%s)
echo "== Section 57 =="
echo "   ocamlopt -I "${SRC}" $(ocamlopt -version)"
echo "   HL1  assume annotations in head57.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' head57.ml)"

echo
echo "-- HL1: the head-only forward and backward, both magnitude tapes, under -zero-alloc-check all"
ocamlopt -I "${SRC}" -g -zero-alloc-check all -warn-error +a -alert @all -O3 -o s57 -I . "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" head57.ml head57_check.ml
echo "   clean: forward_deep and backward_deep hold [@zero_alloc strict] with the tapes"

echo
echo "-- HL2 to HL6"
./s57 "${1:-97}"

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 57 done =="
