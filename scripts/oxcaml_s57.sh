#!/usr/bin/env bash
# Section 57: the tolerance model for a head-only loss. HL1-HL6.
# (!) STOP 26: h_rel, rtol and K are 49's and 51's. (!) STOP 27: coeff_h is zero.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s57"
rm -rf "${B}"; mkdir -p "${B}"; cd "${B}"
cp "${SRC}/head57.ml" "${SRC}/head57_check.ml" .

START=$(date +%s)
echo "== Section 57 =="
echo "   ocamlopt $(ocamlopt -version)"
echo "   HL1  assume annotations in head57.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' head57.ml)"

echo
echo "-- HL1: the head-only forward and backward, both magnitude tapes, under -zero-alloc-check all"
ocamlopt -g -zero-alloc-check all -warn-error +a -alert @all -O3 -o s57 head57.ml head57_check.ml
echo "   clean: forward_deep and backward_deep hold [@zero_alloc strict] with the tapes"

echo
echo "-- HL2 to HL6"
./s57 "${1:-97}"

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 57 done =="
