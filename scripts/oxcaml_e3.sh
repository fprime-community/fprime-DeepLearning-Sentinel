#!/usr/bin/env bash
# E3: [@zero_alloc strict] on real arithmetic. docs/MODELS.md 47.9 X7/X8/X9, 47.13.4.
#
# -g and backtraces enabled are REQUIRED by the pre-registration and are not incidental:
# without -g every `raise` is treated as `raise_notrace`, the relaxed annotation rejects
# what strict rejects, and the distinction strict was mandated for becomes invisible.
# Demonstrated at 47.13.4.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"

SRC="${ROOT}/oxcaml/retrainer"
B="${ROOT}/oxcaml/_build/e3"
mkdir -p "${B}"; cd "${B}"

echo "== E3: [@zero_alloc strict], -g, -zero-alloc-check all, no assume =="
echo "   ocamlopt $(ocamlopt -version)"
echo "   assume annotations in zalloc.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' "${SRC}/zalloc.ml")"

echo "-- X7/X9: the annotated file must compile clean"
ocamlopt -g -zero-alloc-check all -warn-error +a -alert @all \
    -o e3_arith -I "${SRC}" "${SRC}/zalloc.ml" "${SRC}/zalloc_main.ml"
echo "   clean"

echo "-- X7: and the annotated functions must compute what they claim"
./e3_arith

echo "-- X8: a deliberate allocation must FAIL the build"
mkdir -p probes
sed 's|      Array.unsafe_set c ((i \* n) + j) !s|      Array.unsafe_set c ((i * n) + j) !s;\
      ignore (Sys.opaque_identity (Array.make 8 0.0))|' "${SRC}/zalloc.ml" > probes/p_alloc.ml
if ocamlopt -g -zero-alloc-check all -c -I probes probes/p_alloc.ml >probes/p_alloc.log 2>&1
then
    echo "   (!) BUILT -- the gate cannot fail. X8 FAILED."; exit 1
else
    echo "   rejected: $(grep -m1 'Error: called' probes/p_alloc.log | sed 's/^ *Error: //')"
fi
echo "== E3 done =="
