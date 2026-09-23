#!/usr/bin/env bash
# Section 60 / E5-b: the training cycle assembled at float32. AS1-AS7.
# (!) STOP 31: 54.2a's permitted set, extended by nothing. The square root is
# route 2 (60.2) and reaches no C external.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s60"
# D82: the checked accessor, built ONCE here so every -I "${SRC}" sees the same
# one. (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc
# on its include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL
# COMPILER ERROR ("Cannot create parameter Acc.next_depth ... Misc.Fatal_error"),
# not a diagnostic. Recorded in D82 as a rule-19 finding.
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
rm -rf "${B}"; mkdir -p "${B}/v8"; cd "${B}"
cp "${SRC}/deep_f32.ml" "${SRC}/deep_f32_check.ml" "${SRC}/sqrtf_ref.c" .

START=$(date +%s)
echo "== Section 60 / E5-b =="
echo "   ocamlopt -I "${SRC}" $(ocamlopt -version)"
echo "   AS1  assume annotations in deep_f32.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' deep_f32.ml)"
echo "   primitives used: $(grep -oE '"%[a-z0-9_]+"' deep_f32.ml | sort -u | tr '\n' ' ')"
echo "   C externals in the CYCLE: $(grep -cE 'external .*= "caml_' deep_f32.ml) (expf, tanhf -- 54.2's two)"

echo
echo "-- AS1: the whole cycle under -zero-alloc-check all"
ocamlopt -I "${SRC}" -c -g -zero-alloc-check all -warn-error +a -alert @all -O3 -I . deep_f32.ml
echo "   clean: forward_deep, backward_deep, adam_step, save_best and run_cycle hold strict"

ocamlopt -I "${SRC}" -g -O3 -o s60 sqrtf_ref.c -I . "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" deep_f32.ml deep_f32_check.ml

echo
echo "-- AS2, AS4, AS5, AS6"
./s60 all "${1:-250}" "${2:-1021}"

echo
echo "-- AS3: the F32 forward against reference.forward, via 52's comparator, unmodified"
./s60 dump "${1:-250}" "${B}/s60_cell.txt"
set +e; "${ROOT}/.venv/bin/python" "${ROOT}/scripts/s52_reference_check.py" "${B}/s60_cell.txt"; AS3=$?; set -e
echo "   (AS3 exit ${AS3}: 0 HOLD, 1 NO VERDICT, 2 FAIL)"

echo
echo "-- AS7a: a deliberate allocation in the F32 cycle must FAIL the build"
python3 - deep_f32.ml v8/deep_f32.ml <<'PY'
import sys, io
s = io.open(sys.argv[1], encoding="utf-8").read()
old = "      save_best ();                        (* D73 c.2: bookkeeping, not control flow *)"
new = ("      save_best ();\n"
       "      ignore (Sys.opaque_identity (Array.make 4 0.0));")
assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
io.open(sys.argv[2], "w", encoding="utf-8").write(s.replace(old, new))
print("   variant written: Array.make inside run_cycle")
PY
if (cd v8 && ocamlopt -I "${SRC}" -I .. -c -g -zero-alloc-check all -O3 -I .. deep_f32.ml >a.log 2>&1); then
    echo "   (!) BUILT -- the gate cannot fail. AS7a FAILED."; exit 1
else
    echo "   rejected: $(grep -m1 -oE 'called function may allocate.*' v8/a.log)"
fi

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 60 done =="
