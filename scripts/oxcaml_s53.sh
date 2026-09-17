#!/usr/bin/env bash
# Section 53: Adam under [@zero_alloc strict]. AD1-AD6.
# (!) STOP 15: nothing is adjusted after a number is seen.
# (!) STOP 20: nothing here starts training -- no epochs, no stopping rule.
# Builds from a SELF-CONTAINED directory (52.8's structural fix).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s53"
rm -rf "${B}"; mkdir -p "${B}/v5"; cd "${B}"
cp "${SRC}/adam.ml" "${SRC}/gru_deep.ml" "${SRC}/adam_check.ml" .

START=$(date +%s)
echo "== Section 53 =="
echo "   ocamlopt $(ocamlopt -version)"
echo "   AD4  assume annotations in adam.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' adam.ml)"

echo
echo "-- AD1, AD2: the update and BOTH bias-correction forms must compile clean"
ocamlopt -g -zero-alloc-check all -warn-error +a -alert @all -o s53 adam.ml gru_deep.ml adam_check.ml
echo "   clean: update_block, begin_step_pow ( ** ) and begin_step_running (no pow) all hold"
echo "   53.3's reading confirmed: ( ** ) and sqrt are [@@noalloc] and are accepted"

echo
echo "-- AD3: 20 updates against torch's algorithm as read"
./s53 traj "${B}/s53_traj.txt"
set +e
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/s53_adam_reference.py" "${B}/s53_traj.txt"
AD3=$?
set -e

echo
echo "-- AD5: a deliberate allocation inside the update loop must FAIL"
python3 - adam.ml v5/adam.ml <<'PY'
import sys, io
s = io.open(sys.argv[1], encoding="utf-8").read()
old = "    Array.unsafe_set p i ((Array.unsafe_get p i) +. ((-. ss) *. (mi' /. denom)))"
new = ("    Array.unsafe_set p i ((Array.unsafe_get p i) +. ((-. ss) *. (mi' /. denom)));\n"
       "    ignore (Sys.opaque_identity (Array.make 4 0.0))")
assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
io.open(sys.argv[2], "w", encoding="utf-8").write(s.replace(old, new))
print("   variant written: Array.make inside update_block")
PY
if (cd v5 && ocamlopt -g -zero-alloc-check all -c adam.ml >a.log 2>&1); then
    echo "   (!) BUILT -- the gate cannot fail. AD5 FAILED."; exit 1
else
    echo "   rejected: $(grep -m1 -oE 'called function may allocate.*' v5/a.log)"
fi

echo
echo "-- AD6: one full training step, and every parameter must move"
set +e; ./s53 step; AD6=$?; set -e
echo "   (AD6 exit ${AD6}: 0 = all moved, 1 = 1-100 unmoved, 2 = more)"

echo
echo "   wall clock $(( $(date +%s) - START )) s   (53.7 quoted no estimate, by design)"
echo "== Section 53 done =="
