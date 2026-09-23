#!/usr/bin/env bash
# Section 56: window selection. WS1-WS7. Zero bucket operations -- 42's artifacts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s56"
# D82: the checked accessor, built ONCE here so every -I "${SRC}" sees the same
# one. (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc
# on its include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL
# COMPILER ERROR ("Cannot create parameter Acc.next_depth ... Misc.Fatal_error"),
# not a diagnostic. Recorded in D82 as a rule-19 finding.
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
rm -rf "${B}"; mkdir -p "${B}/v7"; cd "${B}"
cp "${SRC}/window56.ml" "${SRC}/window56_check.ml" .

START=$(date +%s)
echo "== Section 56 =="
echo "   ocamlopt -I "${SRC}" $(ocamlopt -version)"
echo "   WS1  assume annotations in window56.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' window56.ml)"

echo
echo "-- WS1: the rule must compile clean under -zero-alloc-check all"
ocamlopt -I "${SRC}" -g -zero-alloc-check all -warn-error +a -alert @all -o s56 -I . "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" window56.ml window56_check.ml
echo "   clean: push and the three gate sets hold [@zero_alloc strict]"

echo
echo "-- WS2 to WS6"
./s56 "${ROOT}/runs/testbed"

echo
echo "-- WS7: the limit gate disabled on purpose must CHANGE WS4's comparison"
python3 - window56.ml v7/window56.ml <<'PY'
import sys, io
s = io.open(sys.argv[1], encoding="utf-8").read()
old = "  if full () && (limits () = 0) && (emits () <= ceiling) then 1 else 0"
new = "  if full () && (emits () <= ceiling) then 1 else 0"
assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
io.open(sys.argv[2], "w", encoding="utf-8").write(s.replace(old, new, 1))
print("   variant written: G-limit removed from admits_all")
PY
cp window56_check.ml v7/
(cd v7 && ocamlopt -I "${SRC}" -I .. -g -o s56v -I .. "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" window56.ml window56_check.ml >b.log 2>&1)
(cd v7 && ./s56v "${ROOT}/runs/testbed" > r.log 2>&1) || true
echo "   with the gate removed, WS4 reads: $(grep -m1 'WS4  three gates' v7/r.log | sed 's/^ *//')"

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 56 done =="
