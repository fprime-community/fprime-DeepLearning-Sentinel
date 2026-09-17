#!/usr/bin/env bash
# Section 55 arm B: the training cycle under D73's fixed step budget. DT3-DT7.
# (!) STOP 21 still applies (55.8 carries it). Only 54.2a's permitted set is used,
# and no C external is called: an F32 Adam would need a float32 square root, which
# 54.2 measured ABSENT, so the optimiser is E5-b's rung and this loop is SGD.
# Builds from a SELF-CONTAINED directory (52.8's structural fix).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s55"
rm -rf "${B}"; mkdir -p "${B}/v7"; cd "${B}"
cp "${SRC}/gru_f32.ml" "${SRC}/cycle55.ml" "${SRC}/cycle55_check.ml" .

START=$(date +%s)
echo "== Section 55 arm B =="
echo "   ocamlopt $(ocamlopt -version)"
echo "   DT6  assume annotations in cycle55.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' cycle55.ml)"
echo "   primitives used: $(grep -oE '"%[a-z0-9_]+"' cycle55.ml | sort -u | tr '\n' ' ')"
echo "   C externals called: $(grep -cE 'external .*= "[a-z]' cycle55.ml)"

echo
echo "-- DT6: the cycle must compile clean under -zero-alloc-check all"
ocamlopt -g -zero-alloc-check all -warn-error +a -alert @all -o s55 \
    gru_f32.ml cycle55.ml cycle55_check.ml
echo "   clean: run_cycle holds [@zero_alloc strict]"

echo
echo "-- DT3, DT4, DT5: run once, then again in a FRESH PROCESS"
./s55 "${B}/cycle55.crc"
echo
./s55 "${B}/cycle55.crc"

echo
echo "-- DT7: determinism broken on purpose must be CAUGHT"
python3 - cycle55.ml v7/cycle55.ml <<'PY'
import sys, io
s = io.open(sys.argv[1], encoding="utf-8").read()
old = "let[@inline] keep seed k i = (rng seed k i) * drop_den >= drop_num * rng_span"
new = ("let[@inline] keep seed k i =\n"
       "  (rng (seed + (int_of_float (Unix.gettimeofday () *. 1e6))) k i) * drop_den\n"
       "  >= drop_num * rng_span")
assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
io.open(sys.argv[2], "w", encoding="utf-8").write(s.replace(old, new))
print("   variant written: the dropout seed drawn from the clock")
PY
cp gru_f32.ml cycle55_check.ml v7/
if (cd v7 && ocamlopt -g -I "$(ocamlfind query unix 2>/dev/null || echo .)" unix.cmxa \
        -o s55v gru_f32.ml cycle55.ml cycle55_check.ml >b.log 2>&1); then
    rm -f "${B}/v7/cycle55.crc"
    set +e
    (cd v7 && ./s55v "${B}/v7/cycle55.crc" >r1.log 2>&1; ./s55v "${B}/v7/cycle55.crc" >r2.log 2>&1)
    RC=$?
    set -e
    if [ "${RC}" -ne 0 ] && grep -q "DIFFER" v7/r2.log; then
        echo "   caught: $(grep -m1 'DT3' v7/r2.log | sed 's/^ *//')"
    else
        echo "   (!) NOT CAUGHT -- DT3's comparison cannot fail. DT7 FAILED."; exit 1
    fi
else
    echo "   variant did not build; DT7 inconclusive:"; tail -3 v7/b.log; exit 1
fi

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 55 arm B done =="
