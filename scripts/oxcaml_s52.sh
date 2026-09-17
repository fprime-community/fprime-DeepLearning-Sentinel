#!/usr/bin/env bash
# Section 52: two layers and the output head. HD1-HD7.
# (!) STOP 15: no constant in gru_deep_check.ml is adjusted after a number is seen.
# (!) STOP 19: stop and report beyond 45 minutes.
# Builds from a SELF-CONTAINED directory: 49.8 and 51.8 both lost a run to an -I on a
# variant link, and a third happened while writing this. Copying sources removes the class.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s52"
NPROC="${NPROC:-8}"
rm -rf "${B}"; mkdir -p "${B}/alloc" "${B}/stack"; cd "${B}"
cp "${SRC}/gru_deep.ml" "${SRC}/gru_deep_check.ml" .

echo "== Section 52 =="
echo "   ocamlopt $(ocamlopt -version)"
echo "   HD5  assume annotations in gru_deep.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' gru_deep.ml)"

echo
echo "-- HD1, HD2, HD3: two layers, the backward sweep, and the head must compile clean"
ocamlopt -g -zero-alloc-check all -warn-error +a -alert @all -o s52 gru_deep.ml gru_deep_check.ml
echo "   clean: layer_fwd, layer_bwd, forward_deep, backward_deep and zero_grads_deep all hold"

echo
echo "-- HD7: two layers and head against reference.forward"
./s52 0 1 250 dump "${B}/s52_cell.txt"
set +e
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/s52_reference_check.py" "${B}/s52_cell.txt"
HD7=$?
set -e

echo
echo "-- HD6a: an allocation inside layer 1's backward must FAIL the build"
python3 - gru_deep.ml alloc/gru_deep.ml <<'PY'
import sys, io
s = io.open(sys.argv[1], encoding="utf-8").read()
old = "    Array.unsafe_set g_hin i ((Array.unsafe_get g_hin i) +. d_h_direct)"
new = ("    Array.unsafe_set g_hin i ((Array.unsafe_get g_hin i) +. d_h_direct);\n"
       "    ignore (Sys.opaque_identity (Array.make 4 0.0))")
assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
io.open(sys.argv[2], "w", encoding="utf-8").write(s.replace(old, new))
print("   variant written: Array.make inside layer_bwd, which serves BOTH layers")
PY
cp gru_deep_check.ml alloc/
if (cd alloc && ocamlopt -g -zero-alloc-check all -c gru_deep.ml >a.log 2>&1); then
    echo "   (!) BUILT -- the gate cannot fail. HD6a FAILED."; exit 1
else
    echo "   rejected: $(grep -m1 -oE 'called function may allocate.*' alloc/a.log)"
fi

echo
echo "-- HD6b: a mis-stacked model must be caught"
echo "   (!) 52.4's wording -- feed layer 1 the input x -- CANNOT BE BUILT: layer 1's input"
echo "       width is 80 and x is 16. The nearest valid mis-stacking is used instead:"
echo "       layer 1 is fed layer 0's output from the PREVIOUS timestep."
python3 - gru_deep.ml stack/gru_deep.ml <<'PY'
import sys, io
s = io.open(sys.argv[1], encoding="utf-8").read()
old = "    layer_fwd w_ih1 w_hh1 b_ih1 b_hh1 hs out0 (t * hs) h_cur1 0 tape1 tb h_cur1 0;"
new = ("    layer_fwd w_ih1 w_hh1 b_ih1 b_hh1 hs out0 (if t = 0 then 0 else (t - 1) * hs)\n"
       "              h_cur1 0 tape1 tb h_cur1 0;")
assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
io.open(sys.argv[2], "w", encoding="utf-8").write(s.replace(old, new))
print("   variant written")
PY
cp gru_deep_check.ml stack/
(cd stack && ocamlopt -g -o s52bad gru_deep.ml gru_deep_check.ml >/dev/null 2>&1)
set +e
BAD=$(cd stack && for p in $(seq 0 $((NPROC-1))); do ./s52bad "$p" "$NPROC" 250 & done; wait)
set -e
BADOUT=$(echo "$BAD" | grep -oE 'outside [0-9]+' | awk '{s+=$2} END {print s+0}')
echo "   mis-stacked model: ${BADOUT} entries outside"
if [ "${BADOUT}" -gt 10 ]; then echo "   HD6b HOLD: the criterion catches it"; else echo "   (!) HD6b FAIL"; fi

echo
echo "-- HD4: all parameters at T = 250, across ${NPROC} processes (bit-identical to serial)"
START=$(date +%s)
set +e
OUT=$(for p in $(seq 0 $((NPROC-1))); do ./s52 "$p" "$NPROC" 250 & done; wait)
set -e
echo "$OUT" | grep -E '^(==|   )' | head -4
echo "$OUT" | grep '^SLICE' | sort -t/ -k1
TOT=$(echo "$OUT"  | grep -oE 'checked [0-9]+'       | awk '{s+=$2} END {print s}')
OUTS=$(echo "$OUT" | grep -oE 'outside [0-9]+'       | awk '{s+=$2} END {print s}')
WA=$(echo "$OUT"   | grep -oE 'worst_abs [0-9.e+-]+' | awk '{if($2>m)m=$2} END {printf "%.6e", m}')
echo "   AGGREGATE: checked ${TOT}, entries outside ${OUTS}, worst abs ${WA}"
echo "$OUT" | grep '^  OUT' | head -12
if [ "${OUTS}" -eq 0 ]; then V="HOLD"; elif [ "${OUTS}" -le 10 ]; then V="NO VERDICT"; else V="FAIL"; fi
echo "   band: 0 outside HOLD, <=10 NO VERDICT, else FAIL -> ${V}"
echo "   wall clock $(( $(date +%s) - START )) s across ${NPROC} processes   (HD7 exit ${HD7})"

echo
echo "== Section 52 done =="
