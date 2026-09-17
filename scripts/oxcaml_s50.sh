#!/usr/bin/env bash
# Section 50: BPTT over a fixed tape. U1-U6.
# (!) STOP 15: no constant in gru_seq_check.ml is adjusted after a number is seen.
# (!) STOP 17: U3 is abandoned and reported if it exceeds 20 minutes of wall clock.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s50"
rm -rf "${B}"; mkdir -p "${B}/v5"; cd "${B}"

echo "== Section 50 =="
echo "   ocamlopt $(ocamlopt -version)"
echo "   U4  assume annotations in gru_seq.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' "${SRC}/gru_seq.ml")"

echo
echo "-- U1 and U2: the sequence forward and the backward sweep must compile clean"
if ocamlopt -g -zero-alloc-check all -warn-error +a -alert @all -I "${SRC}" \
      -o s50_check "${SRC}/gru_cell.ml" "${SRC}/gru_seq.ml" "${SRC}/gru_seq_check.ml" \
      >build.log 2>&1; then
    echo "   clean: forward_seq, backward_seq and zero_step_grads all hold [@zero_alloc strict]"
else
    echo "   (!) BUILD FAILED -- U1 or U2 is not clean:"
    grep -E "Error|failed on function" build.log | sed 's/^/     /'
    exit 1
fi

echo
echo "-- U6: T = 1 must reproduce 49's J1 bit-for-bit (23,616 checked, 1.837013e-10)"
./s50_check 1 | grep -E "checked|entries outside|worst abs|band:" | sed 's/^/     /' || true
./s50_check 1 > u6.log 2>&1 || true
C6=$(grep -oE 'over all [0-9]+' u6.log | grep -oE '[0-9]+')
W6=$(grep -oE 'worst abs error +[0-9.e+-]+' u6.log | awk '{print $NF}')
if [ "${C6}" = "23616" ] && [ "${W6}" = "1.837013e-10" ]; then
    echo "   U6 HOLD: ${C6} checked, worst abs ${W6} -- identical to 49's J1"
else
    echo "   (!) U6: ${C6} checked, worst abs ${W6} -- 49's J1 was 23616 / 1.837013e-10"
fi

echo
echo "-- U5: a deliberate allocation INSIDE the backward-through-time loop must FAIL"
python3 - "${SRC}/gru_seq.ml" "${B}/v5/gru_seq.ml" <<'PY'
import sys, io
s = io.open(sys.argv[1], encoding="utf-8").read()
old = "      Array.unsafe_set carry i (Array.unsafe_get Gru_cell.g_h i)"
new = ("      Array.unsafe_set carry i (Array.unsafe_get Gru_cell.g_h i);\n"
       "      ignore (Sys.opaque_identity (Array.make 4 0.0))")
assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
io.open(sys.argv[2], "w", encoding="utf-8").write(s.replace(old, new))
print("   variant written: Array.make inside backward_seq's reverse loop")
PY
if ocamlopt -g -zero-alloc-check all -I "${SRC}" -I v5 -c "${SRC}/gru_cell.ml" v5/gru_seq.ml \
      >v5.log 2>&1; then
    echo "   (!) BUILT -- the gate cannot fail. U5 FAILED."; exit 1
else
    echo "   rejected in backward_seq: $(grep -m1 -E 'Error: called|may allocate' v5.log | sed 's/^ *Error: //')"
fi

echo
echo "-- U3: the full check, all 27,600 parameters and inputs at T = 250"
echo "   (estimate 207 s from 49's measured 66,800 cell-forwards/s; stop 17 at 20 min)"
START=$(date +%s)
set +e; ./s50_check 250; U3=$?; set -e
echo "   wall clock $(( $(date +%s) - START )) s   (U3 exit ${U3}: 0 HOLD, 1 NO VERDICT, 2 FAIL)"

echo
echo "== Section 50 done =="
