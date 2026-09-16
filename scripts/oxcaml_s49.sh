#!/usr/bin/env bash
# Section 49: the gradient check specified before it runs (Arm A, J1-J3), and the forward
# held to reference.py at float64 (Arm B, J4). J5 watches both instruments fail on D26's
# b_hn defect. J6 proves the dump path left 48's strict result undisturbed.
#
# (!) STOP 15: no constant in gru_fd.ml is adjusted after a number is seen.
# (!) STOP 16: src/sentinel_models/reference.py is not edited.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s49"
rm -rf "${B}"; mkdir -p "${B}/vb"; cd "${B}"

echo "== Section 49 =="
echo "   ocamlopt $(ocamlopt -version)"
echo "   assume annotations in gru_cell.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' "${SRC}/gru_cell.ml")"

echo
echo "-- J6: 48's strict result, with the Arm B dump path present"
ocamlopt -g -zero-alloc-check all -warn-error +a -alert @all -I "${SRC}" \
    -o s49_j6 "${SRC}/gru_cell.ml" "${SRC}/gru_dump.ml"
echo "   gru_cell.ml + gru_dump.ml: clean under -zero-alloc-check all"
sed 's/^let\[@inline\] sigmoid v =/let sigmoid v =/' "${SRC}/gru_cell.ml" > nohint.ml
ocamlopt -g -zero-alloc-check all -c nohint.ml >nohint.log 2>&1 || true
if grep -q "failed on function Nohint.backward" nohint.log; then
    echo "   (!) backward FAILS without the hint -- J6 is not clean"; exit 1
fi
echo "   backward still passes unhinted; only forward needs the hint, as 48.8 recorded"

echo
echo "-- J1, J2, J3: Arm A"
ocamlopt -g -I "${SRC}" -o s49_fd "${SRC}/gru_cell.ml" "${SRC}/gru_fd.ml"
set +e; ./s49_fd; ARM_A=$?; set -e
echo "   (Arm A exit ${ARM_A}: 0 = J1 HOLD, 1 = NO VERDICT, 2 = FAIL)"

echo
echo "-- J4: Arm B, against reference.py at float64"
ocamlopt -g -I "${SRC}" -o s49_dump "${SRC}/gru_cell.ml" "${SRC}/gru_dump.ml"
./s49_dump "${B}/s49_cell.txt"
set +e
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/s49_reference_check.py" "${B}/s49_cell.txt"
ARM_B=$?
set -e
echo "   (Arm B exit ${ARM_B}: 0 = J4 HOLD, 1 = NO VERDICT, 2 = FAIL, 3 = dtype stop)"

echo
echo "-- J5: both instruments must fail on D26's defect (b_hn outside the reset product)"
python3 - "${SRC}/gru_cell.ml" "${B}/vb/gru_cell.ml" <<'PY'
import sys, io
src, dst = sys.argv[1], sys.argv[2]
s = io.open(src, encoding="utf-8").read()
old = ("    let ni = tanh ((Array.unsafe_get proj ((2 * h_size) + i))\n"
       "                   +. (ri *. (Array.unsafe_get rec_ ((2 * h_size) + i)))) in")
new = ("    let bhn = Array.unsafe_get b_hh ((2 * h_size) + i) in\n"
       "    let ni = tanh ((Array.unsafe_get proj ((2 * h_size) + i))\n"
       "                   +. (ri *. ((Array.unsafe_get rec_ ((2 * h_size) + i)) -. bhn))\n"
       "                   +. bhn) in")
assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
io.open(dst, "w", encoding="utf-8").write(s.replace(old, new))
print("   variant written: b_hn moved OUTSIDE the reset product")
PY
ocamlopt -g -I vb -o s49_fd_bhn vb/gru_cell.ml "${SRC}/gru_fd.ml"
set +e; ./s49_fd_bhn > bhn_a.log 2>&1; BHN_A=$?; set -e
grep -E "entries outside|worst relative error|band: 0 outside" bhn_a.log | sed 's/^/     /'
ocamlopt -g -I vb -o s49_dump_bhn vb/gru_cell.ml "${SRC}/gru_dump.ml"
./s49_dump_bhn "${B}/s49_cell_bhn.txt" >/dev/null
set +e
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/s49_reference_check.py" "${B}/s49_cell_bhn.txt" \
    | grep -E "worst \||band:" | sed 's/^/     /'
BHN_B=${PIPESTATUS[0]}
set -e
if [ "${BHN_A}" -ne 0 ] && [ "${BHN_B}" -ne 0 ]; then
    echo "   J5 HOLD: both instruments caught the defect"
elif [ "${BHN_A}" -ne 0 ] || [ "${BHN_B}" -ne 0 ]; then
    echo "   J5 NO VERDICT: one of the two caught it (A=${BHN_A} B=${BHN_B})"
else
    echo "   (!) J5 FAIL: neither caught it. J1 and J4 are WITHDRAWN, not reported."; exit 1
fi

echo
echo "== Section 49 done =="
