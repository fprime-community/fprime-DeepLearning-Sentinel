#!/usr/bin/env bash
# Section 51: the derived tolerance model. TM1-TM5.
# (!) STOP 15: no constant in gru_tol.ml is adjusted after a number is seen.
# (!) STOP 18: stop and report if the sweep exceeds 30 minutes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s51"
rm -rf "${B}"; mkdir -p "${B}/vb"; cd "${B}"

echo "== Section 51 =="
echo "   ocamlopt $(ocamlopt -version)"
ocamlopt -g -zero-alloc-check all -warn-error +a -alert @all -I "${SRC}" \
    -o s51 "${SRC}/gru_cell.ml" "${SRC}/gru_seq.ml" "${SRC}/gru_tol.ml"
echo "   built; gru_cell.ml and gru_seq.ml unchanged, still clean under -zero-alloc-check all"

echo
echo "-- TM1 / TM2 / TM3 / TM4: the sweep"
START=$(date +%s)
printf "    %5s  %-13s  %-13s  %7s  %-13s  %-13s  %8s\n" \
    T S absL "S/absL" atol_new worst_abs outside
for T in 1 5 25 50 125 250; do
  set +e; out=$(./s51 "$T" 2>&1); set -e
  S=$(echo "$out"   | grep -oE 'S \(accumulated magnitude\) [0-9.e+-]+' | awk '{print $NF}')
  AL=$(echo "$out"  | grep -oE 'abs\(L\) \(signed total\) [0-9.e+-]+'   | awk '{print $NF}')
  RT=$(echo "$out"  | grep -oE 'ratio [0-9.]+'                          | awk '{print $NF}')
  AT=$(echo "$out"  | grep -oE '^   atol [0-9.e+-]+'                    | awk '{print $2}')
  WA=$(echo "$out"  | grep -oE 'worst abs error [0-9.e+-]+'             | awk '{print $NF}')
  OU=$(echo "$out"  | grep -oE 'entries outside [0-9]+'                 | awk '{print $NF}')
  printf "    %5s  %-13s  %-13s  %7s  %-13s  %-13s  %8s\n" "$T" "$S" "$AL" "$RT" "$AT" "$WA" "$OU"
  echo "$out" > "run_${T}.log"
done
echo "    sweep wall clock $(( $(date +%s) - START )) s"

echo
echo "-- TM3 detail: T = 1 against 49's J1 (23,616 checked, worst abs 1.837013e-10)"
grep -E "checked|worst abs" run_1.log | sed 's/^/     /'

echo
echo "-- TM4: S(T)/T constancy, against abs(L) which wandered"
"${ROOT}/.venv/bin/python" - <<'PY'
import re, pathlib
rows=[]
for T in (1,5,25,50,125,250):
    t=pathlib.Path(f"run_{T}.log").read_text()
    S=float(re.search(r'S \(accumulated magnitude\) ([0-9.e+-]+)',t).group(1))
    L=float(re.search(r'abs\(L\) \(signed total\) ([0-9.e+-]+)',t).group(1))
    rows.append((T,S,L))
per=[S/T for T,S,_ in rows]
print(f"     S(T)/T : {'  '.join('%.4f'%v for v in per)}")
print(f"     spread : {max(per)/min(per):.4f}x   (band: <=1.20 HOLD, <=2.0 NO VERDICT)")
Ls=[L for _,_,L in rows]
print(f"     abs(L) spread for comparison: {max(Ls)/min(Ls):.1f}x")
mono = all(rows[i][1] < rows[i+1][1] for i in range(len(rows)-1))
print(f"     S monotonically increasing in T: {mono}")
PY

echo
echo "-- TM5: the criterion must STILL FAIL on D26's b_hn defect at T = 250"
python3 - "${SRC}/gru_cell.ml" "${B}/vb/gru_cell.ml" <<'PY'
import sys, io
s = io.open(sys.argv[1], encoding="utf-8").read()
old = ("    let ni = tanh ((Array.unsafe_get proj ((2 * h_size) + i))\n"
       "                   +. (ri *. (Array.unsafe_get rec_ ((2 * h_size) + i)))) in")
new = ("    let bhn = Array.unsafe_get b_hh ((2 * h_size) + i) in\n"
       "    let ni = tanh ((Array.unsafe_get proj ((2 * h_size) + i))\n"
       "                   +. (ri *. ((Array.unsafe_get rec_ ((2 * h_size) + i)) -. bhn))\n"
       "                   +. bhn) in")
assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
io.open(sys.argv[2], "w", encoding="utf-8").write(s.replace(old, new))
print("     variant written: b_hn moved OUTSIDE the reset product")
PY
cp "${SRC}/gru_seq.ml" "${SRC}/gru_tol.ml" vb/
ocamlopt -g -I vb -o s51_bhn vb/gru_cell.ml vb/gru_seq.ml vb/gru_tol.ml
set +e; ./s51_bhn 250 > bhn.log 2>&1; BHN=$?; set -e
grep -E "checked|worst abs|S \(accum" bhn.log | sed 's/^/     /'
if [ "${BHN}" -eq 2 ]; then echo "     TM5 HOLD: FAIL verdict on a genuinely wrong cell"
elif [ "${BHN}" -eq 1 ]; then echo "     TM5 NO VERDICT: 1 to 10 outside"
else echo "     (!) TM5 FAIL: 0 outside -- the criterion accepts a wrong gradient"; fi

echo
echo "== Section 51 done =="
