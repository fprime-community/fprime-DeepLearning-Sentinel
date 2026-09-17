#!/usr/bin/env bash
# Section 58: TM4 re-run with the transient's treatment declared first. TR1-TR5.
# (!) STOP 28: the band is 20% and the six T are 51's. Neither moves.
# gru_cell.ml, gru_seq.ml and gru_tol.ml are UNCHANGED -- this is 51's sweep re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s58"
rm -rf "${B}"; mkdir -p "${B}"; cd "${B}"

START=$(date +%s)
echo "== Section 58 =="
echo "   ocamlopt $(ocamlopt -version)"
ocamlopt -g -zero-alloc-check all -warn-error +a -alert @all -I "${SRC}" \
    -o s58 "${SRC}/gru_cell.ml" "${SRC}/gru_seq.ml" "${SRC}/gru_tol.ml"
echo "   built from UNEDITED 51 sources, still clean under -zero-alloc-check all"

echo
echo "-- the sweep, six T of record"
printf "    %5s  %-15s  %-15s\n" T "S" "abs(L)"
: > s.txt
for T in 1 5 25 50 125 250; do
  set +e; out=$(./s58 "$T" 2>&1); set -e
  S=$(echo "$out"  | grep -oE 'S \(accumulated magnitude\) [0-9.e+-]+' | awk '{print $NF}')
  AL=$(echo "$out" | grep -oE 'abs\(L\) \(signed total\) [0-9.e+-]+'   | awk '{print $NF}')
  printf "    %5s  %-15s  %-15s\n" "$T" "$S" "$AL"
  echo "$T $S $AL" >> s.txt
done

echo
python3 - s.txt <<'PY'
import sys
rows = [l.split() for l in open(sys.argv[1])]
d = {int(t): (float(s), float(a)) for t, s, a in rows}
Ts = [1, 5, 25, 50, 125, 250]
five = Ts[1:]
S1 = d[1][0]

A = {t: d[t][0] / t for t in five}
B = {t: (d[t][0] - S1) / (t - 1) for t in five}
def spread(x): 
    v = list(x.values()); return max(v) / min(v)
print("   ARM A   S(T)/T              " + "  ".join(f"{t}:{A[t]:.6g}" for t in five))
print(f"     spread max/min {spread(A):.4f}")
print("   ARM B   (S(T)-S(1))/(T-1)   " + "  ".join(f"{t}:{B[t]:.6g}" for t in five))
print(f"     spread max/min {spread(B):.4f}")
def verdict(x):
    s = spread(x)
    return "HELD" if s <= 1.20 else ("NO VERDICT" if s <= 2.00 else "FAIL")
print(f"   TR1  arm A within 20%: {verdict(A)}  ({spread(A):.4f})")
print(f"   TR2  arm B within 20%: {verdict(B)}  ({spread(B):.4f})")
print(f"   TR3  arm B tighter than arm A: {'YES' if spread(B) < spread(A) else 'NO'}")
Sv = [d[t][0] for t in Ts]
print(f"   TR4  S monotonically increasing: {'YES' if all(b > a for a, b in zip(Sv, Sv[1:])) else 'NO'}")
Lv = [d[t][1] for t in Ts]
print(f"   TR5  S(250) = {d[250][0]:.6e}   abs(L) spread over six {max(Lv)/min(Lv):.4f}x")
print(f"        51's TM4 reported S(T)/T spread 1.5501x over all six; over all six here: "
      f"{max(d[t][0]/t for t in Ts)/min(d[t][0]/t for t in Ts):.4f}x")
PY

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 58 done =="
