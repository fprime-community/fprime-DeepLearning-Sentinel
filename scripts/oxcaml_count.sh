#!/usr/bin/env bash
# D82.1: the bounds checks a retraining cycle executes, COUNTED. Not timed.
#
# (!) WHY THIS EXISTS. D82 made the flown retrainer bounds-checked and left the
# run-time cost unquantified, because stop 35 forbids quoting a timing figure from
# this work. D82.1 keeps stop 35 -- the wall clock belongs to the flight-hardware
# session -- and asks for a deterministic proxy instead. A COUNT is not a timing
# figure: it is exact, host-independent, and it is the same currency as 19.8 F4's
# **70,080 MAC per tick** for the detector.
#
# The count is taken against `acc_counting.ml`, which is `acc.ml` plus one `incr` per
# access. Two things are checked rather than assumed:
#
#   * the counting build still holds `[@zero_alloc strict]` -- an `int ref` increment
#     allocates nothing, and a proxy that perturbed the property being measured would
#     be worthless;
#   * the count is SHAPE-determined, so it does not depend on the values in the
#     arrays, which is why a zero-initialised cycle gives the flown figure.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"
B="${ROOT}/oxcaml/_build/count"
rm -rf "${B}"; mkdir -p "${B}"; cd "${B}"

cp "${SRC}/acc_counting.ml" acc.ml
cp "${SRC}/acc_int.ml" "${SRC}/deep_f32.ml" .

cat > count_probe.ml <<'ML'
(* One cycle at a budget of 1, then at 2, so the per-step cost separates from the
   fixed set-up. The arrays are whatever `Deep_f32` initialised them to: the count is
   determined by the loop bounds, not by the values. *)
let () =
  let t_steps = Deep_f32.t_max in
  Acc.reset ();
  Deep_f32.run_cycle 1 t_steps 1;
  let one = Acc.count () in
  Acc.reset ();
  Deep_f32.run_cycle 2 t_steps 1;
  let two = Acc.count () in
  Printf.printf "BUDGET1 %d\nBUDGET2 %d\nPERSTEP %d\nFIXED %d\n"
    one two (two - one) (one - (two - one))
ML

echo "== D82.1: bounds checks per retraining cycle, counted =="
echo "   ocamlopt $(ocamlopt -version)"

echo
echo "-- the counting build must still hold [@zero_alloc strict]"
ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null
if ocamlopt -c -g -zero-alloc-check all -O3 -I . deep_f32.ml >z.log 2>&1; then
    echo "   deep_f32.ml holds strict WITH the counter -- the proxy does not perturb"
else
    echo "   (!) the counting build does not hold strict; the proxy is not sound:"
    sed -n '1,6p' z.log | sed 's/^/        /'
    exit 1
fi

echo
echo "-- one cycle, t_steps = Deep_f32.t_max"
ocamlopt -g -O3 -I . -o count acc.cmx acc_int.cmx deep_f32.ml count_probe.ml >/dev/null
OUT=$(./count)
B1=$(echo "$OUT" | awk '/^BUDGET1/{print $2}')
PS=$(echo "$OUT" | awk '/^PERSTEP/{print $2}')
FX=$(echo "$OUT" | awk '/^FIXED/{print $2}')
printf "   bounds checks, one optimiser step   %'d\n" "${B1}"
printf "   per additional step                 %'d\n" "${PS}"
printf "   fixed set-up per cycle              %'d\n" "${FX}"
echo
echo "   for scale: the DETECTOR costs 70,080 MAC per tick (19.8 F4, docs/MODELS.md"
echo "   3694). These are bounds checks in the RETRAINER's training cycle and are not"
echo "   the same quantity -- they are reported in the same currency, per unit of work."
echo "   No timing figure is produced. Stop 35, kept by D82.1."
echo
echo "== D82.1 count done =="
