#!/usr/bin/env bash
# D82's measurement: the flown retrainer is bounds-checked, and the annotation still
# holds. Both arms, and the direction that shows the checks do something.
#
# (!) WHAT THIS SETTLED. Until 2026-09-22 this tree performed 559 unchecked array
# accesses across 19 modules, including every module carrying `[@zero_alloc strict]`,
# and nothing in the record said so. The available defence was that the annotation
# forced it -- `strict` refuses a function whose paths reach an exceptional return
# (47.9: "may allocate ON A PATH TO EXCEPTIONAL RETURN") and a checked access raises.
#
# Measured, that defence does not hold: OCaml's bounds-failure path raises a
# PREALLOCATED exception and allocates nothing. D82 therefore takes the other choice
# and the flown modules are compiled against the checked accessor.
#
# This script is what keeps that a measurement. Arm A compiles the annotated modules
# as they now fly -- `acc.ml` and `acc_int.ml`, every access checked. Arm B compiles
# the same modules against `acc_unchecked.ml`, the comparison arm. Both must hold
# `[@zero_alloc strict]`; if arm A ever stops holding, the checks have become
# unaffordable and D82 has to be revisited rather than quietly reverted.
#
# No timing figure is produced for what the checks cost. Stop 35.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"
# D82: the checked accessor, built ONCE here so every -I "${SRC}" sees the same
# one. (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc
# on its include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL
# COMPILER ERROR ("Cannot create parameter Acc.next_depth ... Misc.Fatal_error"),
# not a diagnostic. Recorded in D82 as a rule-19 finding.
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
B="${ROOT}/oxcaml/_build/checked"
rm -rf "${B}"; mkdir -p "${B}"

# Every module carrying at least one `[@zero_alloc strict]`, in dependency order.
MODULES="zalloc window56 gru_cell gru_seq gru_deep adam head57 gru_f32 cycle55 deep_f32 shadow59"

echo "== D82: the flown retrainer is bounds-checked =="
echo "   ocamlopt -I "${SRC}" $(ocamlopt -version)"
echo -n "   direct Array1.unsafe_* left in the retrainer's own modules: "
grep -rhoE 'Array1?\.unsafe_(get|set)' "${SRC}"/*.ml \
  | grep -c . || echo 0
echo "   (the accessor files hold the primitive by construction; they are the instrument)"

run_arm() {   # $1 = label, $2 = file to install as acc.ml, $3 = file as acc_int.ml
    local label="$1" accfile="$2" accint="$3"
    local D="${B}/${label}"; mkdir -p "${D}"
    cp "${SRC}/${accfile}" "${D}/acc.ml"
    cp "${SRC}/${accint}"  "${D}/acc_int.ml"
    for m in ${MODULES}; do cp "${SRC}/${m}.ml" "${D}/"; done
    ( cd "${D}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
    echo
    echo "-- ${label}: every annotated module under -zero-alloc-check all"
    local failed=0
    for m in ${MODULES}; do
        if ( cd "${D}" && ocamlopt -I "${SRC}" -c -g -zero-alloc-check all -O3 -I . "${m}.ml" \
                 >"${m}.log" 2>&1 ); then
            printf "   %-12s strict HOLDS\n" "${m}"
        else
            printf "   %-12s REJECTED\n" "${m}"
            sed -n '1,6p' "${D}/${m}.log" | sed 's/^/        /'
            failed=1
        fi
    done
    return ${failed}
}

# Arm A: as flown. `window56` and `shadow59` hold `int array` and take the
# int-specialised accessor -- wrapping `Array` at `'a array` defeats representation
# specialisation and `strict` refuses the caller, which is why there are two.
run_arm "as-flown-checked" acc.ml acc_int.ml || {
    echo "   (!) THE FLOWN CONFIGURATION DOES NOT HOLD strict. D82's premise is gone;"
    echo "       report it rather than reverting quietly."; exit 1; }

# Arm B: the comparison. `acc_unchecked.ml` re-exports Stdlib.Array unchanged, so it
# serves both the polymorphic and the int modules and no second file is needed.
run_arm "comparison-unchecked" acc_unchecked.ml acc_unchecked.ml || {
    echo "   (!) the UNCHECKED arm does not hold strict -- that is a regression in the"
    echo "       modules, not a property of checking."; exit 1; }

# ---------------------------------------------------------------------------
# The direction that shows the checks are not decoration.
# ---------------------------------------------------------------------------
echo
echo "-- the negative direction: one deliberate out-of-range read"
for arm in checked unchecked; do
    D="${B}/${arm}_oob"; mkdir -p "${D}"
    if [ "${arm}" = checked ]; then cp "${SRC}/acc.ml" "${D}/acc.ml"
    else cp "${SRC}/acc_unchecked.ml" "${D}/acc.ml"; fi
    cat > "${D}/oob.ml" <<'ML'
open Acc
let a = Array.make 8 1.5
let () =
  let v = try Printf.sprintf "%g" (Array.unsafe_get a 8)
          with Invalid_argument m -> "CAUGHT: " ^ m in
  print_endline v
ML
    ( cd "${D}" && ocamlopt -I "${SRC}" -g -O3 -o oob acc.ml oob.ml >/dev/null 2>&1 )
    printf "   %-10s reading a.(8) of an 8-element array -> %s\n" \
        "${arm}" "$( cd "${D}" && ./oob )"
done

echo
echo "== D82 measurement done =="
