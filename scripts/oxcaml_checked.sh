#!/usr/bin/env bash
# The bounds-checked pass over the training modules, and the measurement that decides
# what "safe" may be claimed for the retrainer.
#
# (!) WHAT THIS EXISTS TO SETTLE. `oxcaml/retrainer/` performs 559 unchecked array
# accesses across 19 modules, including every module carrying `[@zero_alloc strict]`,
# and nothing in the record said so or costed it. The reasonable guess was that the
# two could not coexist: `strict` refuses a function whose paths reach an exceptional
# return -- 47.9 recorded the compiler's own "may allocate ON A PATH TO EXCEPTIONAL
# RETURN" -- and a bounds-checked access raises `Invalid_argument`. If that were true,
# the unchecked accesses would be forced by the annotation and there would be nothing
# to decide.
#
# (!) IT IS NOT TRUE, AND THAT IS THE FINDING. Both variants below compile the same
# modules under `-zero-alloc-check all`. OCaml's bounds-failure path raises a
# preallocated exception and does not allocate, so `strict` is satisfied either way.
# The flown object is unchecked by CHOICE, not by constraint.
#
# (!) AND NOT ONE MODULE IS EDITED TO RUN THIS. 72.4's rule governs -- "a module
# edited after measurement is a module whose figures have to be re-earned" -- so
# Stages 48-72's figures stay attached to the files they were measured on. The
# accessor is injected into COPIES, exactly as this ladder's own negative controls
# inject their allocations (`oxcaml_e3.sh:31`, `oxcaml_s60.sh:49`).
#
# No timing figure is produced. Stop 35.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"
B="${ROOT}/oxcaml/_build/checked"
rm -rf "${B}"; mkdir -p "${B}"

# Dependency order. Every module here carries at least one `[@zero_alloc strict]`.
MODULES="zalloc window56 gru_cell gru_seq gru_deep adam head57 gru_f32 cycle55 deep_f32 shadow59"

echo "== the bounds-checked pass =="
echo "   ocamlopt $(ocamlopt -version)"
echo -n "   unchecked accesses in oxcaml/retrainer/: "
grep -rhoE 'Array1?\.unsafe_(get|set)' "${SRC}"/*.ml | grep -c . || true

# Inject `open Acc` after each module's first comment block, into a COPY.
inject() {   # $1 = module name, $2 = destination dir
    "${ROOT}/.venv/bin/python" - "$1" "$2" <<'PY'
import io, pathlib, re, sys
name, dest = sys.argv[1], sys.argv[2]
src = pathlib.Path(dest).parent.parent.parent / "retrainer" / f"{name}.ml"
s = io.open(src, encoding="utf-8").read()
m = re.search(r"\*\)\s*\n", s)
i = m.end() if m and m.start() < 4000 else 0
io.open(pathlib.Path(dest) / f"{name}.ml", "w", encoding="utf-8").write(
    s[:i] + "\nopen Acc\n" + s[i:])
PY
}

for VARIANT in flight checked; do
    D="${B}/${VARIANT}"; mkdir -p "${D}"
    if [ "${VARIANT}" = flight ]; then cp "${SRC}/acc.ml" "${D}/acc.ml"
    else cp "${SRC}/acc_checked.ml" "${D}/acc.ml"; fi
    for m in ${MODULES}; do inject "$m" "${D}"; done

    echo
    echo "-- ${VARIANT}: every module under -zero-alloc-check all"
    ( cd "${D}" && ocamlopt -c -g -O3 acc.ml >/dev/null )
    FAILED=0
    for m in ${MODULES}; do
        if ( cd "${D}" && ocamlopt -c -g -zero-alloc-check all -O3 "${m}.ml" \
                 >"${m}.log" 2>&1 ); then
            printf "   %-12s strict HOLDS\n" "${m}"
        else
            printf "   %-12s REJECTED\n" "${m}"
            sed -n '1,6p' "${D}/${m}.log" | sed 's/^/        /'
            FAILED=1
        fi
    done
    if [ "${VARIANT}" = flight ] && [ "${FAILED}" -ne 0 ]; then
        echo "   (!) the FLIGHT variant must hold strict at every module: it re-exports"
        echo "       Stdlib.Array unchanged and substitutes nothing. A failure here is a"
        echo "       real regression, not a property of checking."
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# (!) The two modules the polymorphic checked accessor cannot carry, and why that
# is a property of the WRAPPER rather than of bounds-checking. `window56.ml` and
# `shadow59.ml` hold `int array`; wrapping at `'a array` defeats representation
# specialisation. Against an int-specialised accessor both hold `strict` while
# fully bounds-checked, which is what makes the conclusion general.
# ---------------------------------------------------------------------------
echo
echo "-- the int-specialised accessor, for the two modules holding int arrays"
D="${B}/checked_int"; mkdir -p "${D}"
cp "${SRC}/acc_checked_int.ml" "${D}/acc.ml"
( cd "${D}" && ocamlopt -c -g -O3 acc.ml >/dev/null )
for m in window56 shadow59; do
    inject "$m" "${D}"
    if ( cd "${D}" && ocamlopt -c -g -zero-alloc-check all -O3 "${m}.ml" \
             >"${m}.log" 2>&1 ); then
        printf "   %-12s strict HOLDS, bounds-checked\n" "${m}"
    else
        printf "   %-12s REJECTED even specialised -- report this\n" "${m}"
        sed -n '1,6p' "${D}/${m}.log" | sed 's/^/        /'
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# The other direction: a bad index must be CAUGHT in the checked build and must
# NOT be caught in the flight build. Without this the checked pass proves only
# that it compiles.
# ---------------------------------------------------------------------------
echo
echo "-- the negative direction: one deliberate out-of-range read"
for VARIANT in flight checked; do
    D="${B}/${VARIANT}_oob"; mkdir -p "${D}"
    if [ "${VARIANT}" = flight ]; then cp "${SRC}/acc.ml" "${D}/acc.ml"
    else cp "${SRC}/acc_checked.ml" "${D}/acc.ml"; fi
    cat > "${D}/oob.ml" <<'ML'
(* One read one past the end of a preallocated array. The flight accessor returns
   whatever is there; the checked accessor raises Invalid_argument. *)
open Acc
let a = Array.make 8 1.5
let () =
  let v = try Printf.sprintf "%g" (Array.unsafe_get a 8)
          with Invalid_argument m -> "CAUGHT: " ^ m in
  print_endline v
ML
    ( cd "${D}" && ocamlopt -g -O3 -o oob acc.ml oob.ml >/dev/null 2>&1 )
    printf "   %-8s reading a.(8) of an 8-element array -> %s\n" \
        "${VARIANT}" "$( cd "${D}" && ./oob )"
done

echo
echo "== the bounds-checked pass done =="
