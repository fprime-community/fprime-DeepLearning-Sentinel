#!/usr/bin/env bash
# Section 63: the owed sweep. OW1-OW5.
# (!) STOP 36: torch never acquires a path into src/ or the retraining chain. It is read
#     here as a reference and that is the only thing it may ever be.
# (!) STOP 37: an owed item is never discharged by asserting it was already fine. OW5 is
#     discharged by RE-RUNNING 60's artifact, not by citing it.
# Builds from a SELF-CONTAINED directory (52.8's structural fix).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s63"
# D82: the checked accessor, built ONCE here so every -I "${SRC}" sees the same
# one. (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc
# on its include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL
# COMPILER ERROR ("Cannot create parameter Acc.next_depth ... Misc.Fatal_error"),
# not a diagnostic. Recorded in D82 as a rule-19 finding.
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
rm -rf "${B}"; mkdir -p "${B}/boxed"; cd "${B}"
cp "${SRC}/zalloc_u.ml" "${SRC}/zalloc_u_check.ml" .

START=$(date +%s)
echo "== Section 63 =="
echo "   ocamlopt -I "${SRC}" $(ocamlopt -version)"
echo "   OW4  assume annotations in zalloc_u.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' zalloc_u.ml)"
echo "   Float_u from an INSTALLED, VERSIONED library: stdlib_upstream_compatible $(
    grep -m1 '^version' "$(ocamlopt -where)/stdlib_upstream_compatible/META" | cut -d'"' -f2)"

echo
echo "-- OW1, OW2, OW3: torch's Adam as executed"
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/s63_adam_executed.py"

echo
echo "-- OW4: the unboxed checksum holds strict and returns its float"
ocamlopt -I "${SRC}" -g -zero-alloc-check all -warn-error +a -alert @all \
    -I +stdlib_upstream_compatible "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" stdlib_upstream_compatible.cmxa \
    -o ow4 zalloc_u.ml zalloc_u_check.ml
echo "   clean: checksum_unboxed holds strict, returning float#, with no slot"
./ow4

echo
echo "-- OW4b: and the BOXED return must still FAIL, or the check proves nothing"
cat > boxed/b.ml <<'BOXED'
let n = 16
let c = Array.make (n*n) 0.0
let w = Array.make (n*n) 0.0
let[@zero_alloc strict] checksum_boxed () =
  let s = ref 0.0 in
  for i = 0 to (n*n) - 1 do
    s := !s +. (Array.unsafe_get c i) +. (Array.unsafe_get w i)
  done;
  !s
BOXED
if (cd boxed && ocamlopt -I "${SRC}" -g -zero-alloc-check all -c -I .. b.ml >b.log 2>&1); then
    echo "   (!) THE BOXED RETURN COMPILED. 47.13.4's finding does not reproduce."; exit 2
else
    echo "   rejected, as 47.13.4 found: $(grep -m1 'Error' boxed/b.log | cut -c1-90)"
fi

echo
echo "-- OW5: 57's headroom at F32, RE-RUN from 60's own artifact"
echo "   (!) 60.6 records 747 checked / 0.1882. scripts/oxcaml_s60.sh passes no stride, so"
echo "       deep_f32_check.ml:113 defaults to 1021 -> 74 checked / 0.1992. Both are shown."
bash "${ROOT}/scripts/oxcaml_s60.sh" >/dev/null 2>&1
S60="${ROOT}/oxcaml/_build/s60"
(cd "${S60}" && ./s60 all 250 101  | grep -A1 "AS4" | sed 's/^/   stride  101: /')
(cd "${S60}" && ./s60 all 250 1021 | grep -A1 "AS4" | sed 's/^/   stride 1021: /')
"${ROOT}/.venv/bin/python" - <<'PY'
for lbl, r in [("F64 (57.8)", 0.1189), ("F32 stride 101 (60.6)", 0.1882),
               ("F32 stride 1021 (default)", 0.1992)]:
    print(f"   {lbl:28s} err/allowed {r:.4f}  ->  margin {1/r:.2f}x   (band >= 1.00x)")
PY

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 63 done =="
