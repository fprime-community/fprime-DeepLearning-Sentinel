#!/usr/bin/env bash
# Section 54: the cell in float32. FT1-FT6.
# (!) STOP 21: only 54.2's listed primitives plus %divfloat32, which 54.2a added after the
# requirement was DERIVED from flight/src/Gru.cpp:31,34. Nothing else.
# Builds from a SELF-CONTAINED directory (52.8's structural fix).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s54"
rm -rf "${B}"; mkdir -p "${B}/v5"; cd "${B}"
cp "${SRC}/gru_f32.ml" "${SRC}/gru_cell.ml" "${SRC}/gru_f32_check.ml" .

START=$(date +%s)
echo "== Section 54 =="
echo "   ocamlopt $(ocamlopt -version)"
echo "   FT4  assume annotations in gru_f32.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' gru_f32.ml)"
echo "   primitives used: $(grep -oE '"%[a-z0-9_]+"' gru_f32.ml | sort -u | tr '\n' ' ')"

echo
echo "-- FT1, FT2: the float32 forward and backward must compile clean"
ocamlopt -g -zero-alloc-check all -warn-error +a -alert @all -o s54 \
    gru_f32.ml gru_cell.ml gru_f32_check.ml
echo "   clean: forward, backward and zero_grads hold with Bigarray float32 storage"

echo
echo "-- FT6: the F32 cell against 48's F64 cell (and the backward is RUN, not just built)"
./s54 "${B}/h_new_f32.txt"

echo
echo "-- FT3: against flight/'s own Gru::step, both at F32"
clang++ -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off \
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -O2 \
    -I"${ROOT}/flight/include" "${SRC}/f32_flight_cell.cpp" \
    "${ROOT}"/flight/src/Gru.cpp "${ROOT}"/flight/src/ModelFile.cpp \
    "${ROOT}"/flight/src/Crc32.cpp "${ROOT}"/flight/src/Status.cpp -o s54_ft3
set +e; ./s54_ft3; FT3=$?; set -e

echo
echo "-- FT5: a deliberate allocation in the float32 forward must FAIL"
python3 - gru_f32.ml v5/gru_f32.ml <<'PY'
import sys, io
s = io.open(sys.argv[1], encoding="utf-8").read()
old = "    set h_new i (add (mul (sub hi ni) zi) ni)"
new = ("    set h_new i (add (mul (sub hi ni) zi) ni);\n"
       "    ignore (Sys.opaque_identity (Array.make 4 0.0))")
assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
io.open(sys.argv[2], "w", encoding="utf-8").write(s.replace(old, new))
print("   variant written: Array.make inside the float32 forward")
PY
if (cd v5 && ocamlopt -g -zero-alloc-check all -c gru_f32.ml >a.log 2>&1); then
    echo "   (!) BUILT -- the gate cannot fail. FT5 FAILED."; exit 1
else
    echo "   rejected: $(grep -m1 -oE 'called function may allocate.*' v5/a.log)"
fi

echo
echo "   wall clock $(( $(date +%s) - START )) s   (FT3 exit ${FT3})"
echo "== Section 54 done =="
