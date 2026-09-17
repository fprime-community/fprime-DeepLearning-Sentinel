#!/usr/bin/env bash
# Section 59 / E5-a: where the shadow's weights live. MW1-MW6.
# (!) STOP 29: the shape fields are never written. (!) STOP 30: format_version stays 1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s59"
rm -rf "${B}"; mkdir -p "${B}"; cd "${B}"
cp "${SRC}/shadow59.ml" "${SRC}/shadow59_check.ml" .

START=$(date +%s)
echo "== Section 59 / E5-a =="
echo "   ocamlopt $(ocamlopt -version)"
echo "   MW1  assume annotations in shadow59.ml: $(grep -cE '\[@+zero_alloc[^]]*assume' shadow59.ml)"
echo "   C externals called: $(grep -cE 'external .*= "[a-z]' shadow59.ml)"

echo
echo "-- MW1: the writer must compile clean under -zero-alloc-check all"
ocamlopt -g -zero-alloc-check all -warn-error +a -alert @all -o s59 shadow59.ml shadow59_check.ml
echo "   clean: write_shadow, crc_range, get_u32 and put_u32 hold [@zero_alloc strict]"

FLY="${ROOT}/flight/test/vectors/p1.bin"
echo
echo "-- the write"
./s59 "${FLY}" "${B}/shadow.bin"

echo
echo "-- MW2, MW4, MW5, MW6: the C++ reader, linking flight/ itself"
clang++ -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off \
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -O2 \
    -I"${ROOT}/flight/include" "${SRC}/shadow59_load.cpp" \
    "${ROOT}"/flight/src/*.cpp -o s59_load
./s59_load "${FLY}" "${B}/shadow.bin"

echo
echo "-- MW3: RoundTripDump must re-emit the shadow byte-identically"
clang++ -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off \
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -O2 \
    -I"${ROOT}/flight/include" -I"${ROOT}/flight/test" \
    "${ROOT}/flight/test/RoundTripDump.cpp" "${ROOT}"/flight/src/*.cpp -o rtd
./rtd "${B}/shadow.bin" "${B}/shadow.rt" >/dev/null
if cmp -s "${B}/shadow.bin" "${B}/shadow.rt"; then
    echo "   MW3  re-emitted byte-identically"
else
    echo "   MW3  FAIL: re-emitted bytes differ"; exit 1
fi

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 59 done =="
