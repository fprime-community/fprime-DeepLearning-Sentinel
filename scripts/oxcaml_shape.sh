#!/usr/bin/env bash
# D84 / D83.1 route 1: the retraining cycle at a MISSION's shape, generated, gated
# and built -- the object a mission's retrainer deployment links.
#
#   bash scripts/oxcaml_shape.sh --channels 8 --predictions 10
#   EX1=1 bash scripts/oxcaml_shape.sh --channels 8 --predictions 10   # + gradients
#
# Output: oxcaml/_build/shape-c<C>-p<P>/ with the generated deep_f32.ml, the
# generated sentinel_cycle_shape.h, cycle_complete.o and gates.log. The F' build
# selects that directory by SENTINEL_RETRAINER_CHANNELS / _PREDICTIONS
# (fprime/library.cmake), and refuses if it is absent or disagrees.
#
# (!) A GENERATED SHAPE RE-EARNS ITS OWN FIGURES; IT INHERITS NONE. D83.1's cost
# column for route 1, verbatim: "the new shape must re-earn its OWN figures -- its
# gradient check and its strict result -- and HO1 must be re-run against a
# mission-shaped flying file." So every gate below runs against THIS shape's
# generated source, and the script exits non-zero on the first that fails:
#
#   SG1  strict sites in the generated deep_f32.ml equal the template's; assume 0
#   SG2  -zero-alloc-check all over deep_f32, cycle_c, shadow59, shadow_c
#   SG3  a deliberate allocation in the generated cycle must FAIL the build (AS7a)
#   SG4  an out-of-range read inside a flown function RAISES (D82: bounds-checked)
#   SG5  the object links: -output-complete-obj, the same module set as s61's tail
#   SG6  HO1 at this shape: the cycle's own weights -> candidate -> Detector::load,
#        and a flying file at another shape is refused
#   EX1  (EX1=1) every parameter index against 51's criterion, SHARDS processes,
#        after shard 0 of 1021 reproduces deep_f32_check.ml at the same shape
#
# (!) NO TIMING FIGURE IS PRODUCED OR TO BE QUOTED. Stop 35, kept by D82.1.
# (!) Every status below is taken directly, never through a pipe (D83.2).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() { echo "usage: bash scripts/oxcaml_shape.sh --channels C --predictions P" >&2; exit 2; }
C=""; P=""
while [ $# -gt 0 ]; do
    case "$1" in
        --channels) C="${2:-}"; shift 2 ;;
        --predictions) P="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
[ -n "${C}" ] && [ -n "${P}" ] || usage

export OPAMROOT="${ROOT}/oxcaml/.opam"
if [ ! -x "${OPAMROOT}/5.2.0+ox/bin/ocamlopt" ]; then
    echo "oxcaml_shape: no OxCaml switch at ${OPAMROOT}; run scripts/oxcaml_setup.sh" >&2
    exit 1
fi
PY="${ROOT}/.venv/bin/python"
if [ ! -x "${PY}" ]; then
    echo "oxcaml_shape: no ${PY}; HO1 builds its flying file with the ground toolkit." >&2
    echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements-toolkit.txt" >&2
    exit 1
fi
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"
# OXCAML_SHAPE_OUT lets a guard build elsewhere, so a test run never deletes the
# object an F' build cache links.
B="${OXCAML_SHAPE_OUT:-${ROOT}/oxcaml/_build/shape-c${C}-p${P}}"

# The generator refuses a shape outside the box before anything is deleted.
python3 "${ROOT}/scripts/oxcaml_shape.py" --channels "${C}" --predictions "${P}" \
    --out "${B}.check" >/dev/null
rm -rf "${B}.check" "${B}"
mkdir -p "${B}"
exec > >(tee "${B}/gates.log") 2>&1

echo "== D84: the retraining cycle at channels ${C}, predictions ${P} =="
python3 "${ROOT}/scripts/oxcaml_shape.py" --channels "${C}" --predictions "${P}" --out "${B}"
echo "   ocamlopt $(ocamlopt -version)"

# D82: the checked accessor, built ONCE so every -I "${SRC}" sees the same one.
# (!) NEVER also pass "${SRC}/acc.ml" to a command that has a compiled Acc on its
# include path: ocamlopt 5.2.0+ox answers that duplicate with an INTERNAL COMPILER
# ERROR, not a diagnostic (D82 c.3).
( cd "${SRC}" && ocamlopt -c -g -O3 acc.ml acc_int.ml >/dev/null )
cd "${B}"
# Everything the object needs, copied beside the generated deep_f32.ml so the
# generated interface is the one found first (ocamlopt searches the current
# directory before any -I) and nothing is written back into ${SRC}.
cp "${SRC}/cycle_c.ml" "${SRC}/shadow59.ml" "${SRC}/shadow_c.ml" "${SRC}/retrainer.ml" \
   "${SRC}/cycle_stubs.c" "${SRC}/shadow_stubs.c" "${SRC}/retrainer_stubs.c" \
   "${SRC}/sentinel_cycle.h" "${SRC}/sentinel_shadow.h" "${SRC}/sentinel_retrainer.h" .

echo
echo "-- SG1: the annotation count is the template's, and nothing is assumed"
STRICT_T=$(grep -cE '\[@+zero_alloc strict\]' "${SRC}/deep_f32.ml")
STRICT_G=$(grep -cE '\[@+zero_alloc strict\]' deep_f32.ml)
ASSUME=$(cat deep_f32.ml cycle_c.ml shadow59.ml shadow_c.ml | grep -cE '\[@+zero_alloc[^]]*assume' || true)
echo "   strict sites: template ${STRICT_T}, generated ${STRICT_G}; assume: ${ASSUME}"
if [ "${STRICT_T}" != "${STRICT_G}" ] || [ "${ASSUME}" != "0" ]; then
    echo "   (!) SG1 FAILED"; exit 1
fi
if ! grep -q '^open Acc$' deep_f32.ml; then
    echo "   (!) SG1 FAILED: the generated deep_f32.ml does not open the checked accessor"; exit 1
fi

echo
echo "-- SG2: the flown modules at this shape under -zero-alloc-check all"
for m in deep_f32 cycle_c shadow59 shadow_c; do
    ocamlopt -I "${SRC}" -c -g -zero-alloc-check all -warn-error +a -alert @all -O3 -I . "${m}.ml"
    echo "   ${m}: strict HOLDS"
done

echo
echo "-- SG3: a deliberate allocation in the generated cycle must FAIL the build"
mkdir -p sg3
python3 - deep_f32.ml sg3/deep_f32.ml <<'PY'
import io, sys
s = io.open(sys.argv[1], encoding="utf-8").read()
old = "      save_best ();                        (* D73 c.2: bookkeeping, not control flow *)"
new = ("      save_best ();\n"
       "      ignore (Sys.opaque_identity (Array.make 4 0.0));")
assert s.count(old) == 1, f"anchor matched {s.count(old)} times"
io.open(sys.argv[2], "w", encoding="utf-8").write(s.replace(old, new))
PY
if (cd sg3 && ocamlopt -I "${SRC}" -c -g -zero-alloc-check all -O3 deep_f32.ml >sg3.log 2>&1); then
    echo "   (!) BUILT -- the gate cannot fail. SG3 FAILED."; exit 1
else
    echo "   rejected: $(grep -m1 -oE 'called function may allocate.*' sg3/sg3.log || echo 'see sg3/sg3.log')"
fi

echo
echo "-- SG4: an out-of-range read inside a flown function RAISES"
mkdir -p sg4
cat > sg4/probe.ml <<'ML'
(* forward_deep at t_max + 1 walks past every tape; D82's checked accessor must
   raise rather than read someone else's memory. *)
let () =
  match Deep_f32.forward_deep (Deep_f32.t_max + 1) with
  | () -> print_endline "NOT RAISED"
  | exception Invalid_argument m -> print_endline ("RAISED: Invalid_argument " ^ m)
ML
# (!) -I .. BEFORE -I "${SRC}": ${SRC} holds the MAXIMA deep_f32.cmi from every other
# runner, and the first interface found wins. The first run had them the other way
# round and ocamlopt refused the link as "inconsistent assumptions", which is the
# right refusal -- and the reason this order is not cosmetic.
( cd sg4 && ocamlopt -I .. -I "${SRC}" -g -O3 -o probe "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" \
      ../deep_f32.cmx probe.ml >probe.log 2>&1 )
SG4_OUT="$(cd sg4 && ./probe)"
echo "   forward_deep (t_max + 1) -> ${SG4_OUT}"
case "${SG4_OUT}" in RAISED:*) ;; *) echo "   (!) SG4 FAILED"; exit 1 ;; esac

echo
echo "-- SG5: the component's object at this shape (E1's surface, 61's cycle, 72's shadow)"
ocamlopt -I "${SRC}" -output-complete-obj -O3 -o cycle_complete.o \
    -warn-error +a -alert @all \
    -I . "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" \
    retrainer.ml retrainer_stubs.c \
    deep_f32.ml cycle_c.ml cycle_stubs.c \
    shadow59.ml shadow_c.ml shadow_stubs.c
echo "   cycle_complete.o ($(wc -c < cycle_complete.o | tr -d ' ') bytes)"

echo
echo "-- SG6: HO1 at this shape"
FFLAGS="-Wold-style-cast -pedantic -Wall -Wextra -Wconversion -Wdouble-promotion -Wshadow -Werror"
# The other shape: the maxima, unless this IS the maxima, then 8 channels.
if [ "${C}" = "16" ] && [ "${P}" = "10" ]; then OC=8; OP=10; else OC=16; OP=10; fi
PYTHONPATH="${ROOT}/src" "${PY}" "${ROOT}/scripts/s72_flying_file.py" flying.bin \
    --channels "${C}" --predictions "${P}"
PYTHONPATH="${ROOT}/src" "${PY}" "${ROOT}/scripts/s72_flying_file.py" other.bin \
    --channels "${OC}" --predictions "${OP}"
mkdir -p flightobj
( cd flightobj && clang++ -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off \
    ${FFLAGS} -O2 -I"${ROOT}/flight/include" -c "${ROOT}"/flight/src/*.cpp )
clang++ -std=c++14 -fno-exceptions -fno-rtti ${FFLAGS} -O2 \
    -I. -I"${ROOT}/flight/include" -I"$(ocamlopt -where)" \
    -c "${SRC}/shape_harness.cpp" -o shape_harness.o
clang++ -o shape_harness shape_harness.o cycle_complete.o flightobj/*.o -lm
./shape_harness flying.bin other.bin candidate.bin

if [ "${EX1:-0}" = "1" ]; then
    SHARDS="${SHARDS:-10}"
    echo
    echo "-- EX1 at this shape: every parameter index against 51's criterion"
    mkdir -p ex1
    cp deep_f32.ml ex1/
    cp "${SRC}/deep_f32_check.ml" "${SRC}/deep_f32_exhaustive.ml" "${SRC}/sqrtf_ref.c" ex1/
    ( cd ex1 && ocamlopt -I "${SRC}" -g -O3 -o check sqrtf_ref.c -I . \
          "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" deep_f32.ml deep_f32_check.ml )
    ( cd ex1 && ocamlopt -I "${SRC}" -g -O3 -o exhaustive -I . \
          "${SRC}/acc.cmx" "${SRC}/acc_int.cmx" deep_f32.ml deep_f32_exhaustive.ml )
    echo "   validation: deep_f32_check.ml at stride 1021 and exhaustive shard 0 of 1021"
    ( cd ex1 && ./check all 250 1021 > check.txt )
    grep -E 'AS4' -A1 ex1/check.txt | sed 's/^/   /'
    ( cd ex1 && ./exhaustive 0 1021 250 > shard0.txt )
    sed 's/^/   /' ex1/shard0.txt
    "${PY}" - ex1/check.txt ex1/shard0.txt <<'PY'
import re, sys
chk = open(sys.argv[1]).read()
m = re.search(r"outside (\d+) of (\d+) checked; worst \|err\| ([0-9.e+-]+); "
              r"worst err/allowed ([0-9.]+)", chk)
d = dict(kv.split("=") for kv in open(sys.argv[2]).read().split() if "=" in kv)
ok = (m is not None and int(m.group(2)) == int(d["checked"])
      and int(m.group(1)) == int(d["outside"])
      and abs(float(m.group(3)) - float(d["worst"])) <= 1e-6 * max(float(d["worst"]), 1e-30)
      and abs(float(m.group(4)) - float(d["ratio"])) < 5e-5)
print(f"   shard 0 of 1021 reproduces deep_f32_check.ml at this shape: {ok}")
sys.exit(0 if ok else 1)
PY
    echo "   ${SHARDS} shards"
    rm -f ex1/shards.txt
    PIDS=""
    for k in $(seq 0 $((SHARDS - 1))); do
        ( cd ex1 && ./exhaustive "$k" "${SHARDS}" 250 > "shard_${k}.txt" ) &
        PIDS="${PIDS} $!"
    done
    for pid in ${PIDS}; do wait "${pid}"; done
    cat ex1/shard_*.txt > ex1/shards.txt
    sort -k2 -n ex1/shards.txt | sed 's/^/   /'
    N_PARAMS=$(grep -oE 'SENTINEL_CYCLE_N_PARAMS [0-9]+' sentinel_cycle_shape.h | grep -oE '[0-9]+$')
    "${PY}" - ex1/shards.txt "${N_PARAMS}" <<'PY'
import sys
n_params = int(sys.argv[2])
best = (0.0, None, None); tot = out = 0
for ln in open(sys.argv[1]):
    d = dict(kv.split("=") for kv in ln.split() if "=" in kv)
    tot += int(d["checked"]); out += int(d["outside"])
    r = float(d["ratio"])
    if r > best[0]: best = (r, int(d["idx"]), float(d["worst"]))
print(f"   EXHAUSTIVE: checked {tot:,}  outside {out}  "
      f"worst ratio {best[0]:.6f} at idx {best[1]}  worst |err| {best[2]:.9e}")
print(f"   margin {1/best[0]:.3f}x")
held = (out == 0 and tot == n_params)
print(f"   EX1 -> {'HOLD' if held else 'FAIL'}  (band: 0 outside of {n_params:,}, "
      f"every index checked)")
sys.exit(0 if held else 1)
PY
fi

echo
echo "== shape c${C}-p${P}: every gate passed =="
