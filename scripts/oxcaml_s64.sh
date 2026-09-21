#!/usr/bin/env bash
# Section 64: E5-f's sanity band, and 60's gradients exhaustively. EX1, SR1-SR4.
# (!) STOP 38: no target, budgeted or desired alarm rate is an input to anything here.
# (!) STOP 39: the 2.0% margin, the 0.060 drift scalar and 49.2's four constants do not move.
# (!) STOP 40: src/sentinel_eval/synthetic.py is not modified; 64.3 records why.
# (!) deep_f32_check.ml is NOT modified -- it is 60.6's cited artifact. The exhaustive
#     module is a new file beside it, validated against it before it is trusted.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
eval "$(opam env --switch=5.2.0+ox --set-switch)"
SRC="${ROOT}/oxcaml/retrainer"; B="${ROOT}/oxcaml/_build/s64ex"
SHARDS="${SHARDS:-10}"
rm -rf "${B}"; mkdir -p "${B}"; cd "${B}"
cp "${SRC}/deep_f32.ml" "${SRC}/deep_f32_exhaustive.ml" .

START=$(date +%s)
echo "== Section 64 =="
echo "   ocamlopt $(ocamlopt -version)"
ocamlopt -g -O3 -o s64ex deep_f32.ml deep_f32_exhaustive.ml

echo
echo "-- EX1 validation: shard 0 of 1021 covers exactly the indices deep_f32_check samples"
echo "   at its default stride, so it must return the runner's row to every digit."
./s64ex 0 1021 250 | sed 's/^/   /'
echo "   deep_f32_check.ml at stride 1021 reports: 74 checked, 5.725114e-05, 0.1992"

echo
echo "-- EX1: every one of the 75,360 indices, ${SHARDS} shards"
rm -f shards.txt
for k in $(seq 0 $((SHARDS - 1))); do ./s64ex "$k" "${SHARDS}" 250 >> shards.txt & done
wait
sort -k2 -n shards.txt | sed 's/^/   /'
"${ROOT}/.venv/bin/python" - <<'PY'
best = (0.0, None, None); tot = out = 0
for ln in open("shards.txt"):
    d = dict(kv.split("=") for kv in ln.split() if "=" in kv)
    tot += int(d["checked"]); out += int(d["outside"])
    r = float(d["ratio"])
    if r > best[0]: best = (r, int(d["idx"]), float(d["worst"]))
print(f"   EXHAUSTIVE: checked {tot:,}  outside {out}  "
      f"worst ratio {best[0]:.6f} at idx {best[1]}  worst |err| {best[2]:.9e}")
print(f"   margin {1/best[0]:.3f}x   (60.7 quotes 5.3x from a 747-index sample; "
      f"it is understated by {best[0]/0.1882:.2f}x)")
print(f"   EX1 -> {'HOLD' if out == 0 else 'FAIL'}  (band: 0 outside of 75,360)")
PY

echo
echo "-- SR1 to SR4: the pre-launch sanity band, both arms"
PYTHONPATH="${ROOT}/src" "${ROOT}/.venv/bin/python" "${ROOT}/scripts/s64_sanity_band.py" || true

echo
echo "   wall clock $(( $(date +%s) - START )) s"
echo "== Section 64 done =="
