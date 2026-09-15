#!/usr/bin/env bash
# Build the OxCaml toolchain for work item 47 / E1, and print every version it found.
#
# Idempotent. Same shape and same reasons as `scripts/fprime_setup.sh`: the toolchain
# lives INSIDE this directory because everything for this project does, it is gitignored,
# and it is rebuilt from nothing by this script rather than committed -- a second copy in
# the repository would drift from upstream silently, which is the failure D16 exists to
# prevent (`docs/FPRIME.md` 2).
#
# (!) OPAMROOT is forced inside the repository. opam defaults to ~/.opam, which is outside
# this directory and would leave several GiB of this project's dependencies somewhere
# nothing here records or cleans up.
#
# `docs/MODELS.md` 47.8 is the record this script fills. `docs/MODELS.md` 47.9 X12 is the
# pre-registered band on what it costs: under 90 min wall clock and under 12 GiB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAMROOT="${ROOT}/oxcaml/.opam"
SWITCH="5.2.0+ox"
OX_REPO="git+https://github.com/oxcaml/opam-repository.git"

mkdir -p "${ROOT}/oxcaml"

echo "== oxcaml_setup =="
echo "   OPAMROOT  ${OPAMROOT}"
echo "   switch    ${SWITCH}"

command -v opam >/dev/null || { echo "opam is not on PATH; brew install opam"; exit 1; }

# (!) autoconf and automake are build dependencies of the ocaml-variants package the
# switch invariant names, and opam will ABORT rather than install them: the first run of
# this script on 2026-09-15 exited 10 after 13 s for exactly this reason. Checked here so
# the failure is named up front rather than met thirteen seconds into a long step.
for dep in autoconf automake; do
  command -v "$dep" >/dev/null || { echo "${dep} is not on PATH; brew install autoconf automake"; exit 1; }
done

if [ ! -d "${OPAMROOT}" ] || [ ! -f "${OPAMROOT}/config" ]; then
  echo "-- opam init (bare, no shell setup: nothing outside this directory is written)"
  opam init --bare --no-setup -y
else
  echo "-- opam root present, skipping init"
fi

if opam switch list --short 2>/dev/null | grep -qx "${SWITCH}"; then
  echo "-- switch ${SWITCH} present, skipping create"
else
  echo "-- opam switch create ${SWITCH} (this is the long step)"
  opam switch create "${SWITCH}" --repos "ox=${OX_REPO},default" -y
fi

eval "$(opam env --switch="${SWITCH}" --set-switch)"

echo
echo "== versions, as measured $(date -u +%Y-%m-%d) =="
printf "  %-20s %s\n" "host"    "$(uname -sm), $(sysctl -n hw.ncpu 2>/dev/null || nproc) cores"
printf "  %-20s %s\n" "opam"    "$(opam --version)"
printf "  %-20s %s\n" "switch"  "${SWITCH}"
printf "  %-20s %s\n" "ocaml"   "$(ocamlopt -version 2>/dev/null || echo absent)"
printf "  %-20s %s\n" "dune"    "$(dune --version 2>/dev/null || echo 'absent -- opam install dune')"
printf "  %-20s %s\n" "ocamlfind" "$(ocamlfind -version 2>/dev/null || echo absent)"
printf "  %-20s %s\n" "libdir"  "$(ocamlopt -where 2>/dev/null || echo absent)"
echo
echo "  zero_alloc support:"
if ocamlopt -help 2>&1 | grep -q "zero-alloc-check"; then
  ocamlopt -help 2>&1 | grep -A1 "zero-alloc-check" | head -4 | sed 's/^/    /'
else
  echo "    (!) -zero-alloc-check NOT offered by this ocamlopt -- 47.7 C1 is in doubt"
fi
echo
echo "  disk: $(du -sh "${OPAMROOT}" 2>/dev/null | cut -f1) under ${OPAMROOT}"
echo "== done =="
