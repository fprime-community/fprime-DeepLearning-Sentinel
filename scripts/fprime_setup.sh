#!/usr/bin/env bash
#
# Reproduce the work item 9 F' toolchain from nothing.
#
# F' is pinned at v4.3.0 (docs/DECISIONS.md D31). Everything this script creates
# is gitignored and disposable: the framework checkout, the tool virtualenv and
# every build cache. Only fprime/'s own sources are tracked.
#
# It lives inside the repository because this machine keeps everything for this
# project inside this directory; tests/test_no_local_persistence.py exempts the
# two subtrees below and asserts the exemption stays narrow.
#
# Idempotent: safe to re-run. Prints every version it installed at the end, which
# is what docs/FPRIME.md records.
set -euo pipefail

FPRIME_TAG="v4.3.0"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${ROOT}/fprime"
CHECKOUT="${PROJECT}/lib/fprime"
VENV="${PROJECT}/fprime-venv"

echo "== fprime-sentinel: F' ${FPRIME_TAG} toolchain =="

# 1. The framework checkout. Shallow, with submodules -- googletest is one, and
#    the unit-test build needs it.
if [ -d "${CHECKOUT}/.git" ]; then
    echo "-- checkout present: $(git -C "${CHECKOUT}" describe --tags 2>/dev/null || echo unknown)"
else
    echo "-- cloning nasa/fprime ${FPRIME_TAG}"
    mkdir -p "${PROJECT}/lib"
    git clone --branch "${FPRIME_TAG}" --depth 1 \
        --recurse-submodules --shallow-submodules \
        https://github.com/nasa/fprime.git "${CHECKOUT}"
fi

# 2. The tool virtualenv, from F's OWN requirements.txt rather than a list of our
#    own. F' pins cmake and ninja as pip packages, so activating this venv puts
#    the versions F' tests against ahead of anything Homebrew installed -- which
#    matters, because Homebrew's cmake is 4.x and F' pins 3.26.
if [ -x "${VENV}/bin/fprime-util" ]; then
    echo "-- venv present"
else
    echo "-- creating venv and installing F's pinned requirements"
    python3 -m venv "${VENV}"
    "${VENV}/bin/pip" install --quiet --upgrade pip
    "${VENV}/bin/pip" install --quiet -r "${CHECKOUT}/requirements.txt"
fi

# 3. clang-tidy. Not shipped by F' -- docs/DECISIONS.md D31 consequence 5 said
#    "with the F' toolchain", which was wrong about where the tool comes from.
#    D37's sibling correction; recorded in docs/MODELS.md 20.2 item 10.
if command -v clang-tidy >/dev/null 2>&1 || [ -x /opt/homebrew/opt/llvm/bin/clang-tidy ]; then
    echo "-- clang-tidy present"
else
    echo "-- clang-tidy absent; install with: brew install llvm"
fi

echo
echo "== versions =="
{
  echo "fprime            $(git -C "${CHECKOUT}" describe --tags 2>/dev/null || echo unknown)"
  echo "fprime commit     $(git -C "${CHECKOUT}" rev-parse HEAD)"
  echo "python            $("${VENV}/bin/python" --version 2>&1 | awk '{print $2}')"
  echo "cmake             $("${VENV}/bin/cmake" --version | head -1 | awk '{print $3}')"
  echo "ninja             $("${VENV}/bin/ninja" --version)"
  echo "fpp               $("${VENV}/bin/fpp-check" --help 2>&1 | head -1 | awk '{print $2}')"
  "${VENV}/bin/pip" list 2>/dev/null | awk '/^fprime/ {printf "%-17s %s\n", $1, $2}'
  echo "clang++           $(clang++ --version | head -1)"
  echo "clang-tidy        $(/opt/homebrew/opt/llvm/bin/clang-tidy --version 2>/dev/null | head -1 || echo absent)"
} | sed 's/^/  /'

echo
echo "To use it:  . fprime/fprime-venv/bin/activate  &&  cd fprime"
