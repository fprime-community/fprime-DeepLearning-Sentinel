# Work item 8's build. Work item 9 builds the same sources under F' with CMake.
#
# The flag set is pre-registered in `docs/MODELS.md` 19 and specified in
# `docs/MODEL_FILE.md` 9. It is not a build-system detail:
#
#   -fno-exceptions -fno-rtti   F' CPP-25. The loader returns a status and
#                               cannot throw, by construction.
#   -ffp-contract=off           forbids contracting a*b+c into an FMA, which
#                               rounds once instead of twice and would silently
#                               break equivalence with the NumPy reference.
#   -std=c++14                  F' CPP-5.
#   -Werror at -Wall -Wextra -Wpedantic -Wconversion -Wshadow
#                               the build must be silent, not merely successful.
#                               -Wconversion in particular makes every F64 -> F32
#                               narrowing explicit at the point the reference
#                               casts, so the compiler holds the dtype map.
#
# -ffast-math and -Ofast are forbidden outright: they license reassociation and
# reciprocal substitution, and the second would break the sigmoid's x >= 0 branch.

CXX      ?= clang++
STD      := -std=c++14
SAFETY   := -fno-exceptions -fno-rtti
DETERM   := -ffp-contract=off
WARN     := -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
OPT      ?= -O2
CXXFLAGS := $(STD) $(SAFETY) $(DETERM) $(WARN) $(OPT) -Iinclude

BUILD    := build
CORE_SRC := src/Crc32.cpp src/Status.cpp src/ModelFile.cpp src/Gru.cpp \
            src/Ewma.cpp src/Detector.cpp src/Baseline.cpp \
            src/TrailingWindow.cpp src/DynamicThreshold.cpp \
            src/DerivativeStream.cpp
CORE_OBJ := $(patsubst src/%.cpp,$(BUILD)/%.o,$(CORE_SRC))

TESTS    := GoldenVectors RefusalTests DeterminismTest RoundTripDump Footprint \
            BaselineVectors TrailingVectors ThresholdVectors
BINS     := $(patsubst %,$(BUILD)/%,$(TESTS))

.PHONY: all test lint clean

all: $(BINS)

$(BUILD)/%.o: src/%.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%: test/%.cpp $(CORE_OBJ)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(CORE_OBJ) -o $@

test: $(BINS)
	@rm -f $(BUILD)/determinism.crc
	@$(BUILD)/Footprint
	@$(BUILD)/RefusalTests
	@$(BUILD)/DeterminismTest
	@$(BUILD)/DeterminismTest
	@$(BUILD)/GoldenVectors
	@$(BUILD)/BaselineVectors
	@$(BUILD)/TrailingVectors
	@$(BUILD)/ThresholdVectors
	@echo "== round trip =="
	@set -e; ok=0; for t in g1 g2 g3 g4_0 g4_1 g4_2 g4_3; do \
	    if [ -f test/vectors/$$t.bin ]; then \
	        $(BUILD)/RoundTripDump test/vectors/$$t.bin $(BUILD)/$$t.rt; \
	        if cmp -s test/vectors/$$t.bin $(BUILD)/$$t.rt; then \
	            echo "    $$t: re-emitted byte-identically"; ok=$$((ok+1)); \
	        else echo "    FAIL $$t: re-emitted bytes differ"; exit 1; fi; \
	    fi; done; echo "  round trip: $$ok file(s) byte-identical"

# clang-tidy, run for the first time at work item 9. D31 consequence 5 said it
# would arrive "with the F' toolchain"; it does not -- F' ships no clang-tidy, and
# it comes from Homebrew's llvm, which is what .clang-tidy said all along
# (docs/MODELS.md 20.2 correction 10). Homebrew does not put it on PATH, so this
# looks there too rather than reporting an absence that is really a PATH.
#
# Three configurations, because a core this project calls flight code should be
# held to F's own bar as well as its own: flight/.clang-tidy, then F's root
# .clang-tidy over everything, then F's release.clang-tidy, which is the one F'
# reserves for flight code and whose single check is misc-no-recursion.
TIDY := $(shell command -v clang-tidy 2>/dev/null || \
                 ls /opt/homebrew/opt/llvm/bin/clang-tidy 2>/dev/null)
FPRIME_DIR := ../fprime/lib/fprime

# (!) EXIT STATUS IS LOAD-BEARING AND ITS ABSENCE WAS A DEFECT, TWICE.
#
# Until 2026-09-11 this recipe ran clang-tidy three times, checked no exit status,
# and echoed "lint: clean" unconditionally -- so the target could not fail. It was
# reporting nine errors at the time, three of them in `src/DynamicThreshold.cpp`,
# and still exiting 0. A gate that cannot go red is not a gate.
#
# The second time was the same shape one level up. From the day the build trees
# were deleted (2026-09-21) `$(FPRIME_DIR)` was absent, so two of the three configs
# did not run -- and the recipe still printed "lint: clean" and exited 0. The skip
# was announced, but it was announced INSIDE a green result, which is the form
# nobody reads. **A skip inside a pass is not a pass** (Decision 79).
#
# So: "clean" is printed only when all three configs ran and passed. Anything less
# is PARTIAL, names what was skipped and why, and exits non-zero. `LINT_ALLOW_PARTIAL=1`
# accepts a partial run deliberately -- on a tree without the framework checkout,
# which is the normal state of a fresh clone -- and still says PARTIAL.
#
# Exit status is checked with an explicit `|| exit 1` per invocation rather than
# `set -e`, because `set -e` is not in effect inside every compound command a
# portable shell may choose, and this is the one property the target exists for.
lint:
	@ran=0; skipped=''; \
	if [ -z "$(TIDY)" ]; then \
	    echo "== lint: clang-tidy is not installed. brew install llvm provides it."; \
	    skipped='all three configs -- no clang-tidy on this machine'; \
	else \
	    printf '== lint: '; $(TIDY) --version | head -1; \
	    echo "-- flight/.clang-tidy, core and tests"; \
	    $(TIDY) $(CORE_SRC) test/*.cpp -- $(CXXFLAGS) -Itest || exit 1; \
	    ran=1; \
	    if [ -f $(FPRIME_DIR)/.clang-tidy ]; then \
	        echo "-- the framework config, core and tests"; \
	        $(TIDY) --config-file=$(FPRIME_DIR)/.clang-tidy $(CORE_SRC) test/*.cpp -- $(CXXFLAGS) -Itest || exit 1; \
	        echo "-- the framework release config, flight code only"; \
	        $(TIDY) --config-file=$(FPRIME_DIR)/release.clang-tidy $(CORE_SRC) -- $(CXXFLAGS) || exit 1; \
	        ran=3; \
	    else \
	        skipped="the framework config and its release config -- $(FPRIME_DIR) is absent; run scripts/fprime_setup.sh"; \
	    fi; \
	fi; \
	if [ -z "$$skipped" ]; then \
	    echo "  lint: clean ($$ran of 3 configs ran)"; \
	    exit 0; \
	fi; \
	echo "  lint: PARTIAL ($$ran of 3 configs ran)"; \
	echo "        skipped: $$skipped"; \
	if [ -n "$(LINT_ALLOW_PARTIAL)" ]; then \
	    echo "        LINT_ALLOW_PARTIAL is set, so a partial run is accepted here."; \
	    exit 0; \
	fi; \
	echo "        A skip inside a green gate is not a pass. Set LINT_ALLOW_PARTIAL=1"; \
	echo "        to accept a partial run on a tree without the framework checkout."; \
	exit 1

clean:
	rm -rf $(BUILD)
