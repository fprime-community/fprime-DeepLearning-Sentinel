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
            src/Ewma.cpp src/Detector.cpp
CORE_OBJ := $(patsubst src/%.cpp,$(BUILD)/%.o,$(CORE_SRC))

TESTS    := GoldenVectors RefusalTests DeterminismTest RoundTripDump Footprint
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
	@echo "== round trip =="
	@set -e; ok=0; for t in g1 g2 g3 g4_0 g4_1 g4_2 g4_3; do \
	    if [ -f test/vectors/$$t.bin ]; then \
	        $(BUILD)/RoundTripDump test/vectors/$$t.bin $(BUILD)/$$t.rt; \
	        if cmp -s test/vectors/$$t.bin $(BUILD)/$$t.rt; then \
	            echo "    $$t: re-emitted byte-identically"; ok=$$((ok+1)); \
	        else echo "    FAIL $$t: re-emitted bytes differ"; exit 1; fi; \
	    fi; done; echo "  round trip: $$ok file(s) byte-identical"

# clang-tidy is deferred to work item 9 with the F' toolchain (D31); it is not
# available on the development machine. The target is written now so it runs the
# moment a toolchain has it, and says so plainly when it cannot.
lint:
	@if command -v clang-tidy >/dev/null 2>&1; then \
	    clang-tidy $(CORE_SRC) test/*.cpp -- $(CXXFLAGS); \
	else \
	    echo "lint: clang-tidy not installed -- deferred to work item 9 (D31)."; \
	    echo "      brew install llvm provides it; flight/.clang-tidy is ready."; \
	fi

clean:
	rm -rf $(BUILD)
