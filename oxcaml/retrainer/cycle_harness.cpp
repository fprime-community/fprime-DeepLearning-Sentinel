// Section 61 / E5-c: the C++ side of the cycle boundary, at F's own flag set.
//
// (!) BUILT WITH -Wold-style-cast AND -Wdouble-promotion, which flight/Makefile:25
// does not carry. cmake/flags.cmake:46-60 is the framework's validation set and it
// is OPT-IN; this harness opts in. -Wdouble-promotion is the one that matters here:
// a float32 boundary that silently widens to double is exactly the defect it is for.
//
// Every buffer below is this file's. The OCaml side receives CAML_BA_EXTERNAL views
// that do not outlive their call.
#include <cstdint>
#include <cstdio>

#include "sentinel_cycle.h"

namespace {

constexpr uint32_t T_MAX = 250U;
constexpr uint32_t INS = 16U;
constexpr uint32_t N_PARAMS = 75360U;
constexpr int32_t BUDGET = 3;

// CPP-1: claimed once, at file scope, never after init.
float g_window[T_MAX * INS];
float g_export[N_PARAMS];

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("   %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) { ++failures; }
}

}  // namespace

int main()
{
    std::printf("== Section 61 / E5-c: the cycle across the C boundary ==\n");

    check(sentinel_cycle_boot() == SENTINEL_CYC_OK, "the runtime boots");
    check(sentinel_cycle_boot() == SENTINEL_CYC_OK, "booting twice is idempotent");
    check(sentinel_cycle_init(7) == SENTINEL_CYC_OK, "a starting model is seeded");

    // A deterministic, bounded drive. Not telemetry.
    for (uint32_t t = 0U; t < T_MAX; ++t) {
        for (uint32_t c = 0U; c < INS; ++c) {
            const uint32_t mix = ((t * 2654435761U) + (c * 40503U)) & 0xFFFFU;
            g_window[(t * INS) + c] = static_cast<float>(mix) / 65535.0F;
        }
    }
    check(sentinel_cycle_load(g_window, T_MAX * INS) == SENTINEL_CYC_OK,
          "the caller's window crosses as CAML_BA_EXTERNAL");

    // FC7a: a wrong shape is a STATUS, not a throw.
    check(sentinel_cycle_load(g_window, 7U) == SENTINEL_CYC_ERR_SHAPE,
          "a wrong window extent returns a status, not an exception");
    check(sentinel_cycle_run(0, 250, 1) == SENTINEL_CYC_ERR_BUDGET,
          "a zero budget returns a status");
    check(sentinel_cycle_run(BUDGET, 10000, 1) == SENTINEL_CYC_ERR_BUDGET,
          "an over-long sequence returns a status");
    check(sentinel_cycle_export(g_export, 3U) == SENTINEL_CYC_ERR_SHAPE,
          "a wrong export extent returns a status");

    // FC5: a full cycle, and the step count is D73's budget exactly.
    check(sentinel_cycle_run(BUDGET, 60, 1) == SENTINEL_CYC_OK, "a full cycle runs");
    const int32_t steps = sentinel_cycle_steps();
    std::printf("   steps taken %d against a budget of %d\n", steps, BUDGET);
    check(steps == BUDGET, "the cycle took exactly the budgeted number of steps");

    // 56's gate: a refused window costs nothing.
    check(sentinel_cycle_run(BUDGET, 60, 0) == SENTINEL_CYC_OK, "a refused window returns OK");
    check(sentinel_cycle_steps() == 0, "a refused window takes zero steps");

    // The best-weights block comes back across the same boundary.
    check(sentinel_cycle_run(BUDGET, 60, 1) == SENTINEL_CYC_OK, "a second cycle runs");
    check(sentinel_cycle_export(g_export, N_PARAMS) == SENTINEL_CYC_OK,
          "the best-weights block crosses back");
    bool finite = true, nonzero = false;
    for (uint32_t i = 0U; i < N_PARAMS; ++i) {
        const float v = g_export[i];
        if (!(v == v) || (v > 1e30F) || (v < -1e30F)) { finite = false; }
        if (v != 0.0F) { nonzero = true; }
    }
    check(finite, "every exported weight is finite");
    check(nonzero, "the exported block is not all zero");

    std::printf("   %s\n", failures == 0 ? "boundary: all checks passed" : "FAILURES ABOVE");
    return failures == 0 ? 0 : 1;
}
