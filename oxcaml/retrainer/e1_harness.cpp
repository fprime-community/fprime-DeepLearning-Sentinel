// E1 stage 1: the boundary, under the flight flag set. docs/MODELS.md 47.9, X1 to X3.
//
// (!) THIS IS COMPILED WITH THE EXACT NINE FLAGS flight/Makefile USES, AND THAT IS THE
// POINT OF IT. X2 predicts the flag set survives contact with the OCaml runtime:
//   -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off
//   -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
// If any one of them has to be dropped or weakened to make this link, X2 FAILS and 47.12
// stop 10 fires -- that flag set is what every measurement in this repository was taken
// under, and a boundary that needs it relaxed is not a boundary this project can use.
//
// The result is hand-checkable on purpose (no ML, 47.2): feed 1.0 through 10.0 and the
// answers are count 10, sum 55, mean 5.5 -- all three exact in F64, so "close enough" is
// not a possible outcome and a transcription error cannot hide in a tolerance.
#include "sentinel_retrainer.h"

#include <cstdio>
#include <cstdint>

namespace {

const double TOLERANCE = 0.0;  // exact. 55 and 5.5 are representable in F64.

bool check(const char *what, double got, double want)
{
    const double diff = got - want;
    const bool ok = (diff <= TOLERANCE) && (-diff <= TOLERANCE);
    std::printf("    %-10s got %.17g  want %.17g   %s\n",
                what, got, want, ok ? "OK" : "MISMATCH");
    return ok;
}

bool expect(const char *what, int32_t got, int32_t want)
{
    const bool ok = (got == want);
    std::printf("    %-10s got %d  want %d   %s\n", what, got, want, ok ? "OK" : "MISMATCH");
    return ok;
}

}  // namespace

int main()
{
    bool pass = true;
    std::printf("== E1 stage 1: the boundary ==\n");

    std::printf("  boot\n");
    pass = expect("boot", sentinel_retrainer_boot(), SENTINEL_RT_OK) && pass;
    pass = expect("status", sentinel_retrainer_status(), SENTINEL_RT_STATE_UNINIT) && pass;

    std::printf("  init\n");
    pass = expect("init", sentinel_retrainer_init(64U), SENTINEL_RT_OK) && pass;
    pass = expect("status", sentinel_retrainer_status(), SENTINEL_RT_STATE_READY) && pass;
    // The refusals are exercised, not just the accept path -- flight/test/RefusalTests.cpp
    // makes the same argument about the model loader.
    pass = expect("re-init", sentinel_retrainer_init(64U), SENTINEL_RT_ERR_ALREADY_INIT) && pass;

    std::printf("  feed 1..10\n");
    double samples[10];
    for (uint32_t i = 0U; i < 10U; ++i) {
        samples[i] = static_cast<double>(i + 1U);
    }
    pass = expect("feed", sentinel_retrainer_feed(samples, 10U), SENTINEL_RT_OK) && pass;
    pass = expect("status", sentinel_retrainer_status(), SENTINEL_RT_STATE_FED) && pass;

    std::printf("  step\n");
    pass = expect("step", sentinel_retrainer_step(), SENTINEL_RT_OK) && pass;
    pass = expect("status", sentinel_retrainer_status(), SENTINEL_RT_STATE_STEPPED) && pass;

    std::printf("  export, and these are the hand-checked numbers\n");
    double out[3] = { 0.0, 0.0, 0.0 };
    pass = expect("export", sentinel_retrainer_export(out, 3U), SENTINEL_RT_OK) && pass;
    pass = check("count", out[0], 10.0) && pass;
    pass = check("sum",   out[1], 55.0) && pass;
    pass = check("mean",  out[2], 5.5)  && pass;

    std::printf("  refusals\n");
    pass = expect("short buf", sentinel_retrainer_export(out, 2U),
                  SENTINEL_RT_ERR_SHORT_BUFFER) && pass;
    pass = expect("null buf", sentinel_retrainer_export(nullptr, 3U),
                  SENTINEL_RT_ERR_SHORT_BUFFER) && pass;
    // 64 capacity, 10 already fed, so 60 more overflows -- and it must be REFUSED with a
    // status rather than raising, because an OCaml exception reaching here is UB under
    // -fno-exceptions (47.6).
    double big[60];
    for (uint32_t i = 0U; i < 60U; ++i) {
        big[i] = 1.0;
    }
    pass = expect("overflow", sentinel_retrainer_feed(big, 60U),
                  SENTINEL_RT_ERR_OVERFLOW) && pass;
    // And the state is unchanged by a refused call.
    pass = expect("status", sentinel_retrainer_status(), SENTINEL_RT_STATE_STEPPED) && pass;

    std::printf("== E1 stage 1: %s ==\n", pass ? "all checks passed" : "FAILED");
    return pass ? 0 : 1;
}
