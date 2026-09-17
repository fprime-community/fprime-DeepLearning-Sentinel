// Same inputs, same outputs -- bit for bit, within a run and across runs.
//
// Objective.md 11 rule 5. The golden vectors prove the core matches the reference
// at 1e-5; that is a different claim from "this binary produces identical bits
// every time", which is what a flight review board asks for and what
// `-ffp-contract=off` and the absence of `-ffast-math` exist to guarantee.
//
// Run twice: the first run writes its digest, the second compares. `make test`
// invokes it twice, so the across-process half is real rather than nominal.
#include <cstdio>
#include <cstring>

#include "sentinel/Crc32.hpp"
#include "sentinel/Detector.hpp"
#include "TestSupport.hpp"

namespace SentinelTest {
int failures = 0;
}

using namespace Sentinel;

namespace {

U8 g_model[SentinelTest::MAX_FILE_BYTES];
const char* DIGEST_PATH = "build/determinism.crc";

//! 55's arm A. What a run actually reached, counted rather than assumed.
//!
//! The claim of record ran 400 ticks. The threshold re-solves every STRIDE = 70,
//! so that run solved about five times -- but the ring holds SOLVE_WINDOW = 2,170
//! and the moments span ERROR_WINDOW = 2,100, so neither had wrapped, and
//! `TrailingWindow`'s eviction path (`m_sum[c] -= leaving`, a running sum with
//! subtraction) was never executed. `Detector::reset` zeroes the fill, so a
//! segment has to fill the ring by itself: the longest one in a run is half of it.
struct Coverage {
    U32 segmentsWrapped;    //!< segments that reached the ring's capacity
    U32 momentEvictions;    //!< ticks at or past ERROR_WINDOW, where the sum subtracts
    U32 solvesAfterWrap;    //!< solves performed with the ring at capacity
};

//! CRC over the raw bit patterns of every output the detector produces.
//!
//! Bit patterns, not values: two floats that compare equal can differ in their
//! encoding, and the claim under test is about the encoding.
U32 runPipeline(const U8* model, U32 length, U32 steps, Coverage* coverage = nullptr) {
    Detector detector;
    if (detector.load(model, length) != LoadStatus::OK) {
        return 0U;
    }
    U32 previousSolves = 0U;
    Crc32 digest;
    F32 values[Config::MAX_CHANNELS];
    const U32 channels = detector.model().nChannels;

    for (U32 t = 0U; t < steps; ++t) {
        if (t == (steps / 2U)) {
            detector.reset();
        }
        for (U32 c = 0U; c < channels; ++c) {
            // A deterministic, bounded drive. Not telemetry; the point is
            // repeatability, not realism.
            const U32 mix = ((t * 2654435761U) + (c * 40503U)) & 0xFFFFU;
            values[c] = static_cast<F32>(mix) / 65535.0F;
        }
        detector.step(values, (t % 97U) != 0U);   // exercise the invalid path too

        digest.update(reinterpret_cast<const U8*>(detector.hidden(0)),
                      4U * detector.model().hidden[0]);
        digest.update(reinterpret_cast<const U8*>(detector.headOutput()),
                      4U * detector.model().nOutputs);
        digest.update(reinterpret_cast<const U8*>(detector.forecast()), 4U * channels);
        digest.update(reinterpret_cast<const U8*>(detector.residual()), 4U * channels);
        digest.update(reinterpret_cast<const U8*>(detector.smoothed()), 4U * channels);
        const F32 score = detector.score();
        digest.update(reinterpret_cast<const U8*>(&score), 4U);
        const U8 flags = static_cast<U8>((detector.crossing() ? 1U : 0U)
                                         | (detector.emitted() ? 2U : 0U));
        digest.update(&flags, 1U);

        // Read-only, and after the digest, so coverage cannot influence what is
        // hashed. `steps()` counts pushes since the last reset; the window is at
        // capacity once it reaches SOLVE_WINDOW.
        const U64 depth = detector.threshold().steps();
        const U32 solves = detector.threshold().solves();
        if (coverage != nullptr) {
            if (depth == static_cast<U64>(Config::SOLVE_WINDOW)) {
                ++coverage->segmentsWrapped;
            }
            if (depth >= static_cast<U64>(Config::ERROR_WINDOW)) {
                ++coverage->momentEvictions;
            }
            if ((depth >= static_cast<U64>(Config::SOLVE_WINDOW))
                && (solves > previousSolves)) {
                ++coverage->solvesAfterWrap;
            }
        }
        previousSolves = solves;
    }
    return digest.value();
}

}  // namespace

int main() {
    std::printf("== determinism ==\n");

    const U32 length = SentinelTest::readFile("test/vectors/g1.bin", g_model,
                                              SentinelTest::MAX_FILE_BYTES);
    if (length == 0U) {
        std::printf("    FAIL  test/vectors/g1.bin not readable\n");
        return 1;
    }

    const U32 first = runPipeline(g_model, length, 400U);
    const U32 second = runPipeline(g_model, length, 400U);
    std::printf("    output digest  0x%08X\n", first);
    SentinelTest::check(first != 0U, "the pipeline produced something to hash");
    SentinelTest::check(first == second,
                        "two runs in one process agree bit for bit");

    // A reset must put the detector back where it started: the first half of a
    // fresh run and the second half of a reset run drive identical arithmetic.
    const U32 shorter = runPipeline(g_model, length, 200U);
    SentinelTest::check(shorter != first, "the 200-step run differs, as it must");

    // -- 55 arm A: a run long enough to wrap the window ----------------------
    //
    // Derived at 55.3a, not chosen: a segment must fill the ring (2,170) and then
    // solve at least twelve times (12 x 70 = 840), so a segment is at least 3,010
    // ticks, and a segment is half a run. 6,400 gives segments of 3,200.
    const U32 LONG_STEPS = 6400U;
    Coverage firstCoverage = {0U, 0U, 0U};
    Coverage secondCoverage = {0U, 0U, 0U};
    const U32 longFirst = runPipeline(g_model, length, LONG_STEPS, &firstCoverage);
    const U32 longSecond = runPipeline(g_model, length, LONG_STEPS, &secondCoverage);

    std::printf("    long run       %u ticks, digest 0x%08X\n", LONG_STEPS, longFirst);
    std::printf("      segments that wrapped the ring   %u\n", firstCoverage.segmentsWrapped);
    std::printf("      ticks evicting from the moments  %u\n", firstCoverage.momentEvictions);
    std::printf("      solves with the ring at capacity %u\n", firstCoverage.solvesAfterWrap);

    // DT2 before DT1: a long run that did not wrap is a short run with more of the
    // same, and would prove nothing the 400-tick claim did not already say.
    SentinelTest::check(firstCoverage.segmentsWrapped == 2U,
                        "both segments of the long run wrapped the ring");
    SentinelTest::check(firstCoverage.momentEvictions > 0U,
                        "the long run executed the moment-window eviction path");
    SentinelTest::check(firstCoverage.solvesAfterWrap >= 12U,
                        "at least twelve solves ran with the ring at capacity");

    SentinelTest::check(longFirst != 0U, "the long run produced something to hash");
    SentinelTest::check(longFirst == longSecond,
                        "two long runs in one process agree bit for bit");
    SentinelTest::check(firstCoverage.solvesAfterWrap == secondCoverage.solvesAfterWrap,
                        "the two long runs covered the same ground");
    SentinelTest::check(longFirst != first,
                        "the long run differs from the 400-tick run, as it must");

    std::FILE* stored = std::fopen(DIGEST_PATH, "rb");
    if (stored == nullptr) {
        std::FILE* out = std::fopen(DIGEST_PATH, "wb");
        if (out != nullptr) {
            (void)std::fprintf(out, "%08X %08X\n", first, longFirst);
            (void)std::fclose(out);
        }
        std::printf("    first run: digest recorded for the next process to check\n");
    } else {
        unsigned int previous = 0U;
        unsigned int previousLong = 0U;
        const int read = std::fscanf(stored, "%08X %08X", &previous, &previousLong);
        (void)std::fclose(stored);
        SentinelTest::check(read == 2, "both recorded digests were readable");
        SentinelTest::check(static_cast<U32>(previous) == first,
                            "a fresh process reproduces the digest bit for bit");
        SentinelTest::check(static_cast<U32>(previousLong) == longFirst,
                            "a fresh process reproduces the WRAPPED-window digest bit for bit");
        std::printf("    across processes: 0x%08X == 0x%08X, long 0x%08X == 0x%08X\n",
                    static_cast<U32>(previous), first,
                    static_cast<U32>(previousLong), longFirst);
    }

    return SentinelTest::report("determinism");
}
