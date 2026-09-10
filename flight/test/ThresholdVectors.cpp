// `DynamicThreshold` against `src/sentinel_models/flight_reference.py`, tiers
// D1-D2. Tolerance 1e-5 on `eps`, and EXACT on the emission flag
// (`docs/MODELS.md` 39.9, N1).
//
// The reference is the streamed statement of the rule, and it is itself pinned to
// `telemanom.channel_ratios` by `tests/test_flight_reference.py`: with backward
// dilation restored the two produce byte-identical emission arrays. So a failure
// here is a failure of the transcription, not of the restatement.
//
// These vectors carry FORWARD-ONLY dilation, which is the flight rule (39.5) and
// not the published one. The published one cannot be emitted.
#include <cmath>
#include <cstdio>
#include <cstring>

#include "sentinel/Config.hpp"
#include "sentinel/DynamicThreshold.hpp"
#include "TestSupport.hpp"

namespace SentinelTest {
int failures = 0;
}

using namespace Sentinel;

namespace {

constexpr F64 TOLERANCE = 1e-5;

F32 readF32(const U8* p) { F32 v = 0.0F; std::memcpy(&v, p, sizeof(v)); return v; }
F64 readF64(const U8* p) { F64 v = 0.0; std::memcpy(&v, p, sizeof(v)); return v; }

//! One tier, streamed. Returns the worst absolute `eps` difference.
F64 runTier(const char* tier, const char* path, U32& tiersRun) {
    std::FILE* handle = std::fopen(path, "rb");
    if (handle == nullptr) {
        std::printf("    %-4s SKIPPED -- %s absent\n", tier, path);
        return 0.0;
    }

    U8 head[24];
    if (std::fread(head, 1U, sizeof(head), handle) != sizeof(head)
        || (std::memcmp(head, "SNTD", 4U) != 0)) {
        (void)std::fclose(handle);
        SentinelTest::check(false, "vector is not a SNTD v1 header");
        return 0.0;
    }
    U16 version = 0U;
    U16 channels = 0U;
    U32 steps = 0U;
    U32 span = 0U;
    U32 stride = 0U;
    U32 solve = 0U;
    std::memcpy(&version, head + 4, sizeof(version));
    std::memcpy(&channels, head + 6, sizeof(channels));
    std::memcpy(&steps, head + 8, sizeof(steps));
    std::memcpy(&span, head + 12, sizeof(span));
    std::memcpy(&stride, head + 16, sizeof(stride));
    std::memcpy(&solve, head + 20, sizeof(solve));

    SentinelTest::checkEqualU32(span, Config::ERROR_WINDOW, "span is ERROR_WINDOW");
    SentinelTest::checkEqualU32(stride, Config::STRIDE, "stride is STRIDE");
    SentinelTest::checkEqualU32(solve, Config::SOLVE_WINDOW, "solve is SOLVE_WINDOW");
    SentinelTest::check(channels <= Config::MAX_CHANNELS, "width fits the maxima");

    DynamicThreshold detector;
    detector.configure(channels);

    const U32 record = (static_cast<U32>(channels) * 12U) + 1U;
    U8 row[(Config::MAX_CHANNELS * 12U) + 1U];
    F32 values[Config::MAX_CHANNELS];

    F64 worst = 0.0;
    U32 worstStep = 0U;
    U32 flagDiffs = 0U;
    U32 firstFlagDiff = 0U;
    U32 emissions = 0U;
    U32 expectedEmissions = 0U;
    U32 read = 0U;

    for (U32 t = 0U; t < steps; ++t) {
        if (std::fread(row, 1U, static_cast<size_t>(record), handle)
                != static_cast<size_t>(record)) {
            break;
        }
        ++read;
        for (U32 c = 0U; c < channels; ++c) {
            values[c] = readF32(row + (c * 4U));
        }

        // (!) BEFORE the step, not after. `eps_held[t]` is the threshold IN FORCE
        // at `t`, which is the one solved at the end of the previous segment. On
        // a segment's last tick `step` performs a solve and replaces it, so
        // reading afterwards compares the next segment's threshold against this
        // segment's expectation -- which is what the first run of this test did,
        // and it is why every difference it reported sat on a boundary tick.
        const U8* epsRow = row + (static_cast<U32>(channels) * 4U);
        for (U32 c = 0U; c < channels; ++c) {
            const F64 diff = std::fabs(detector.epsilon(c) - readF64(epsRow + (c * 8U)));
            if (diff > worst) { worst = diff; worstStep = t; }
        }

        detector.step(values);

        const bool expected = (row[record - 1U] != 0U);
        if (expected) { ++expectedEmissions; }
        if (detector.emitted()) { ++emissions; }
        if (detector.emitted() != expected) {
            if (flagDiffs == 0U) { firstFlagDiff = t; }
            ++flagDiffs;
        }
    }
    (void)std::fclose(handle);

    SentinelTest::checkEqualU32(read, steps, "every step in the file was read");
    std::printf("    %-4s %2u ch x %4u steps   eps max |diff| %.3e (step %u)   "
                "emissions %u of %u\n",
                tier, static_cast<unsigned>(channels), static_cast<unsigned>(read),
                worst, static_cast<unsigned>(worstStep),
                static_cast<unsigned>(emissions),
                static_cast<unsigned>(expectedEmissions));
    if (flagDiffs != 0U) {
        std::printf("    %-4s FLAG DIFFERS on %u step(s), first at %u\n", tier,
                    static_cast<unsigned>(flagDiffs),
                    static_cast<unsigned>(firstFlagDiff));
    }
    SentinelTest::check(worst <= TOLERANCE, "eps within 1e-5 of the reference");
    SentinelTest::checkEqualU32(flagDiffs, 0U, "the emission flag is exact");
    ++tiersRun;
    return worst;
}

//! The rule stays silent when it cannot choose, rather than guessing.
void checkSilence() {
    DynamicThreshold detector;
    detector.configure(1U);
    const F32 flat = 0.25F;
    for (U32 t = 0U; t < Config::STRIDE * 5U; ++t) {
        detector.step(&flat);
        SentinelTest::check(!detector.emitted(),
                            "a channel with zero variance never emits");
    }
    SentinelTest::check(detector.solves() == 5U, "it still solved every segment");
}

//! Emission is an instant, not an extent: it fires on the segment's last tick.
void checkEmissionIsAnInstant() {
    DynamicThreshold detector;
    detector.configure(1U);
    U32 emitted = 0U;
    U32 offSegment = 0U;
    for (U32 t = 0U; t < Config::STRIDE * 30U; ++t) {
        const F32 sample = ((t >= 500U) && (t < 560U)) ? 40.0F : 0.5F;
        detector.step(&sample);
        if (detector.emitted()) {
            ++emitted;
            if (((t + 1U) % Config::STRIDE) != 0U) {
                ++offSegment;
            }
        }
    }
    SentinelTest::check(emitted > 0U, "a clear excursion emits at least once");
    SentinelTest::checkEqualU32(offSegment, 0U,
                                "every emission lands on a segment boundary");
}

void checkRefusal() {
    DynamicThreshold detector;
    detector.configure(Config::MAX_CHANNELS + 1U);
    SentinelTest::checkEqualU32(detector.nChannels(), 0U,
                                "an oversized width configures to zero channels");
    const F32 sample = 1.0F;
    detector.step(&sample);
    SentinelTest::check(!detector.emitted(), "an inert threshold never emits");
    SentinelTest::check(detector.steps() == 0U, "an inert threshold never advances");
}

}  // namespace

int main() {
    std::printf("== dynamic threshold ==\n");

    U32 tiers = 0U;
    F64 worst = 0.0;
    const char* paths[2] = {"test/vectors/d1.dvec", "test/vectors/d2.dvec"};
    const char* names[2] = {"d1", "d2"};
    for (U32 i = 0U; i < 2U; ++i) {
        const F64 seen = runTier(names[i], paths[i], tiers);
        if (seen > worst) { worst = seen; }
    }
    std::printf("    %u tier(s) run\n", static_cast<unsigned>(tiers));
    std::printf("    worst eps over all tiers: %.3e, tolerance %.0e\n", worst, TOLERANCE);
    SentinelTest::check(tiers == 2U, "every committed tier ran; none was skipped");

    checkSilence();
    checkEmissionIsAnInstant();
    checkRefusal();

    return SentinelTest::report("dynamic threshold");
}
