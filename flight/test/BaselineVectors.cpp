// The Level 1 baseline, held to `src/sentinel_models/baseline_reference.py`.
//
// Same discipline as GoldenVectors: seeded inputs, a committed vector per tier,
// per-channel scores compared at 1e-5 and the crossing, emitted and peak-channel
// flags compared EXACTLY. Tier B3 is the N(1000, 3) regime that exposes the
// float32 cancellation D37 records, and it exists so an implementation that
// reintroduces float32 accumulation fails here rather than passing quietly.
//
// `.bvec` layout, little-endian throughout, written by
// `scripts/make_baseline_vectors.py`:
//
//     off  size   field
//       0     4   magic 'SNTB'
//       4     2   version = 1
//       6     2   n_channels
//       8     4   steps
//      12     4   window
//      16     8   threshold          F64
//      24   8*C   scale              F64[C]
//     then `steps` records of 12*C + 12 bytes:
//       +0   4*C   values            F32[C]
//       +.   8*C   channel scores    F64[C]
//       +.     8   combined score    F64
//       +.     4   valid, crossing, emitted, peak   U8 each
#include <cmath>
#include <cstdio>
#include <cstring>

#include "TestSupport.hpp"
#include "sentinel/Baseline.hpp"

namespace SentinelTest {
int failures = 0;
}

using Sentinel::Baseline;
using Sentinel::F32;
using Sentinel::F64;
using Sentinel::U16;
using Sentinel::U32;
using Sentinel::U8;

namespace {

//! `docs/MODELS.md` 20.5 C3. The prediction beside it is 1e-9.
const F64 TOLERANCE = 1e-5;

const char* const TIERS[] = {"b1", "b2", "b3", "b4"};
const U32 TIER_COUNT = 4U;

U8 g_buffer[SentinelTest::MAX_FILE_BYTES];

U16 readU16(const U8* p) {
    return static_cast<U16>(static_cast<U16>(p[0]) |
                            static_cast<U16>(static_cast<U16>(p[1]) << 8U));
}

U32 readU32(const U8* p) {
    return static_cast<U32>(p[0]) | (static_cast<U32>(p[1]) << 8U) |
           (static_cast<U32>(p[2]) << 16U) | (static_cast<U32>(p[3]) << 24U);
}

F32 readF32(const U8* p) {
    F32 value = 0.0F;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

F64 readF64(const U8* p) {
    F64 value = 0.0;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

//! Both infinite with the same sign counts as equal; an invalid tick scores
//! -infinity on both sides and subtracting them would yield NaN.
bool closeEnough(F64 a, F64 b, F64& difference) {
    if (std::isinf(a) && std::isinf(b)) {
        difference = 0.0;
        return (a > 0.0) == (b > 0.0);
    }
    if (std::isnan(a) || std::isnan(b)) {
        difference = 1.0;
        return false;
    }
    difference = std::fabs(a - b);
    return difference <= TOLERANCE;
}

F64 g_worst = 0.0;
const char* g_worstTier = "-";
U32 g_worstStep = 0U;

bool runTier(const char* tier) {
    char path[128];
    (void)std::snprintf(path, sizeof(path), "test/vectors/%s.bvec", tier);
    if (!SentinelTest::exists(path)) {
        std::printf("    %s: absent, skipped\n", tier);
        return false;
    }

    const U32 length = SentinelTest::readFile(path, g_buffer, SentinelTest::MAX_FILE_BYTES);
    SentinelTest::check(length > 24U, "vector file is at least a header");
    if (length <= 24U) {
        return false;
    }
    SentinelTest::check(std::memcmp(g_buffer, "SNTB", 4U) == 0, "magic is SNTB");
    SentinelTest::checkEqualU32(readU16(g_buffer + 4), 1U, "vector format version");

    const U32 channels = readU16(g_buffer + 6);
    const U32 steps = readU32(g_buffer + 8);
    const U32 window = readU32(g_buffer + 12);
    const F64 threshold = readF64(g_buffer + 16);
    SentinelTest::checkEqualU32(window, Sentinel::Config::BASELINE_WINDOW,
                                "vector window matches Config::BASELINE_WINDOW");

    F64 scale[Sentinel::Config::MAX_CHANNELS];
    for (U32 c = 0U; c < channels; ++c) {
        scale[c] = readF64(g_buffer + 24U + (8U * c));
    }

    const U32 recordBytes = (12U * channels) + 12U;
    const U32 headerBytes = 24U + (8U * channels);
    SentinelTest::checkEqualU32(length, headerBytes + (recordBytes * steps),
                                "vector file length matches its own shape");

    Baseline baseline;
    baseline.configure(channels);
    baseline.setScale(scale, channels);
    baseline.setThreshold(threshold);

    F64 worst = 0.0;
    U32 worstStep = 0U;
    U32 flagMismatches = 0U;
    U32 peakMismatches = 0U;

    F32 values[Sentinel::Config::MAX_CHANNELS];
    for (U32 t = 0U; t < steps; ++t) {
        const U8* record = g_buffer + headerBytes + (recordBytes * t);
        for (U32 c = 0U; c < channels; ++c) {
            values[c] = readF32(record + (4U * c));
        }
        const U8* scoresAt = record + (4U * channels);
        const F64 expectedCombined = readF64(scoresAt + (8U * channels));
        const U8* flags = scoresAt + (8U * channels) + 8U;
        const bool valid = (flags[0] != 0U);

        baseline.step(values, valid);

        for (U32 c = 0U; c < channels; ++c) {
            F64 difference = 0.0;
            const bool ok = closeEnough(baseline.channelScores()[c],
                                        readF64(scoresAt + (8U * c)), difference);
            if (difference > worst) {
                worst = difference;
                worstStep = t;
            }
            if (!ok) {
                SentinelTest::check(false, "per-channel score within tolerance");
            }
        }

        F64 combinedDifference = 0.0;
        if (!closeEnough(baseline.score(), expectedCombined, combinedDifference)) {
            SentinelTest::check(false, "combined score within tolerance");
        }
        if (combinedDifference > worst) {
            worst = combinedDifference;
            worstStep = t;
        }

        if ((baseline.crossing() != (flags[1] != 0U)) ||
            (baseline.emitted() != (flags[2] != 0U))) {
            flagMismatches += 1U;
        }
        if (valid && (baseline.peakChannel() != static_cast<U32>(flags[3]))) {
            peakMismatches += 1U;
        }
    }

    SentinelTest::checkEqualU32(flagMismatches, 0U, "crossing and emitted flags exact");
    SentinelTest::checkEqualU32(peakMismatches, 0U, "peak channel exact");
    std::printf("    %-3s %2u ch x %4u steps   max |diff| %.3e  (step %u)\n",
                tier, channels, steps, worst, worstStep);

    if (worst > g_worst) {
        g_worst = worst;
        g_worstTier = tier;
        g_worstStep = worstStep;
    }
    return true;
}

//! The warm-up gate, and the fact that makes it worth stating: a degraded
//! component emits 2,230 ticks earlier than a healthy one (prediction C10).
void checkWarmup() {
    Baseline baseline;
    baseline.configure(4U);
    baseline.setThreshold(-1.0);   // everything crosses, so only warm-up gates
    const F32 values[4] = {1.0F, 2.0F, 3.0F, 4.0F};

    // Silent for calls 1..WINDOW, speaking on call WINDOW + 1. That is
    // `harness.py:160-167`'s convention and `Detector.cpp:209`'s, not an
    // off-by-one: see Baseline::warmed.
    U32 emittedBeforeWarm = 0U;
    for (U32 t = 0U; t < Sentinel::Config::BASELINE_WINDOW; ++t) {
        baseline.step(values, true);
        if (baseline.emitted()) {
            emittedBeforeWarm += 1U;
        }
    }
    SentinelTest::checkEqualU32(emittedBeforeWarm, 0U, "silent for the whole warm-up");
    SentinelTest::check(baseline.crossing(), "but crossing, so the gate is warm-up alone");
    baseline.step(values, true);
    SentinelTest::check(baseline.emitted(), "emits on call WINDOW + 1, and not before");

    baseline.reset();
    baseline.step(values, true);
    SentinelTest::check(!baseline.emitted(), "a reset restarts the warm-up");
}

//! An unconfigured baseline, a null pointer and an invalid tick must all be
//! quiet rather than dangerous. Level 1 never fails the topology.
void checkInertPaths() {
    Baseline unconfigured;
    const F32 values[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    unconfigured.setThreshold(-1.0);
    unconfigured.step(values, true);
    SentinelTest::check(!unconfigured.emitted(), "unconfigured baseline emits nothing");
    SentinelTest::check(!unconfigured.crossing(), "unconfigured baseline never crosses");

    Baseline wide;
    wide.configure(Sentinel::Config::MAX_CHANNELS + 1U);
    SentinelTest::checkEqualU32(wide.nChannels(), 0U,
                                "a width past MAX_CHANNELS goes inert, not out of bounds");

    Baseline ok;
    ok.configure(4U);
    ok.setThreshold(-1.0);
    ok.step(nullptr, true);
    SentinelTest::check(!ok.emitted(), "a null sample emits nothing");

    for (U32 t = 0U; t <= Sentinel::Config::BASELINE_WINDOW; ++t) {
        ok.step(values, true);
    }
    SentinelTest::check(ok.emitted(), "and it still works afterwards");
    ok.step(values, false);
    SentinelTest::check(!ok.emitted(), "an invalid tick never alarms");
    SentinelTest::check(ok.score() < 0.0, "an invalid tick scores negative infinity");
}

}  // namespace

int main() {
    std::printf("== baseline vectors ==\n");
    checkWarmup();
    checkInertPaths();

    U32 ran = 0U;
    for (U32 i = 0U; i < TIER_COUNT; ++i) {
        if (runTier(TIERS[i])) {
            ran += 1U;
        }
    }
    std::printf("    %u tier(s) run\n", ran);
    std::printf("    worst over all tiers: %.3e  (%s, step %u), tolerance %.0e\n",
                g_worst, g_worstTier, g_worstStep, TOLERANCE);
    return SentinelTest::report("baseline");
}
