// The C++ core against `reference.py`, stage by stage.
//
// Each vector is a three-phase scenario: 24 steps from a zero state, 24 more
// continuing with the carried state (the chunk boundary), then a reset and 24
// more. Acceptance is <= 1e-5 on the state and every intermediate, and **exact**
// on the crossing flag (`docs/MODELS.md` 19, prediction F5).
//
// A divergence is a finding, not something to tune away. The maxima are printed
// per stage whether they pass or fail, so the work-item report quotes measured
// numbers rather than "within tolerance".
#include <cmath>
#include <cstdio>
#include <cstring>

#include "sentinel/Detector.hpp"
#include "TestSupport.hpp"

namespace SentinelTest {
int failures = 0;
}

using namespace Sentinel;

namespace {

constexpr F64 TOLERANCE = 1e-5;

U8 g_model[SentinelTest::MAX_FILE_BYTES];
U8 g_vector[SentinelTest::MAX_FILE_BYTES];

U32 readU32(const U8* p) {
    return static_cast<U32>(p[0]) | (static_cast<U32>(p[1]) << 8U)
         | (static_cast<U32>(p[2]) << 16U) | (static_cast<U32>(p[3]) << 24U);
}

F32 readF32(const U8* p) {
    const U32 bits = readU32(p);
    F32 value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

F64 readF64(const U8* p) {
    U64 bits = static_cast<U64>(readU32(p))
             | (static_cast<U64>(readU32(p + 4)) << 32U);
    F64 value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

//! Largest absolute difference seen for one stage, across a whole run.
struct Worst {
    F64 value;
    U32 step;
    Worst() : value(0.0), step(0U) {}
    void see(F64 difference, U32 at) {
        const F64 magnitude = (difference < 0.0) ? -difference : difference;
        if (magnitude > value) {
            value = magnitude;
            step = at;
        }
    }
};

void compare(Worst& worst, const F32* actual, const U8* expected, U32 count, U32 step) {
    for (U32 i = 0U; i < count; ++i) {
        worst.see(static_cast<F64>(actual[i]) - static_cast<F64>(readF32(expected + (4U * i))),
                  step);
    }
}

void line(const char* stage, const Worst& worst) {
    std::printf("      %-22s max |diff| %.3e  (step %u)%s\n", stage, worst.value,
                worst.step, (worst.value <= TOLERANCE) ? "" : "   <-- OVER TOLERANCE");
}

bool runTier(const char* tier) {
    char modelPath[128];
    char vectorPath[128];
    (void)std::snprintf(modelPath, sizeof(modelPath), "test/vectors/%s.bin", tier);
    (void)std::snprintf(vectorPath, sizeof(vectorPath), "test/vectors/%s.vec", tier);
    if (!SentinelTest::exists(modelPath) || !SentinelTest::exists(vectorPath)) {
        return false;
    }

    const U32 modelBytes = SentinelTest::readFile(modelPath, g_model,
                                                  SentinelTest::MAX_FILE_BYTES);
    const U32 vectorBytes = SentinelTest::readFile(vectorPath, g_vector,
                                                   SentinelTest::MAX_FILE_BYTES);
    if ((modelBytes == 0U) || (vectorBytes == 0U)) {
        std::printf("    FAIL  %s: unreadable\n", tier);
        SentinelTest::failures += 1;
        return true;
    }

    if ((g_vector[0] != 'S') || (g_vector[1] != 'N') || (g_vector[2] != 'T')
            || (g_vector[3] != 'V')) {
        std::printf("    FAIL  %s: bad vector magic\n", tier);
        SentinelTest::failures += 1;
        return true;
    }

    const U32 channels = readU32(g_vector + 8);
    const U32 layers = readU32(g_vector + 12);
    const U32 predictions = readU32(g_vector + 32);
    const U32 steps = readU32(g_vector + 36);
    const U32 phase2End = readU32(g_vector + 44);
    const U32 warmup = readU32(g_vector + 52);
    const F64 threshold = readF64(g_vector + 56);

    U32 hidden[Config::MAX_LAYERS];
    U32 hiddenTotal = 0U;
    for (U32 i = 0U; i < layers; ++i) {
        hidden[i] = readU32(g_vector + 16U + (4U * i));
        hiddenTotal += hidden[i];
    }

    Detector detector;
    const LoadStatus status = detector.load(g_model, modelBytes);
    if (status != LoadStatus::OK) {
        std::printf("    FAIL  %s: load returned %s\n", tier, statusName(status));
        SentinelTest::failures += 1;
        return true;
    }
    SentinelTest::check(detector.model().nChannels == channels,
                        "model and vector agree on the channel count");
    SentinelTest::check(detector.model().threshold == threshold,
                        "the threshold survived the file exactly");
    SentinelTest::check(detector.model().warmupSteps == warmup,
                        "the warm-up survived the file");

    // Body offsets, in the order the generator writes them.
    U32 cursor = 64U;
    const U8* input = g_vector + cursor;      cursor += 4U * steps * channels;
    const U8* hiddenTrace = g_vector + cursor; cursor += 4U * steps * hiddenTotal;
    const U8* head = g_vector + cursor;       cursor += 4U * steps * predictions * channels;
    const U8* forecast = g_vector + cursor;   cursor += 4U * steps * channels;
    const U8* residual = g_vector + cursor;   cursor += 4U * steps * channels;
    const U8* ewma = g_vector + cursor;       cursor += 4U * steps * channels;
    const U8* score = g_vector + cursor;      cursor += 4U * steps;
    const U8* crossing = g_vector + cursor;   cursor += steps;
    const U8* emitted = g_vector + cursor;    cursor += steps;
    SentinelTest::checkEqualU32(cursor, vectorBytes, "vector body accounted for");

    Worst worstHidden;
    Worst worstHead;
    Worst worstForecast;
    Worst worstResidual;
    Worst worstEwma;
    Worst worstScore;
    U32 flagMismatch = 0U;
    U32 emitMismatch = 0U;

    F32 values[Config::MAX_CHANNELS];
    for (U32 t = 0U; t < steps; ++t) {
        if (t == phase2End) {
            detector.reset();          // phase C. Nothing happens at phase1End:
        }                              // the state simply carries, which is the point.
        for (U32 c = 0U; c < channels; ++c) {
            values[c] = readF32(input + (4U * ((t * channels) + c)));
        }
        detector.step(values, true);

        U32 base = 0U;
        for (U32 layer = 0U; layer < layers; ++layer) {
            compare(worstHidden, detector.hidden(layer),
                    hiddenTrace + (4U * ((t * hiddenTotal) + base)), hidden[layer], t);
            base += hidden[layer];
        }
        compare(worstHead, detector.headOutput(),
                head + (4U * t * predictions * channels), predictions * channels, t);
        compare(worstForecast, detector.forecast(),
                forecast + (4U * t * channels), channels, t);
        compare(worstResidual, detector.residual(),
                residual + (4U * t * channels), channels, t);
        compare(worstEwma, detector.smoothed(),
                ewma + (4U * t * channels), channels, t);
        worstScore.see(static_cast<F64>(detector.score())
                       - static_cast<F64>(readF32(score + (4U * t))), t);

        if (detector.crossing() != (crossing[t] != 0U)) {
            flagMismatch += 1U;
        }
        if (detector.emitted() != (emitted[t] != 0U)) {
            emitMismatch += 1U;
        }
    }

    std::printf("    %s: %u channels, %u layers, l_p=%u, %u steps\n", tier, channels,
                layers, predictions, steps);
    line("hidden state", worstHidden);
    line("head output", worstHead);
    line("forecast", worstForecast);
    line("residual", worstResidual);
    line("EWMA", worstEwma);
    line("score", worstScore);

    SentinelTest::check(worstHidden.value <= TOLERANCE, "hidden state within 1e-5");
    SentinelTest::check(worstHead.value <= TOLERANCE, "head output within 1e-5");
    SentinelTest::check(worstForecast.value <= TOLERANCE, "forecast within 1e-5");
    SentinelTest::check(worstResidual.value <= TOLERANCE, "residual within 1e-5");
    SentinelTest::check(worstEwma.value <= TOLERANCE, "EWMA within 1e-5");
    SentinelTest::check(worstScore.value <= TOLERANCE, "score within 1e-5");
    if (flagMismatch != 0U) {
        std::printf("    FAIL  %s: crossing flag differs on %u of %u steps\n", tier,
                    flagMismatch, steps);
        SentinelTest::failures += 1;
    }
    if (emitMismatch != 0U) {
        std::printf("    FAIL  %s: emitted flag differs on %u of %u steps\n", tier,
                    emitMismatch, steps);
        SentinelTest::failures += 1;
    }
    return true;
}

//! How far the recursive EWMA denominator drifts from the reference's closed form.
//!
//! `docs/MODEL_FILE.md` and `Ewma.hpp` claim the two are the same sequence. This
//! measures it rather than asserting it, over far more steps than any vector.
void reportDenominatorDrift() {
    const F64 alpha = 2.0 / 106.0;
    const F64 decay = 1.0 - alpha;
    F64 recursive = 0.0;
    F64 worst = 0.0;
    for (U32 t = 0U; t < 8000U; ++t) {
        recursive = 1.0 + (decay * recursive);
        const F64 closed = (1.0 - std::pow(decay, static_cast<F64>(t) + 1.0)) / alpha;
        const F64 difference = (recursive - closed) / closed;
        const F64 magnitude = (difference < 0.0) ? -difference : difference;
        if (magnitude > worst) {
            worst = magnitude;
        }
    }
    std::printf("    EWMA denominator, recursive vs the reference's closed form:\n"
                "      max relative deviation over 8,000 steps  %.3e  "
                "(asymptote 1/alpha = %.1f)\n", worst, 1.0 / alpha);
    SentinelTest::check(worst < 1e-12,
                        "the recursion is the closed form to float64 resolution");
}

}  // namespace

int main() {
    std::printf("== golden vectors ==\n");

    const char* tiers[] = {"g1", "g2", "g3", "g4_0", "g4_1", "g4_2", "g4_3"};
    U32 ran = 0U;
    for (U32 i = 0U; i < (sizeof(tiers) / sizeof(tiers[0])); ++i) {
        if (runTier(tiers[i])) {
            ran += 1U;
        }
    }
    if (ran == 0U) {
        std::printf("    FAIL  no tier found under test/vectors\n");
        SentinelTest::failures += 1;
    } else {
        std::printf("    %u tier(s) run\n", ran);
    }

    reportDenominatorDrift();
    return SentinelTest::report("golden vectors");
}
