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
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "sentinel/Config.hpp"
#include "sentinel/DerivativeStream.hpp"
#include "sentinel/Detector.hpp"
#include "sentinel/Crc32.hpp"
#include "sentinel/DynamicThreshold.hpp"
#include "TestSupport.hpp"

namespace SentinelTest {
int failures = 0;
}

using namespace Sentinel;

namespace {

constexpr F64 TOLERANCE = 1e-5;

void writeU32(U8* p, U32 v) { std::memcpy(p, &v, sizeof(v)); }

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

    if (record > sizeof(row)) {
        (void)std::fclose(handle);
        SentinelTest::check(false,
                            "the vector's record is wider than the reader's buffer");
        return 0.0;
    }

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

//! `DerivativeStream` against `scripts/decision_layer_arms.py:224`.
F64 runDerivativeTier(const char* tier, const char* path, U32& tiersRun) {
    std::FILE* handle = std::fopen(path, "rb");
    if (handle == nullptr) {
        std::printf("    %-4s SKIPPED -- %s absent\n", tier, path);
        return 0.0;
    }
    U8 head[16];
    if (std::fread(head, 1U, sizeof(head), handle) != sizeof(head)
        || (std::memcmp(head, "SNTF", 4U) != 0)) {
        (void)std::fclose(handle);
        SentinelTest::check(false, "vector is not a SNTF v1 header");
        return 0.0;
    }
    U16 channels = 0U;
    U32 steps = 0U;
    U32 span = 0U;
    std::memcpy(&channels, head + 6, sizeof(channels));
    std::memcpy(&steps, head + 8, sizeof(steps));
    std::memcpy(&span, head + 12, sizeof(span));
    SentinelTest::checkEqualU32(span, Config::ERROR_WINDOW,
                                "the derivative standardises over ERROR_WINDOW");

    DerivativeStream stream;
    stream.configure(channels);

    const U32 record = static_cast<U32>(channels) * 12U;
    U8 row[Config::MAX_CHANNELS * 12U];
    F32 values[Config::MAX_CHANNELS];
    F64 worst = 0.0;
    U32 worstStep = 0U;
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
        stream.step(values);
        const U8* zRow = row + (static_cast<U32>(channels) * 4U);
        for (U32 c = 0U; c < channels; ++c) {
            const F64 diff = std::fabs(stream.z(c) - readF64(zRow + (c * 8U)));
            if (diff > worst) { worst = diff; worstStep = t; }
        }
    }
    (void)std::fclose(handle);

    SentinelTest::checkEqualU32(read, steps, "every step in the file was read");
    std::printf("    %-4s %2u ch x %4u steps   z_d max |diff| %.3e (step %u)\n",
                tier, static_cast<unsigned>(channels), static_cast<unsigned>(read),
                worst, static_cast<unsigned>(worstStep));
    SentinelTest::check(worst <= TOLERANCE, "z_derivative within 1e-5");
    ++tiersRun;
    return worst;
}

//! `dx[0]` is zero, not a step from nothing. `np.diff(x, prepend=x[0])`.
void checkTheFirstDifferenceIsZero() {
    DerivativeStream stream;
    stream.configure(2U);
    const F32 first[2] = {1000.0F, -7.5F};
    stream.step(first);
    SentinelTest::check(stream.delta(0U) == 0.0, "dx[0] is exactly zero");
    SentinelTest::check(stream.delta(1U) == 0.0, "dx[0] is zero on every channel");
    const F32 second[2] = {1002.0F, -7.5F};
    stream.step(second);
    SentinelTest::check(std::fabs(stream.delta(0U) - 2.0) < 1e-9,
                        "dx is the absolute first difference");
    SentinelTest::check(stream.delta(1U) == 0.0, "an unchanged channel has no derivative");
}

//! The fused statistic is the larger of the two streams, and never in the flag.
void checkFusionAndThatItDoesNotEmit() {
    SentinelTest::check(Config::ERROR_WINDOW < Config::SOLVE_WINDOW,
                        "the moment window is shorter than the solve window");
    Detector detector;
    // Unloaded: inert by construction, which is the property that matters here.
    SentinelTest::check(!detector.emitted(), "an unloaded detector never emits");
    SentinelTest::check(!detector.dynamicEmitted(),
                        "an unloaded detector's dynamic rule never emits either");
    SentinelTest::check(detector.fusedScore() == 0.0,
                        "the fused score starts at zero");
}

//! D68's flight configuration, end to end: bytes on disk -> a flag.
//!
//! Not the streams in isolation -- `t*.tvec`, `d*.dvec` and `f*.fvec` already
//! cover those -- but a real `model.bin` at `param_version` 2 loaded into a
//! `Detector`, stepped, and checked step by step.
F64 runFusedTier(const char* path, const char* modelPath, U32& tiersRun) {
    std::FILE* handle = std::fopen(path, "rb");
    if (handle == nullptr) {
        std::printf("    p1   SKIPPED -- %s absent\n", path);
        return 0.0;
    }
    U8 head[20];
    if (std::fread(head, 1U, sizeof(head), handle) != sizeof(head)
        || (std::memcmp(head, "SNTP", 4U) != 0)) {
        (void)std::fclose(handle);
        SentinelTest::check(false, "vector is not a SNTP v1 header");
        return 0.0;
    }
    U16 channels = 0U;
    U32 steps = 0U;
    F64 threshold = 0.0;
    std::memcpy(&channels, head + 6, sizeof(channels));
    std::memcpy(&steps, head + 8, sizeof(steps));
    std::memcpy(&threshold, head + 12, sizeof(threshold));

    static U8 model[SentinelTest::MAX_FILE_BYTES];
    const U32 modelBytes = SentinelTest::readFile(modelPath, model, sizeof(model));
    SentinelTest::check(modelBytes > 0U, "the paired model.bin was read");

    Detector detector;
    const LoadStatus status = detector.load(model, modelBytes);
    SentinelTest::check(status == LoadStatus::OK, "a param_version 2 file loads");
    SentinelTest::checkEqualU32(detector.model().paramVersion,
                                Format::PARAM_VERSION_FUSED,
                                "and it reports the version it carries");
    SentinelTest::check(detector.model().threshold == threshold,
                        "the cut in the file is the cut the vector expects");

    const U32 record = (static_cast<U32>(channels) * 4U) + 10U;
    U8 row[(Config::MAX_CHANNELS * 4U) + 10U];
    F32 values[Config::MAX_CHANNELS];
    F64 worst = 0.0;
    U32 worstStep = 0U;
    U32 flagDiffs = 0U;
    U32 emitted = 0U;
    U32 expectedEmitted = 0U;
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
        detector.step(values, true);

        const F64 expectedScore = readF64(row + (static_cast<U32>(channels) * 4U));
        const bool expectedCrossing = (row[record - 2U] != 0U);
        const bool expectedEmit = (row[record - 1U] != 0U);

        const F64 diff = std::fabs(detector.fusedScore() - expectedScore);
        if (diff > worst) { worst = diff; worstStep = t; }
        if (detector.crossing() != expectedCrossing) { ++flagDiffs; }
        if (detector.emitted() != expectedEmit) { ++flagDiffs; }
        if (detector.emitted()) { ++emitted; }
        if (expectedEmit) { ++expectedEmitted; }
    }
    (void)std::fclose(handle);

    SentinelTest::checkEqualU32(read, steps, "every step in the file was read");
    std::printf("    p1   %2u ch x %4u steps   fused max |diff| %.3e (step %u)   "
                "emitted %u of %u\n",
                static_cast<unsigned>(channels), static_cast<unsigned>(read),
                worst, static_cast<unsigned>(worstStep),
                static_cast<unsigned>(emitted),
                static_cast<unsigned>(expectedEmitted));
    SentinelTest::check(worst <= TOLERANCE, "the fused score is within 1e-5");
    SentinelTest::checkEqualU32(flagDiffs, 0U, "crossing and emitted are exact");
    SentinelTest::check(expectedEmitted > 20U,
                        "the tier emits often enough for the flag to mean something");
    ++tiersRun;
    return worst;
}

//! (!) The version is not decoration: the same threshold cuts a different
//! statistic, so the two PARAMS generations must disagree on the flags.
//!
//! Patched in place rather than generated as a second file, so the ONLY
//! difference between the two runs is the two bytes naming the version.
void checkTheVersionChangesTheDecision() {
    static U8 model[SentinelTest::MAX_FILE_BYTES];
    const U32 bytes = SentinelTest::readFile("test/vectors/p1.bin", model, sizeof(model));
    if (bytes == 0U) {
        return;
    }
    Detector fusedDetector;
    SentinelTest::check(fusedDetector.load(model, bytes) == LoadStatus::OK,
                        "the version-2 file loads");
    const U32 channels = fusedDetector.model().nChannels;
    const U32 paramLo = bytes - (Format::PARAM_FIXED_BYTES + (8U * channels));

    static U8 patched[SentinelTest::MAX_FILE_BYTES];
    for (U32 i = 0U; i < bytes; ++i) { patched[i] = model[i]; }
    patched[paramLo] = 1U;                      // param_version 2 -> 1
    patched[paramLo + 1U] = 0U;
    writeU32(patched + 48, crc32(patched + paramLo, bytes - paramLo));
    writeU32(patched + Format::HEADER_CRC_OFFSET,
             crc32(patched, Format::HEADER_CRC_OFFSET));

    Detector staticDetector;
    SentinelTest::check(staticDetector.load(patched, bytes) == LoadStatus::OK,
                        "the patched version-1 file loads too");

    // Driven over the tier's own inputs, not a short synthetic run: 600 steps of
    // a sine cross neither rule, so a comparison there would have proved only
    // that both stayed silent.
    std::FILE* vec = std::fopen("test/vectors/p1.pvec", "rb");
    if (vec == nullptr) {
        return;
    }
    U8 vhead[20];
    if (std::fread(vhead, 1U, sizeof(vhead), vec) != sizeof(vhead)) {
        (void)std::fclose(vec);
        return;
    }
    U32 steps = 0U;
    std::memcpy(&steps, vhead + 8, sizeof(steps));
    const U32 record = (static_cast<U32>(channels) * 4U) + 10U;

    U32 disagreements = 0U;
    U32 fusedCrossings = 0U;
    U32 staticCrossings = 0U;
    U8 row[(Config::MAX_CHANNELS * 4U) + 10U];
    F32 values[Config::MAX_CHANNELS];
    for (U32 t = 0U; t < steps; ++t) {
        if (std::fread(row, 1U, static_cast<size_t>(record), vec)
                != static_cast<size_t>(record)) {
            break;
        }
        for (U32 c = 0U; c < channels; ++c) {
            values[c] = readF32(row + (c * 4U));
        }
        fusedDetector.step(values, true);
        staticDetector.step(values, true);
        if (fusedDetector.crossing()) { ++fusedCrossings; }
        if (staticDetector.crossing()) { ++staticCrossings; }
        if (fusedDetector.crossing() != staticDetector.crossing()) {
            ++disagreements;
        }
    }
    (void)std::fclose(vec);

    std::printf("    same bytes, same cut: version 2 crosses %u, version 1 crosses %u, "
                "they differ on %u of %u steps\n",
                static_cast<unsigned>(fusedCrossings),
                static_cast<unsigned>(staticCrossings),
                static_cast<unsigned>(disagreements), static_cast<unsigned>(steps));
    SentinelTest::check(disagreements > 0U,
                        "the two PARAMS versions decide differently on the same bytes");
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
        worst = std::max(worst, seen);
    }
    std::printf("    %u tier(s) run\n", static_cast<unsigned>(tiers));
    std::printf("    worst eps over all tiers: %.3e, tolerance %.0e\n", worst, TOLERANCE);
    SentinelTest::check(tiers == 2U, "every committed tier ran; none was skipped");

    std::printf("== derivative stream ==\n");
    U32 fTiers = 0U;
    F64 fWorst = 0.0;
    const char* fPaths[2] = {"test/vectors/f1.fvec", "test/vectors/f2.fvec"};
    const char* fNames[2] = {"f1", "f2"};
    for (U32 i = 0U; i < 2U; ++i) {
        const F64 seen = runDerivativeTier(fNames[i], fPaths[i], fTiers);
        fWorst = std::max(fWorst, seen);
    }
    std::printf("    %u tier(s) run, worst %.3e, tolerance %.0e\n",
                static_cast<unsigned>(fTiers), fWorst, TOLERANCE);
    SentinelTest::check(fTiers == 2U, "every committed derivative tier ran");
    checkTheFirstDifferenceIsZero();
    checkFusionAndThatItDoesNotEmit();

    std::printf("== D68 flight configuration ==\n");
    U32 pTiers = 0U;
    (void)runFusedTier("test/vectors/p1.pvec", "test/vectors/p1.bin", pTiers);
    SentinelTest::check(pTiers == 1U, "the D68 tier ran; it was not skipped");
    checkTheVersionChangesTheDecision();

    checkSilence();
    checkEmissionIsAnInstant();
    checkRefusal();

    return SentinelTest::report("dynamic threshold");
}
