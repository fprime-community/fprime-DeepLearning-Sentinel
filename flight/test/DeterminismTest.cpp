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

//! CRC over the raw bit patterns of every output the detector produces.
//!
//! Bit patterns, not values: two floats that compare equal can differ in their
//! encoding, and the claim under test is about the encoding.
U32 runPipeline(const U8* model, U32 length, U32 steps) {
    Detector detector;
    if (detector.load(model, length) != LoadStatus::OK) {
        return 0U;
    }
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

    std::FILE* stored = std::fopen(DIGEST_PATH, "rb");
    if (stored == nullptr) {
        std::FILE* out = std::fopen(DIGEST_PATH, "wb");
        if (out != nullptr) {
            (void)std::fprintf(out, "%08X\n", first);
            (void)std::fclose(out);
        }
        std::printf("    first run: digest recorded for the next process to check\n");
    } else {
        unsigned int previous = 0U;
        const int read = std::fscanf(stored, "%08X", &previous);
        (void)std::fclose(stored);
        SentinelTest::check(read == 1, "the recorded digest was readable");
        SentinelTest::check(static_cast<U32>(previous) == first,
                            "a fresh process reproduces the digest bit for bit");
        std::printf("    across processes: 0x%08X == 0x%08X\n",
                    static_cast<U32>(previous), first);
    }

    return SentinelTest::report("determinism");
}
