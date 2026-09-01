// The loader refuses, with a status code and no exception.
//
// This is the hook work item 9's Level 1 safe failure mode is built on (D5,
// Objective.md 14.10): a corrupt file, a version mismatch, a failed CRC or a
// radiation bit-flip must degrade with an event and never fail the topology.
// Nothing here can throw -- the whole build is `-fno-exceptions`.
#include <cstdio>
#include <cstring>

#include "sentinel/Crc32.hpp"
#include "sentinel/Detector.hpp"
#include "TestSupport.hpp"

namespace SentinelTest {
int failures = 0;
}

using namespace Sentinel;
using SentinelTest::check;

namespace {

U8 g_original[SentinelTest::MAX_FILE_BYTES];
U8 g_working[SentinelTest::MAX_FILE_BYTES];
U32 g_length = 0U;
U32 g_cases = 0U;

void writeU16(U8* p, U32 value) {
    p[0] = static_cast<U8>(value & 0xFFU);
    p[1] = static_cast<U8>((value >> 8U) & 0xFFU);
}

void writeU32(U8* p, U32 value) {
    p[0] = static_cast<U8>(value & 0xFFU);
    p[1] = static_cast<U8>((value >> 8U) & 0xFFU);
    p[2] = static_cast<U8>((value >> 16U) & 0xFFU);
    p[3] = static_cast<U8>((value >> 24U) & 0xFFU);
}

//! Restore the pristine file, so each case starts from a valid one.
void restore() {
    std::memcpy(g_working, g_original, static_cast<size_t>(g_length));
}

//! Re-sign the header, so a field test reaches the check it is aiming at rather
//! than stopping at BAD_HEADER_CRC.
void resign() {
    writeU32(g_working + Format::HEADER_CRC_OFFSET,
             crc32(g_working, Format::HEADER_CRC_OFFSET));
}

void expect(LoadStatus wanted, const char* what) {
    g_cases += 1U;
    Model model;
    const LoadStatus got = model.load(g_working, g_length);
    if (got != wanted) {
        std::printf("    FAIL  %s: got %s, expected %s\n", what, statusName(got),
                    statusName(wanted));
        SentinelTest::failures += 1;
    }
    if (got != LoadStatus::OK) {
        check(!model.loaded(), "a refused model must not report itself loaded");
    }
}

}  // namespace

int main() {
    std::printf("== refusals ==\n");

    // The CRC table is built, not typed out, so its check value is the proof it
    // is right. `docs/MODEL_FILE.md` 7.
    const char* probe = "123456789";
    const U32 checkValue = crc32(reinterpret_cast<const U8*>(probe), 9U);
    if (checkValue != 0xCBF43926U) {
        std::printf("    FAIL  CRC-32/IEEE 802.3 check value: got 0x%08X, "
                    "expected 0xCBF43926\n", checkValue);
        SentinelTest::failures += 1;
    }

    g_length = SentinelTest::readFile("test/vectors/g1.bin", g_original,
                                      SentinelTest::MAX_FILE_BYTES);
    if (g_length == 0U) {
        std::printf("    FAIL  test/vectors/g1.bin not readable\n");
        return 1;
    }
    SentinelTest::checkEqualU32(g_length, 1276U, "g1.bin size");

    restore();
    expect(LoadStatus::OK, "the pristine file loads");

    restore();
    g_working[0] = 'X';
    expect(LoadStatus::BAD_MAGIC, "bad magic");

    restore();
    writeU16(g_working + 4, Format::FORMAT_VERSION + 1U);
    expect(LoadStatus::BAD_VERSION, "bad version (checked before the header CRC)");

    restore();
    g_working[20] ^= 0xFFU;                      // a header byte, CRC left stale
    expect(LoadStatus::BAD_HEADER_CRC, "bad header CRC");

    restore();
    // A corrupted length with a stale CRC must be caught by the CRC, not acted on.
    writeU32(g_working + 36, 0xFFFFFFF0U);
    expect(LoadStatus::BAD_HEADER_CRC,
           "the header CRC is checked before any length field is used");

    restore();
    g_working[Format::HEADER_BYTES + 40U] ^= 0xFFU;   // inside the weight block
    expect(LoadStatus::BAD_STATIC_CRC, "bad static CRC (a weight bit-flip)");

    restore();
    g_working[g_length - 4U] ^= 0xFFU;               // inside the parameter block
    expect(LoadStatus::BAD_PARAM_CRC, "bad param CRC (a threshold bit-flip)");

    restore();
    writeU16(g_working + 8, 2U);                     // arch_id = LSTM
    resign();
    expect(LoadStatus::BAD_ARCH, "a non-GRU architecture is refused by name");

    restore();
    writeU16(g_working + 10, 2U);                    // the LSTM's gate order
    resign();
    expect(LoadStatus::BAD_GATE_ORDER, "a wrong gate order is refused by name");

    restore();
    writeU32(g_working + 52, 1U);                    // reserved0
    resign();
    expect(LoadStatus::BAD_SHAPE, "a non-zero reserved field is refused");

    restore();
    writeU16(g_working + 14, 20U);                   // n_channels, past MAX_CHANNELS
    writeU16(g_working + 16, 20U);                   // n_inputs, kept coherent
    resign();
    expect(LoadStatus::TOO_LARGE, "a model past the compile-time maxima is refused");

    restore();
    writeU16(g_working + 24, Config::MAX_HIDDEN + 1U);
    resign();
    expect(LoadStatus::TOO_LARGE, "a hidden size past MAX_HIDDEN is refused");

    restore();
    writeU16(g_working + 14, 4U);                    // n_channels 3 -> 4
    writeU16(g_working + 16, 4U);
    resign();
    expect(LoadStatus::BAD_SHAPE,
           "a shape that does not account for the payload is refused");

    restore();
    {
        const U32 paramLo = g_length - (Format::PARAM_FIXED_BYTES + (8U * 3U));
        writeU16(g_working + paramLo + 2U, 1U);      // norm_policy != identity
        writeU32(g_working + 48,
                 crc32(g_working + paramLo, g_length - paramLo));
        resign();
        expect(LoadStatus::BAD_NORM_POLICY, "a non-identity normalisation policy");
    }

    {
        const U32 trueLength = g_length;
        restore();
        g_length = trueLength - 1U;
        expect(LoadStatus::TRUNCATED, "a truncated file");
        g_length = 10U;
        expect(LoadStatus::TRUNCATED, "a file shorter than its own header");
        g_length = trueLength;                   // later cases need the real one
    }

    {
        Model model;
        check(model.load(nullptr, 1000U) == LoadStatus::TRUNCATED,
              "a null buffer is refused rather than dereferenced");
    }

    // A detector whose load failed must stay inert: no state, no emission. That
    // is what "never fail the topology" needs from this layer.
    {
        restore();
        g_working[0] = 'X';
        Detector detector;
        check(detector.load(g_working, g_length) == LoadStatus::BAD_MAGIC,
              "the detector propagates the loader's status");
        F32 values[Config::MAX_CHANNELS];
        for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
            values[c] = 0.5F;
        }
        detector.step(values, true);
        check(!detector.emitted(), "a refused model emits nothing");
        check(detector.steps() == 0U, "a refused model advances no state");
    }

    std::printf("    %u load cases, CRC check value 0x%08X\n", g_cases, checkValue);
    return SentinelTest::report("refusals");
}
