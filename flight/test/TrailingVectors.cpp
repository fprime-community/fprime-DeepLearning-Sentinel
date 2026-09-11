// `TrailingWindow` against `scripts/decision_layer_arms.py`'s `trailing_stats`,
// tiers T1-T3. Tolerance 1e-5 on the moments (`docs/MODELS.md` 39.9, N1).
//
// The two compute the same trailing window by different means, which is the
// whole point of the comparison: the reference differences float64 prefix sums
// over the entire series, the flight core carries a running window and subtracts
// the sample that leaves it. Agreement to 1e-5 says the running form is sound;
// T3 is the tier that would say otherwise, because a large offset with small
// variation (N(1000, 3)) is the regime D37's float32 defect destroyed --
// 7.6584e+00 of error against a true sigma of 3.0.
//
// Format is written at the head of `scripts/make_trailing_vectors.py`.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "sentinel/Config.hpp"
#include "sentinel/TrailingWindow.hpp"
#include "TestSupport.hpp"

namespace SentinelTest {
int failures = 0;
}

using namespace Sentinel;

namespace {

constexpr F64 TOLERANCE = 1e-5;

struct Header {
    U16 version;
    U16 channels;
    U32 steps;
    U32 span;
    U32 bodyOffset;
};

bool readHeader(const U8* data, U32 length, Header& out) {
    if ((data == nullptr) || (length < 16U)) {
        return false;
    }
    if (std::memcmp(data, "SNTT", 4U) != 0) {
        return false;
    }
    std::memcpy(&out.version, data + 4, sizeof(out.version));
    std::memcpy(&out.channels, data + 6, sizeof(out.channels));
    std::memcpy(&out.steps, data + 8, sizeof(out.steps));
    std::memcpy(&out.span, data + 12, sizeof(out.span));
    out.bodyOffset = 16U + (static_cast<U32>(out.channels) * 4U);   // past scale[]
    return true;
}

F32 readF32(const U8* p) {
    F32 v = 0.0F;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

F64 readF64(const U8* p) {
    F64 v = 0.0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

//! One tier, streamed a record at a time. Returns the worst absolute difference.
//!
//! (!) STREAMED, NOT SLURPED, AND THE REASON IS A MEASUREMENT. `TestSupport`'s
//! `readFile` fills a 512 KiB buffer; tier T3 is 552,064 B. Reading it whole
//! would have truncated it silently -- `readFile` returns what it read and a
//! short read looks like a short file -- and the comparison would then have run
//! against the wrong bytes. A record here is at most
//! `MAX_CHANNELS * 20` = 320 bytes, so the whole file never has to be resident,
//! and `docs/MODELS.md` 39.7's larger decision-layer vectors will not hit the
//! same wall.
F64 runTier(const char* tier, const char* path, U32& tiersRun) {
    std::FILE* handle = std::fopen(path, "rb");
    if (handle == nullptr) {
        std::printf("    %-4s SKIPPED -- %s absent\n", tier, path);
        return 0.0;
    }

    U8 head[16];
    if (std::fread(head, 1U, sizeof(head), handle) != sizeof(head)) {
        (void)std::fclose(handle);
        SentinelTest::check(false, "vector file is shorter than its header");
        return 0.0;
    }
    Header h{};
    if (!readHeader(head, sizeof(head), h)) {
        (void)std::fclose(handle);
        SentinelTest::check(false, "vector header is not a SNTT v1 header");
        return 0.0;
    }
    SentinelTest::checkEqualU32(h.span, Config::ERROR_WINDOW,
                                "the vector's span is the core's ERROR_WINDOW");
    SentinelTest::check(h.channels <= Config::MAX_CHANNELS,
                        "the vector's width fits the core's maxima");

    // Past the per-channel scale block, which is provenance and not input.
    if (std::fseek(handle, static_cast<long>(h.bodyOffset), SEEK_SET) != 0) {
        (void)std::fclose(handle);
        SentinelTest::check(false, "could not seek to the vector body");
        return 0.0;
    }

    TrailingWindow window;
    window.configure(h.channels, h.span);

    const U32 record = static_cast<U32>(h.channels) * 20U;   // F32 + 2 x F64
    U8 row[Config::MAX_CHANNELS * 20U];
    F32 values[Config::MAX_CHANNELS];

    if (record > sizeof(row)) {
        (void)std::fclose(handle);
        SentinelTest::check(false,
                            "the vector's record is wider than the reader's buffer");
        return 0.0;
    }

    F64 worst = 0.0;
    U32 worstStep = 0U;
    const char* worstWhat = "-";
    U32 read = 0U;

    for (U32 t = 0U; t < h.steps; ++t) {
        if (std::fread(row, 1U, static_cast<size_t>(record), handle)
                != static_cast<size_t>(record)) {
            break;
        }
        ++read;
        for (U32 c = 0U; c < h.channels; ++c) {
            values[c] = readF32(row + (c * 4U));
        }
        window.push(values);

        const U8* means = row + (static_cast<U32>(h.channels) * 4U);
        const U8* sds = means + (static_cast<U32>(h.channels) * 8U);
        for (U32 c = 0U; c < h.channels; ++c) {
            const F64 dMean = std::fabs(window.mean(c) - readF64(means + (c * 8U)));
            const F64 dSd = std::fabs(window.sd(c) - readF64(sds + (c * 8U)));
            if (dMean > worst) { worst = dMean; worstStep = t; worstWhat = "mean"; }
            if (dSd > worst) { worst = dSd; worstStep = t; worstWhat = "sd"; }
        }
    }
    (void)std::fclose(handle);

    SentinelTest::checkEqualU32(read, h.steps, "every step in the file was read");
    std::printf("    %-4s %2u ch x %4u steps   max |diff| %.3e  (%s, step %u)\n",
                tier, static_cast<unsigned>(h.channels),
                static_cast<unsigned>(read), worst, worstWhat,
                static_cast<unsigned>(worstStep));
    SentinelTest::check(worst <= TOLERANCE, "tier within 1e-5 of the reference");
    ++tiersRun;
    return worst;
}

//! The ring must wrap during the run, or the tiers prove nothing about the
//! subtraction. Asserted rather than assumed.
void checkTheRingWraps(U32 steps) {
    SentinelTest::check(steps > Config::ERROR_WINDOW,
                        "the tier runs past the window, so the ring wraps");
}

//! `at()` returns the window oldest-first, which every rule reading it needs.
void checkOrdering() {
    TrailingWindow window;
    window.configure(1U, Config::ERROR_WINDOW);
    for (U32 t = 0U; t < Config::ERROR_WINDOW + 37U; ++t) {
        const F32 sample = static_cast<F32>(t);
        window.push(&sample);
    }
    SentinelTest::checkEqualU32(window.filled(), Config::ERROR_WINDOW,
                                "a wrapped window is exactly its span deep");
    const F32 oldest = window.at(0U, 0U);
    const F32 newest = window.at(0U, window.filled() - 1U);
    SentinelTest::check(oldest < newest, "at(0) is older than at(filled - 1)");
    SentinelTest::check(
        std::fabs(static_cast<F64>(newest) - static_cast<F64>(Config::ERROR_WINDOW + 36U))
            < 0.5,
        "the newest sample is the one just pushed");
    SentinelTest::check(
        std::fabs(static_cast<F64>(oldest) - 37.0) < 0.5,
        "the oldest sample is the first one still inside the window");
}

//! A width past the maxima goes inert instead of reading past an array.
void checkRefusal() {
    TrailingWindow window;
    window.configure(Config::MAX_CHANNELS + 1U, Config::ERROR_WINDOW);
    SentinelTest::checkEqualU32(window.nChannels(), 0U,
                                "an oversized width configures to zero channels");
    const F32 sample = 1.0F;
    window.push(&sample);
    SentinelTest::check(window.steps() == 0U, "an inert window never advances");
    SentinelTest::check(window.mean(0U) == 0.0, "an inert window scores zero");

    TrailingWindow deep;
    deep.configure(1U, Config::SOLVE_WINDOW + 1U);
    SentinelTest::checkEqualU32(deep.nChannels(), 0U,
                                "a span past the ring configures to zero channels");
}

}  // namespace

int main() {
    std::printf("== trailing window ==\n");

    U32 tiers = 0U;
    F64 worst = 0.0;
    const char* paths[3] = {"test/vectors/t1.tvec",
                            "test/vectors/t2.tvec",
                            "test/vectors/t3.tvec",};
    const char* names[3] = {"t1", "t2", "t3"};
    for (U32 i = 0U; i < 3U; ++i) {
        const F64 seen = runTier(names[i], paths[i], tiers);
        worst = std::max(worst, seen);
    }
    std::printf("    %u tier(s) run\n", static_cast<unsigned>(tiers));
    std::printf("    worst over all tiers: %.3e, tolerance %.0e\n", worst, TOLERANCE);
    SentinelTest::check(tiers == 3U, "every committed tier ran; none was skipped");

    checkTheRingWraps(Config::ERROR_WINDOW + 200U);
    SentinelTest::checkEqualU32(Config::SOLVE_WINDOW,
                                Config::ERROR_WINDOW + Config::STRIDE,
                                "the solve window is history plus the judged segment");
    checkOrdering();
    checkRefusal();

    return SentinelTest::report("trailing window");
}
