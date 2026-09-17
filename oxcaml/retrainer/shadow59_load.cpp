// Section 59 / E5-a, MW2, MW4, MW5, MW6: the C++ side of the shadow file.
//
// (!) IT LINKS flight/ ITSELF, not a restatement. Sentinel::ModelFile is the same
// reader the F' component runs and the golden vectors pin.
#include <cstdio>
#include <cstring>
#include <vector>

#include "sentinel/Detector.hpp"
#include "sentinel/ModelFile.hpp"

using namespace Sentinel;

namespace {
bool slurp(const char* path, std::vector<unsigned char>& out) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { return false; }
    unsigned char buf[65536];
    size_t got = 0U;
    while ((got = std::fread(buf, 1U, sizeof(buf), f)) > 0U) {
        out.insert(out.end(), buf, buf + got);
    }
    std::fclose(f);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: shadow59_load flying shadow\n"); return 2; }
    std::vector<unsigned char> flying, shadow;
    if (!slurp(argv[1], flying) || !slurp(argv[2], shadow)) {
        std::fprintf(stderr, "cannot read inputs\n"); return 2;
    }
    int failures = 0;

    // MW2: the reader loads it unaided.
    static Detector detector;
    const LoadStatus st = detector.load(shadow.data(), static_cast<U32>(shadow.size()));
    std::printf("   MW2  Detector::load(shadow) -> %s\n", statusName(st));
    if (st != LoadStatus::OK) { ++failures; }

    // MW6: format_version is still 1.
    const U32 fv = static_cast<U32>(shadow[4]) | (static_cast<U32>(shadow[5]) << 8);
    std::printf("   MW6  format_version %u (D30's freeze: %s)\n", fv,
                (fv == Format::FORMAT_VERSION) ? "held" : "BROKEN");
    if (fv != Format::FORMAT_VERSION) { ++failures; }

    // MW4: only the weights payload and the two CRCs differ.
    if (flying.size() != shadow.size()) {
        std::printf("   MW4  sizes differ: %zu vs %zu\n", flying.size(), shadow.size());
        ++failures;
    } else {
        const U32 cb = static_cast<U32>(shadow[32]) | (static_cast<U32>(shadow[33]) << 8)
                     | (static_cast<U32>(shadow[34]) << 16) | (static_cast<U32>(shadow[35]) << 24);
        const U32 wb = static_cast<U32>(shadow[36]) | (static_cast<U32>(shadow[37]) << 8)
                     | (static_cast<U32>(shadow[38]) << 16) | (static_cast<U32>(shadow[39]) << 24);
        const U32 wLo = 64U + cb, wHi = wLo + wb;
        U32 outside = 0U, insideSame = 0U;
        for (U32 i = 0U; i < shadow.size(); ++i) {
            const bool differs = flying[i] != shadow[i];
            const bool expected = ((i >= wLo) && (i < wHi))
                               || ((i >= 44U) && (i < 48U))
                               || ((i >= 60U) && (i < 64U));
            if (differs && !expected) { ++outside; }
            if (!differs && (i >= wLo) && (i < wHi)) { ++insideSame; }
        }
        std::printf("   MW4  bytes differing outside {weights, 44-47, 60-63}: %u   "
                    "(weights region %u B, %u unchanged within it)\n", outside, wb, insideSame);
        if (outside != 0U) { ++failures; }
    }

    // MW5: the static CRC still guards. Flip one weights byte after the write.
    {
        std::vector<unsigned char> bad = shadow;
        const U32 cb = static_cast<U32>(bad[32]) | (static_cast<U32>(bad[33]) << 8)
                     | (static_cast<U32>(bad[34]) << 16) | (static_cast<U32>(bad[35]) << 24);
        bad[64U + cb] = static_cast<unsigned char>(bad[64U + cb] ^ 0xFFU);
        static Detector d2;
        const LoadStatus s2 = d2.load(bad.data(), static_cast<U32>(bad.size()));
        std::printf("   MW5  one weights byte flipped -> %s\n", statusName(s2));
        if (s2 != LoadStatus::BAD_STATIC_CRC) { ++failures; }
    }

    std::printf("   %s\n", failures == 0 ? "all C++ side checks passed"
                                         : "FAILURES ABOVE");
    return failures == 0 ? 0 : 1;
}
