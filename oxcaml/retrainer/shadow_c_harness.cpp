// Section 72 / E5-e, HO1: the candidate crosses OUT of the OCaml process.
//
// (!) IT LINKS flight/ ITSELF, not a restatement. Sentinel::Detector is the same
// reader the F' component runs and the golden vectors pin, so HO1's band --
// "written, and Detector::load returns OK" -- is met here with NO FILE AT ALL.
// The file write below is for HO2's downlink to have something to carry; the
// band does not depend on it.
#include <cstdio>
#include <cstring>
#include <vector>

#include "sentinel_shadow.h"

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

bool spit(const char* path, const unsigned char* data, U32 n) {
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) { return false; }
    const size_t put = std::fwrite(data, 1U, static_cast<size_t>(n), f);
    std::fclose(f);
    return put == static_cast<size_t>(n);
}

U32 le32(const std::vector<unsigned char>& b, U32 off) {
    return static_cast<U32>(b[off]) | (static_cast<U32>(b[off + 1U]) << 8)
         | (static_cast<U32>(b[off + 2U]) << 16) | (static_cast<U32>(b[off + 3U]) << 24);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: s72 flying candidate\n");
        return 2;
    }
    std::vector<unsigned char> flying;
    if (!slurp(argv[1], flying)) { std::fprintf(stderr, "cannot read flying\n"); return 2; }
    int failures = 0;

    // Offset 36 is weights_bytes. ModelFile.hpp names HEADER_CRC_OFFSET but not
    // this one, so it is stated here with its source, as shadow59.ml:35 does.
    const U32 WEIGHTS_BYTES_OFFSET = 36U;
    const U32 wb = le32(flying, WEIGHTS_BYTES_OFFSET);
    const U32 nWeights = wb / 4U;
    std::printf("   flying %zu B, weights block %u B -> %u float32\n",
                flying.size(), wb, nWeights);

    // HS1: the flying bytes go IN.
    const int32_t inRc = sentinel_shadow_load(flying.data(), static_cast<uint32_t>(flying.size()));
    std::printf("   HS1  sentinel_shadow_load -> %d\n", inRc);
    if (inRc != SENTINEL_SHD_OK) { ++failures; }

    // HS2: new weights are written over the payload. Deterministic, so the
    // candidate is reproducible; the provenance of real weights is the cycle's
    // and is exercised on the F' side, not here.
    std::vector<float> w(static_cast<size_t>(nWeights));
    for (U32 i = 0U; i < nWeights; ++i) {
        w[i] = 0.001F * static_cast<float>((i % 997U)) - 0.5F;
    }
    const int32_t wrRc = sentinel_shadow_write(w.data(), nWeights);
    std::printf("   HS2  sentinel_shadow_write -> %d\n", wrRc);
    if (wrRc != SENTINEL_SHD_OK) { ++failures; }

    // HS3: the candidate comes OUT, and the call reports its own length.
    std::vector<unsigned char> cand(flying.size() + 64U, 0U);
    const int32_t n = sentinel_shadow_export(cand.data(), static_cast<uint32_t>(cand.size()));
    std::printf("   HS3  sentinel_shadow_export -> %d bytes\n", n);
    if (n <= 0) { ++failures; }
    const U32 candBytes = (n > 0) ? static_cast<U32>(n) : 0U;
    if (candBytes != static_cast<U32>(flying.size())) {
        std::printf("   HS3  (!) candidate %u B, flying %zu B -- the shapes should match\n",
                    candBytes, flying.size());
        ++failures;
    }

    // HO1's band, met with no file: flight/'s own reader on the exported bytes.
    static Detector detector;
    const LoadStatus st = detector.load(cand.data(), candBytes);
    std::printf("   HO1  Detector::load(exported bytes) -> %s\n", statusName(st));
    if (st != LoadStatus::OK) { ++failures; }

    // format_version is still 1. D30's freeze is not engaged.
    const U32 fv = static_cast<U32>(cand[4]) | (static_cast<U32>(cand[5]) << 8);
    std::printf("   HS4  format_version %u (D30's freeze: %s)\n", fv,
                (fv == Format::FORMAT_VERSION) ? "held" : "BROKEN");
    if (fv != Format::FORMAT_VERSION) { ++failures; }

    // (!) BOTH DIRECTIONS. Each refusal is exercised, not assumed.
    {
        std::vector<unsigned char> tiny(16U, 0U);
        const int32_t r = sentinel_shadow_export(tiny.data(), static_cast<uint32_t>(tiny.size()));
        std::printf("   HS5  export into a 16 B buffer -> %d (want %d)\n",
                    r, SENTINEL_SHD_ERR_SHAPE);
        if (r != SENTINEL_SHD_ERR_SHAPE) { ++failures; }
    }
    {
        const int32_t r = sentinel_shadow_write(w.data(), nWeights + 1U);
        std::printf("   HS6  write with one weight too many -> %d (want %d)\n",
                    r, SENTINEL_SHD_ERR_SHAPE);
        if (r != SENTINEL_SHD_ERR_SHAPE) { ++failures; }
    }
    {
        const int32_t r = sentinel_shadow_load(nullptr, 8U);
        std::printf("   HS7  load(nullptr) -> %d (want %d)\n", r, SENTINEL_SHD_ERR_NO_RUNTIME);
        if (r != SENTINEL_SHD_ERR_NO_RUNTIME) { ++failures; }
    }

    if (!spit(argv[2], cand.data(), candBytes)) {
        std::fprintf(stderr, "cannot write candidate\n");
        ++failures;
    }

    std::printf("   %s\n", failures == 0 ? "all HO1 checks passed" : "FAILURES ABOVE");
    return failures == 0 ? 0 : 1;
}
