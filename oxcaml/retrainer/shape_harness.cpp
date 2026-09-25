// D84 / MODELS 77: HO1 at a GENERATED shape -- the cycle's own weights, not synthetic.
//
// 72's HO1 (`shadow_c_harness.cpp`) wrote deterministic synthetic weights of the
// flying file's own size and loaded the result. That proved the crossing, and it
// could not prove the thing D83.1 left owed: that the CYCLE's output fits a model
// at a mission's shape. Here the weights come from `sentinel_cycle_export`, at the
// extents `sentinel_cycle_shape.h` was generated with, so the chain is
//
//     cycle (generated deep_f32) -> export -> shadow write -> export -> Detector::load
//
// (!) IT LINKS flight/ ITSELF, as 72's harness does. Sentinel::Detector is the
// reader the F' component runs, so "Detector::load returns OK" is the band.
//
// (!) BOTH DIRECTIONS. A flying file at ANY OTHER shape must be refused by
// `shadow59.ml`'s size check -- which is sufficient only because the weight count
// is unique across every admissible shape, and
// tests/test_retrainer_shape_is_generated.py asserts that rather than assuming it.
#include <cstdio>
#include <vector>

#include "sentinel_cycle.h"
#include "sentinel_cycle_shape.h"
#include "sentinel_shadow.h"

#include "sentinel/Detector.hpp"
#include "sentinel/ModelFile.hpp"

using namespace Sentinel;

namespace {

const U32 CHANNELS = SENTINEL_CYCLE_CHANNELS;
const U32 WINDOW = SENTINEL_CYCLE_WINDOW;
const U32 N_PARAMS = SENTINEL_CYCLE_N_PARAMS;

// CPP-1 is the component's rule, not this harness's; these are file-scope anyway
// so the extents are claimed once.
float g_window[WINDOW * CHANNELS];
float g_export[N_PARAMS];

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("   %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) { ++failures; }
}

bool slurp(const char* path, std::vector<unsigned char>& out)
{
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

bool spit(const char* path, const unsigned char* data, U32 n)
{
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) { return false; }
    const size_t put = std::fwrite(data, 1U, static_cast<size_t>(n), f);
    std::fclose(f);
    return put == static_cast<size_t>(n);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: shape_harness flying other_shape_flying candidate\n");
        return 2;
    }
    std::printf("== HO1 at the generated shape: %u channels, %u predictions, %u parameters ==\n",
                CHANNELS, SENTINEL_CYCLE_PREDICTIONS, N_PARAMS);

    std::vector<unsigned char> flying;
    std::vector<unsigned char> other;
    check(slurp(argv[1], flying), "the flying file at this shape is readable");
    check(slurp(argv[2], other), "a flying file at another shape is readable");
    if (failures != 0) { return 1; }

    // The cycle at this shape. The same deterministic drive as the component's
    // (Retrainer.cpp) -- not telemetry.
    check(sentinel_cycle_boot() == SENTINEL_CYC_OK, "the runtime boots");
    check(sentinel_cycle_init(7) == SENTINEL_CYC_OK, "a starting model is seeded");
    for (U32 t = 0U; t < WINDOW; ++t) {
        for (U32 c = 0U; c < CHANNELS; ++c) {
            const U32 mix = ((t * 2654435761U) + (c * 40503U)) & 0xFFFFU;
            g_window[(t * CHANNELS) + c] = static_cast<float>(mix) / 65535.0F;
        }
    }
    check(sentinel_cycle_load(g_window, WINDOW * CHANNELS) == SENTINEL_CYC_OK,
          "the window loads at this shape's extent");
    check(sentinel_cycle_run(1, 8, 1) == SENTINEL_CYC_OK, "one cycle runs");
    check(sentinel_cycle_export(g_export, N_PARAMS) == SENTINEL_CYC_OK,
          "the cycle's weights export at this shape's parameter count");

    // Into the flying file, and out as a candidate.
    check(sentinel_shadow_load(flying.data(), static_cast<uint32_t>(flying.size()))
              == SENTINEL_SHD_OK, "the flying bytes load into the shadow");
    check(sentinel_shadow_write(g_export, N_PARAMS) == SENTINEL_SHD_OK,
          "the cycle's weights fit the flying file");
    std::vector<unsigned char> cand(flying.size() + 64U, 0U);
    const int32_t n = sentinel_shadow_export(cand.data(), static_cast<uint32_t>(cand.size()));
    const U32 candBytes = (n > 0) ? static_cast<U32>(n) : 0U;
    check(candBytes == static_cast<U32>(flying.size()), "the candidate is the flying file's size");

    // HO1's band: flight/'s own reader accepts it, at this shape.
    static Detector detector;
    const LoadStatus st = detector.load(cand.data(), candBytes);
    std::printf("   HO1  Detector::load(candidate) -> %s\n", statusName(st));
    check(st == LoadStatus::OK, "Detector::load accepts the candidate");
    check(detector.model().nChannels == CHANNELS, "the candidate declares this shape's channels");
    check(detector.model().nPredictions == SENTINEL_CYCLE_PREDICTIONS,
          "the candidate declares this shape's predictions");

    // The other direction: a flying file at any other shape is refused.
    check(sentinel_shadow_load(other.data(), static_cast<uint32_t>(other.size()))
              == SENTINEL_SHD_OK, "the other-shape flying bytes load into the shadow");
    check(sentinel_shadow_write(g_export, N_PARAMS) == SENTINEL_SHD_ERR_SHAPE,
          "the cycle's weights are REFUSED by a file at another shape");

    check(spit(argv[3], cand.data(), candBytes), "the candidate is written");
    std::printf("   %s\n", failures == 0 ? "HO1 at this shape: all checks passed"
                                          : "FAILURES ABOVE");
    return failures == 0 ? 0 : 1;
}
