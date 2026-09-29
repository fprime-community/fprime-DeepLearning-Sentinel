// D84 / D85 / MODELS 77-78: the cycle at a GENERATED shape -- HO1, the warm start,
// learning, and 56's window rule, each in both directions.
//
// 72's HO1 (`shadow_c_harness.cpp`) wrote synthetic weights of the flying file's
// own size and loaded the result. Here the weights come from the CYCLE, at the
// extents `sentinel_cycle_shape.h` was generated with:
//
//     flying file -> warm start -> cycle (MSE against the future block, D85)
//                 -> export -> shadow write -> export -> Detector::load
//
// (!) IT LINKS flight/ ITSELF, as 72's harness does. Sentinel::Detector is the
// reader the F' component runs, so "Detector::load returns OK" is HO1's band.
//
// (!) BOTH DIRECTIONS, every time:
//   - a flying file at ANY OTHER shape is refused, by the warm start and by the
//     shadow write (the weight count is unique across admissible shapes);
//   - the loss FALLS over repeated steps on a window with structure (D85: until
//     then the cycle's loss had no target and could not fall for a reason);
//   - 56's rule refuses at W-1 ticks and admits at W, and a single limit tick
//     inside the window refuses it again.
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>

#include "sentinel_cycle.h"
#include "sentinel_cycle_shape.h"
#include "sentinel_shadow.h"
#include "sentinel_window.h"

#include "sentinel/Detector.hpp"
#include "sentinel/ModelFile.hpp"

using namespace Sentinel;

namespace {

const U32 CHANNELS = SENTINEL_CYCLE_CHANNELS;
const U32 PREDICTIONS = SENTINEL_CYCLE_PREDICTIONS;
const U32 WINDOW = SENTINEL_CYCLE_WINDOW;
const U32 N_PARAMS = SENTINEL_CYCLE_N_PARAMS;
const U32 SPAN = WINDOW + PREDICTIONS;           // D85: input rows, then the target rows
const U32 W56 = 6550U;                           // window56.ml's W

float g_window[SPAN * CHANNELS];
float g_export[N_PARAMS];
float g_flying[N_PARAMS];

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("   %-66s %s\n", what, ok ? "ok" : "FAIL");
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

U32 le32(const std::vector<unsigned char>& b, U32 off)
{
    return static_cast<U32>(b[off]) | (static_cast<U32>(b[off + 1U]) << 8)
         | (static_cast<U32>(b[off + 2U]) << 16) | (static_cast<U32>(b[off + 3U]) << 24);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: shape_harness flying other_shape_flying candidate\n");
        return 2;
    }
    std::printf("== the cycle at the generated shape: %u channels, %u predictions, %u parameters ==\n",
                CHANNELS, PREDICTIONS, N_PARAMS);

    std::vector<unsigned char> flying;
    std::vector<unsigned char> other;
    check(slurp(argv[1], flying), "the flying file at this shape is readable");
    check(slurp(argv[2], other), "a flying file at another shape is readable");
    if (failures != 0) { return 1; }

    // A window with structure: a slow sinusoid per channel, so there is something
    // to forecast. The same values every run.
    for (U32 t = 0U; t < SPAN; ++t) {
        for (U32 c = 0U; c < CHANNELS; ++c) {
            g_window[(t * CHANNELS) + c] =
                0.5F * static_cast<float>(std::sin((static_cast<double>(t) / 23.0)
                                                   + static_cast<double>(c)));
        }
    }

    check(sentinel_cycle_boot() == SENTINEL_CYC_OK, "the runtime boots");
    check(sentinel_cycle_init(7) == SENTINEL_CYC_OK, "the cycle initialises");

    // -- D85 / D76: the warm start, and the other direction ----------------------
    check(sentinel_shadow_load(other.data(), static_cast<uint32_t>(other.size()))
              == SENTINEL_SHD_OK, "the other-shape flying bytes load into the shadow");
    check(sentinel_shadow_warm() == SENTINEL_SHD_ERR_SHAPE,
          "a warm start from a file at ANOTHER shape is REFUSED");
    check(sentinel_shadow_load(flying.data(), static_cast<uint32_t>(flying.size()))
              == SENTINEL_SHD_OK, "the flying bytes load into the shadow");
    check(sentinel_shadow_warm() == SENTINEL_SHD_OK, "the cycle warm-starts from the flying file");
    check(sentinel_cycle_export(g_export, N_PARAMS) == SENTINEL_CYC_OK,
          "the warm-started weights export");
    {
        const U32 cb = le32(flying, 32U);
        const U32 base = 64U + cb;
        bool same = true;
        for (U32 i = 0U; i < N_PARAMS; ++i) {
            U32 bits = le32(flying, base + (4U * i));
            float v = 0.0F;
            static_assert(sizeof(v) == sizeof(bits), "float32");
            std::memcpy(&v, &bits, sizeof(v));
            g_flying[i] = v;
            if (!(g_export[i] == v)) { same = false; }
        }
        check(same, "the warm start is the flying file's weights, bit for bit");
    }

    // -- D85: it LEARNS -- the MSE loss against the future block falls -----------
    check(sentinel_cycle_load(g_window, SPAN * CHANNELS) == SENTINEL_CYC_OK,
          "the window loads with its target block ((t_max + P) x C)");
    check(sentinel_cycle_load(g_window, WINDOW * CHANNELS) == SENTINEL_CYC_ERR_SHAPE,
          "a window without its target block is REFUSED");
    float loss0[1] = {0.0F};
    float lossN[1] = {0.0F};
    check(sentinel_cycle_run(1, static_cast<int32_t>(WINDOW), 1) == SENTINEL_CYC_OK, "step 1 runs");
    (void)sentinel_shadow_loss(loss0, 1U);
    for (int k = 0; k < 29; ++k) {
        (void)sentinel_cycle_run(1, static_cast<int32_t>(WINDOW), 1);
    }
    (void)sentinel_shadow_loss(lossN, 1U);
    std::printf("   loss at step 1 %.6e, at step 30 %.6e\n",
                static_cast<double>(loss0[0]), static_cast<double>(lossN[0]));
    check(lossN[0] < loss0[0], "the forecast loss FALLS over 30 steps on one window");
    check(sentinel_cycle_run(1, static_cast<int32_t>(WINDOW), 0) == SENTINEL_CYC_OK
              && sentinel_cycle_steps() == 0, "a refused window (admit 0) takes no step");
    check(sentinel_cycle_export(g_export, N_PARAMS) == SENTINEL_CYC_OK,
          "the trained weights export");
    {
        bool moved = false;
        for (U32 i = 0U; i < N_PARAMS; ++i) {
            if (!(g_export[i] == g_flying[i])) { moved = true; break; }
        }
        check(moved, "training moved the weights off the flying model's");
    }

    // -- HO1 at this shape, on the cycle's own trained weights -------------------
    check(sentinel_shadow_write(g_export, N_PARAMS) == SENTINEL_SHD_OK,
          "the cycle's weights fit the flying file");
    std::vector<unsigned char> cand(flying.size() + 64U, 0U);
    const int32_t n = sentinel_shadow_export(cand.data(), static_cast<uint32_t>(cand.size()));
    const U32 candBytes = (n > 0) ? static_cast<U32>(n) : 0U;
    check(candBytes == static_cast<U32>(flying.size()), "the candidate is the flying file's size");
    static Detector detector;
    const LoadStatus st = detector.load(cand.data(), candBytes);
    std::printf("   HO1  Detector::load(candidate) -> %s\n", statusName(st));
    check(st == LoadStatus::OK, "Detector::load accepts the candidate");
    check(detector.model().nChannels == CHANNELS, "the candidate declares this shape's channels");
    check(detector.model().nPredictions == PREDICTIONS,
          "the candidate declares this shape's predictions");
    check(sentinel_shadow_load(other.data(), static_cast<uint32_t>(other.size()))
              == SENTINEL_SHD_OK
              && sentinel_shadow_write(g_export, N_PARAMS) == SENTINEL_SHD_ERR_SHAPE,
          "the cycle's weights are REFUSED by a file at another shape");

    // -- D85: 56's window rule, crossed into C, both directions ------------------
    check(sentinel_window_reset() == SENTINEL_WIN_OK, "the window rule resets");
    check(sentinel_window_push(2, 0) == SENTINEL_WIN_ERR_ARG, "a push of 2 is REFUSED at the boundary");
    for (U32 t = 0U; t + 1U < W56; ++t) { (void)sentinel_window_push(0, 0); }
    check(sentinel_window_admits() == 0, "W - 1 quiet ticks: REFUSED (G-suff)");
    (void)sentinel_window_push(0, 0);
    check(sentinel_window_admits() == 1, "W quiet ticks: ADMITTED");
    (void)sentinel_window_push(1, 0);
    check(sentinel_window_admits() == 0, "one limit tick in the window: REFUSED (G-limit)");

    check(spit(argv[3], cand.data(), candBytes), "the candidate is written");
    std::printf("   %s\n", failures == 0 ? "the cycle at this shape: all checks passed"
                                          : "FAILURES ABOVE");
    return failures == 0 ? 0 : 1;
}
