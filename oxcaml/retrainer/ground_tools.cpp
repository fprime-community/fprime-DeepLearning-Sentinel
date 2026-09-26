// D85: the ground's half of the gate, linked against the SAME cycle object the
// retrainer flies and the SAME flight core the detector flies.
//
//   ground_tools train    <flying.bin> <segment.f32> <rows> <offset> <windows> <budget> <out.bin>
//   ground_tools residual <model.bin> <trace.f32> <rows> <lo> <hi> <warm>
//
// `train` reproduces a CONTROL exactly the way RetrainLoop trains a candidate: warm
// from the flying file's weights (Adam reset), then `windows` consecutive windows of
// the stored segment starting at `offset`, `budget` optimiser steps each, each window
// SPAN = t_max + P rows (input, then the future block it is trained against). The
// result is written as a model file -- the flying file with the trained weights --
// exactly as a candidate is. Same algorithm, same arithmetic, same object: that is
// D85's route C9, and it is why the control is REPRODUCED rather than approximated.
//
// `residual` is D78's res_held: the flight core's own |x - forecast|, averaged over
// channels and over rows [lo + warm, hi) after stepping from row lo. It uses
// Sentinel::Detector, so the forecast is the one the detector would make -- the
// aggregation over the prediction ring included -- and not a restatement.
//
// Both read row-major float32 files of `rows` x CHANNELS. No telemetry is written
// anywhere by this program.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "sentinel_cycle.h"
#include "sentinel_cycle_shape.h"
#include "sentinel_shadow.h"

#include "sentinel/Detector.hpp"

using namespace Sentinel;

namespace {

const U32 CHANNELS = SENTINEL_CYCLE_CHANNELS;
const U32 PREDICTIONS = SENTINEL_CYCLE_PREDICTIONS;
const U32 WINDOW = SENTINEL_CYCLE_WINDOW;
const U32 SPAN = WINDOW + PREDICTIONS;
const U32 N_PARAMS = SENTINEL_CYCLE_N_PARAMS;

float g_window[SPAN * CHANNELS];
float g_weights[N_PARAMS];

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

bool readRows(const char* path, U32 rows, std::vector<float>& out)
{
    std::vector<unsigned char> raw;
    if (!slurp(path, raw)) { return false; }
    if (raw.size() != static_cast<size_t>(rows) * CHANNELS * sizeof(float)) { return false; }
    out.resize(static_cast<size_t>(rows) * CHANNELS);
    std::memcpy(out.data(), raw.data(), raw.size());
    return true;
}

int train(char** argv)
{
    std::vector<unsigned char> flying;
    std::vector<float> seg;
    const U32 rows = static_cast<U32>(std::strtoul(argv[3], nullptr, 10));
    const U32 offset = static_cast<U32>(std::strtoul(argv[4], nullptr, 10));
    const U32 windows = static_cast<U32>(std::strtoul(argv[5], nullptr, 10));
    const I32 budget = static_cast<I32>(std::strtol(argv[6], nullptr, 10));
    if (!slurp(argv[1], flying) || !readRows(argv[2], rows, seg)) {
        std::fprintf(stderr, "train: cannot read inputs\n");
        return 2;
    }
    if ((budget <= 0) || (static_cast<U64>(offset) + windows + SPAN - 1U > rows)) {
        std::fprintf(stderr, "train: offset %u + %u windows of %u rows exceeds %u rows\n",
                     offset, windows, SPAN, rows);
        return 2;
    }
    if (sentinel_cycle_boot() != SENTINEL_CYC_OK || sentinel_cycle_init(7) != SENTINEL_CYC_OK
        || sentinel_shadow_load(flying.data(), static_cast<uint32_t>(flying.size())) != SENTINEL_SHD_OK
        || sentinel_shadow_warm() != SENTINEL_SHD_OK) {
        std::fprintf(stderr, "train: the cycle could not warm-start from %s\n", argv[1]);
        return 1;
    }
    U32 steps = 0U;
    for (U32 k = 0U; k < windows; ++k) {
        std::memcpy(g_window, &seg[static_cast<size_t>(offset + k) * CHANNELS],
                    sizeof(float) * SPAN * CHANNELS);
        if (sentinel_cycle_load(g_window, SPAN * CHANNELS) != SENTINEL_CYC_OK
            || sentinel_cycle_run(budget, static_cast<int32_t>(WINDOW), 1) != SENTINEL_CYC_OK) {
            std::fprintf(stderr, "train: the cycle refused window %u\n", k);
            return 1;
        }
        steps += static_cast<U32>(sentinel_cycle_steps());
    }
    std::vector<unsigned char> out(flying.size() + 64U, 0U);
    if (sentinel_cycle_export(g_weights, N_PARAMS) != SENTINEL_CYC_OK
        || sentinel_shadow_write(g_weights, N_PARAMS) != SENTINEL_SHD_OK) {
        std::fprintf(stderr, "train: the weights could not be written\n");
        return 1;
    }
    const int32_t n = sentinel_shadow_export(out.data(), static_cast<uint32_t>(out.size()));
    std::FILE* f = std::fopen(argv[7], "wb");
    if ((n <= 0) || (f == nullptr)) { return 1; }
    const size_t put = std::fwrite(out.data(), 1U, static_cast<size_t>(n), f);
    std::fclose(f);
    std::printf("TRAIN offset=%u windows=%u budget=%d steps=%u bytes=%d\n",
                offset, windows, budget, steps, n);
    return (put == static_cast<size_t>(n)) ? 0 : 1;
}

int residual(char** argv)
{
    std::vector<unsigned char> model;
    std::vector<float> trace;
    const U32 rows = static_cast<U32>(std::strtoul(argv[3], nullptr, 10));
    const U32 lo = static_cast<U32>(std::strtoul(argv[4], nullptr, 10));
    const U32 hi = static_cast<U32>(std::strtoul(argv[5], nullptr, 10));
    const U32 warm = static_cast<U32>(std::strtoul(argv[6], nullptr, 10));
    if (!slurp(argv[1], model) || !readRows(argv[2], rows, trace) || (hi > rows)
        || (lo + warm >= hi)) {
        std::fprintf(stderr, "residual: bad inputs\n");
        return 2;
    }
    static Detector detector;
    const LoadStatus st = detector.load(model.data(), static_cast<U32>(model.size()));
    if (st != LoadStatus::OK) {
        std::fprintf(stderr, "residual: %s refused: %s\n", argv[1], statusName(st));
        return 1;
    }
    double total = 0.0;
    U64 count = 0U;
    for (U32 t = lo; t < hi; ++t) {
        detector.step(&trace[static_cast<size_t>(t) * CHANNELS], true);
        if (t >= lo + warm) {
            const F32* r = detector.residual();
            for (U32 c = 0U; c < CHANNELS; ++c) {
                total += static_cast<double>(r[c]);
                ++count;
            }
        }
    }
    std::printf("RESIDUAL mean=%.9e n=%llu\n", total / static_cast<double>(count),
                static_cast<unsigned long long>(count));
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if ((argc == 9) && (std::strcmp(argv[1], "train") == 0)) {
        return train(argv + 1);
    }
    if ((argc == 8) && (std::strcmp(argv[1], "residual") == 0)) {
        return residual(argv + 1);
    }
    std::fprintf(stderr,
        "usage: ground_tools train <flying.bin> <segment.f32> <rows> <offset> <windows> <budget> <out.bin>\n"
        "       ground_tools residual <model.bin> <trace.f32> <rows> <lo> <hi> <warm>\n");
    return 2;
}
