// The testbed's run harness (`docs/MODELS.md` 42).
#include <string>
//
// (!) IT LINKS THE FLIGHT CORE, NOT A RESTATEMENT OF IT. `Sentinel::Detector` is
// the same object the F' component runs and the same one the golden vectors pin.
// A harness that scored a Python copy would be measuring the copy.
//
// Two modes, and they exist for two different reasons:
//
//   dump   the plant's telemetry as a flat F64 matrix, for the toolkit to fit a
//          model on. Healthy only -- a model fitted on a degradation would have
//          learned the fault (docs/HARNESS.md 4).
//   score  plant and detector in lockstep, one tick each, recording the two
//          instants a warning time is the gap between: the first tick the
//          detector emits, and the first tick any channel leaves ANY limit band.
//
// (!) THE LIMIT INSTANT IS THE FIRST CROSSING OF ANY COLOUR, NOT THE RED TRIP.
// 42.8.1: on the seeded run CellTemp crosses yellow 915 ticks before red, so a
// ground system with yellow alarms sees the fault first. Scoring against red
// would credit a win over a limit check that had not been won.
//
// Wall clock is `steady_clock`, read once per tick. It is the testbed's own
// clock and it is real; it is not a spacecraft's, because the physics' time
// constants are chosen (42.3 departure 2). Every figure is reported in ticks and
// in testbed seconds, and the transferable one is the ratio.
#include "PowerPlant.hpp"
#include "sentinel/Detector.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace Testbed;

namespace {

// YELLOW limits, transcribed from PowerSim.fpp. RED lives in PowerPlant.hpp and
// is guarded by tests/test_powersim_limits.py; this table is guarded by the same
// test's yellow-band check plus its own row there.
const double YELLOW_LOW[PLANT_CHANNELS] =
    {-2.0, -10.0, 0.5, 27.0, -5.0, -35.0, 0.0, 0.30};
const double YELLOW_HIGH[PLANT_CHANNELS] =
    {300.0, 10.0, 9.5, 32.0, 40.0, 30.0, 1.0, 1.0};

uint32_t firstAnyCrossing(const PowerPlant& p) {
    for (uint32_t c = 0u; c < PLANT_CHANNELS; ++c) {
        const double v = p.value(c);
        if ((v < PLANT_RED[c].low) || (v > PLANT_RED[c].high) ||
            (v < YELLOW_LOW[c]) || (v > YELLOW_HIGH[c])) {
            return c;
        }
    }
    return PLANT_CHANNELS;
}

int doDump(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "dump seed faultStart rate N out\n"); return 2; }
    const uint32_t seed = (uint32_t)std::atoi(argv[2]);
    const uint32_t fs = (uint32_t)std::atoi(argv[3]);
    const double rate = std::atof(argv[4]);
    const uint32_t N = (uint32_t)std::atoi(argv[5]);
    PowerPlant p; p.reset(seed);
    std::FILE* f = std::fopen(argv[6], "wb");
    if (f == nullptr) { std::fprintf(stderr, "cannot open %s\n", argv[6]); return 2; }
    for (uint32_t t = 0u; t < N; ++t) {
        p.step((fs != 0u) && (t >= fs), rate);
        double row[PLANT_CHANNELS];
        for (uint32_t c = 0u; c < PLANT_CHANNELS; ++c) { row[c] = p.value(c); }
        std::fwrite(row, sizeof(double), PLANT_CHANNELS, f);
    }
    std::fclose(f);
    std::printf("  dumped %u x %u F64 to %s\n", N, (unsigned)PLANT_CHANNELS, argv[6]);
    return 0;
}

int doScore(int argc, char** argv) {
    if (argc < 8) { std::fprintf(stderr, "score model seed faultStart rate N trace\n"); return 2; }
    std::FILE* mf = std::fopen(argv[2], "rb");
    if (mf == nullptr) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    std::vector<unsigned char> model;
    unsigned char buf[65536];
    size_t got = 0u;
    while ((got = std::fread(buf, 1u, sizeof(buf), mf)) > 0u) { model.insert(model.end(), buf, buf + got); }
    std::fclose(mf);

    static Sentinel::Detector detector;
    const Sentinel::LoadStatus st = detector.load(model.data(), (Sentinel::U32)model.size());
    if (st != Sentinel::LoadStatus::OK) {
        std::fprintf(stderr, "  model refused: status %d\n", (int)st); return 3;
    }
    const uint32_t seed = (uint32_t)std::atoi(argv[3]);
    const uint32_t fs = (uint32_t)std::atoi(argv[4]);
    const double rate = std::atof(argv[5]);
    const uint32_t N = (uint32_t)std::atoi(argv[6]);

    PowerPlant p; p.reset(seed);
    std::FILE* tr = std::fopen(argv[7], "w");
    std::fprintf(tr, "tick,wall_us,emitted,fused,peak,limit_any,limit_red\n");

    uint32_t firstWarn = 0xFFFFFFFFu, firstAny = 0xFFFFFFFFu, firstRed = 0xFFFFFFFFu;
    long long warnWall = -1, anyWall = -1;
    uint32_t warnTicks = 0u, warmedTicks = 0u;
    const auto t0 = std::chrono::steady_clock::now();

    for (uint32_t t = 0u; t < N; ++t) {
        p.step((fs != 0u) && (t >= fs), rate);
        float values[Sentinel::Config::MAX_CHANNELS];
        for (uint32_t c = 0u; c < PLANT_CHANNELS; ++c) { values[c] = (float)p.value(c); }
        detector.step(values, true);
        const auto now = std::chrono::steady_clock::now();
        const long long wall =
            std::chrono::duration_cast<std::chrono::microseconds>(now - t0).count();

        const bool emitted = detector.emitted();
        const uint32_t anyCh = firstAnyCrossing(p);
        const uint32_t redCh = p.firstRedCrossing();
        if (emitted) {
            ++warnTicks;
            if (firstWarn == 0xFFFFFFFFu) { firstWarn = t; warnWall = wall; }
        }
        if ((anyCh < PLANT_CHANNELS) && (firstAny == 0xFFFFFFFFu)) { firstAny = t; anyWall = wall; }
        if ((redCh < PLANT_CHANNELS) && (firstRed == 0xFFFFFFFFu)) { firstRed = t; }
        // warm-up is the detector's own; ticks before it cannot alarm by construction
        if (t >= 2350u) { ++warmedTicks; }
        std::fprintf(tr, "%u,%lld,%d,%.9f,%u,%d,%d\n", t, wall, emitted ? 1 : 0,
                     detector.fusedScore(), detector.fusedChannel(),
                     (anyCh < PLANT_CHANNELS) ? (int)anyCh : -1,
                     (redCh < PLANT_CHANNELS) ? (int)redCh : -1);
    }
    std::fclose(tr);

    std::printf("  seed %u  faultStart %u  rate %.1e  N %u\n", seed, fs, rate, N);
    std::printf("    first warning        %s   wall %lld us\n",
                firstWarn == 0xFFFFFFFFu ? "never" : std::to_string(firstWarn).c_str(), warnWall);
    std::printf("    first limit ANY      %s   wall %lld us\n",
                firstAny == 0xFFFFFFFFu ? "never" : std::to_string(firstAny).c_str(), anyWall);
    std::printf("    first limit RED      %s\n",
                firstRed == 0xFFFFFFFFu ? "never" : std::to_string(firstRed).c_str());
    if ((firstWarn != 0xFFFFFFFFu) && (firstAny != 0xFFFFFFFFu)) {
        std::printf("    LEAD vs any-colour   %+d ticks   %+lld us\n",
                    (int)firstAny - (int)firstWarn, anyWall - warnWall);
    }
    std::printf("    warning ticks        %u of %u warmed  (%.4f%%)\n", warnTicks, warmedTicks,
                warmedTicks ? (100.0 * (double)warnTicks / (double)warmedTicks) : 0.0);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: TestbedRun dump|score ...\n"); return 2; }
    if (std::strcmp(argv[1], "dump") == 0) { return doDump(argc, argv); }
    if (std::strcmp(argv[1], "score") == 0) { return doScore(argc, argv); }
    std::fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
