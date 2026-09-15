// E2's X6, rebuilt. docs/MODELS.md 47.9.1, and 47.13.2 records why it needed rebuilding.
//
// (!) THE FIRST PROBE TESTED THE WRONG THING AND FAILED ITS OWN BAND. It called
// std::this_thread::sleep_for(1ms) and compared steady_clock's reading against the
// REQUESTED duration: injected 1 ms, recovered 1.264 ms, 26% over a 10% band. But
// sleep_for sleeps AT LEAST its argument plus scheduler wake-up latency, so the sleep
// really did last 1.264 ms and the clock read it correctly. The probe was measuring
// sleep_for's accuracy, not the instrument's.
//
// This one injects a BUSY-WAIT of a known duration into the real measurement path --
// inside the same timed region, around the same Detector::step, recorded into the same
// array by the same code -- and asks whether the recorded value contains it. That is
// end-to-end sensitivity, and it is strictly more demanding than what 47.9.1 asked for.
#include "sentinel/Detector.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace {
const uint32_t CHANNELS = 8U;
const uint32_t CYCLES   = 20000U;
const uint32_t EVERY    = 2000U;      // inject on every Nth cycle

// A busy-wait, not a sleep: it ends when the clock says it has, so requested and
// actual are the same quantity and the comparison is about the instrument.
void stallNs(long long want)
{
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count() >= want) {
            return;
        }
    }
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: e2_probe <model.bin>\n"); return 2; }
    std::FILE* mf = std::fopen(argv[1], "rb");
    if (mf == nullptr) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 3; }
    std::vector<unsigned char> model;
    unsigned char buf[65536];
    size_t got = 0U;
    while ((got = std::fread(buf, 1U, sizeof(buf), mf)) > 0U) {
        model.insert(model.end(), buf, buf + got);
    }
    std::fclose(mf);

    static Sentinel::Detector detector;
    if (detector.load(model.data(), static_cast<Sentinel::U32>(model.size()))
        != Sentinel::LoadStatus::OK) {
        std::fprintf(stderr, "model refused\n"); return 3;
    }

    const long long INJECT = 1000000LL;   // 1 ms, the figure 47.9.1 names
    std::vector<uint32_t> ns(CYCLES, 0U);
    std::vector<uint32_t> injected;

    uint32_t s = 12345U;
    for (uint32_t i = 0U; i < CYCLES; ++i) {
        float values[Sentinel::Config::MAX_CHANNELS];
        for (uint32_t c = 0U; c < CHANNELS; ++c) {
            s = (s * 1103515245U) + 12345U;
            values[c] = static_cast<float>((s >> 16) & 0x7FFFU) / 32768.0F;
        }
        const bool inject = ((i > 0U) && ((i % EVERY) == 0U));

        const auto t0 = std::chrono::steady_clock::now();
        detector.step(values, true);
        if (inject) { stallNs(INJECT); }
        const auto t1 = std::chrono::steady_clock::now();

        ns[i] = static_cast<uint32_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        if (inject) { injected.push_back(ns[i]); }
    }

    std::vector<uint32_t> clean;
    for (uint32_t i = 0U; i < CYCLES; ++i) {
        if ((i == 0U) || ((i % EVERY) != 0U)) { clean.push_back(ns[i]); }
    }
    std::sort(clean.begin(), clean.end());
    const uint32_t cleanMedian = clean[clean.size() / 2U];

    std::printf("X6 probe, rebuilt: %zu injections of %lld ns\n", injected.size(), INJECT);
    std::printf("  clean median          %u ns\n", cleanMedian);
    long long worstErrPct = 0;
    for (size_t k = 0; k < injected.size(); ++k) {
        const long long recovered = static_cast<long long>(injected[k]) - cleanMedian;
        const long long errPct = ((recovered - INJECT) * 100LL) / INJECT;
        std::printf("  injection %zu  recorded %u ns  minus clean median = %lld ns  "
                    "error %+lld%%\n", k + 1U, injected[k], recovered, errPct);
        const long long mag = (errPct < 0) ? -errPct : errPct;
        if (mag > worstErrPct) { worstErrPct = mag; }
    }
    std::printf("  worst absolute error  %lld%%   band 10%%   -> %s\n",
                worstErrPct, (worstErrPct <= 10) ? "HELD" : "FAILED");
    return (worstErrPct <= 10) ? 0 : 1;
}
