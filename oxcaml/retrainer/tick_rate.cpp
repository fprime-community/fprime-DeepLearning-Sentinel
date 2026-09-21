// Section 62 / E5-d: the detector's tick, at a rate, with and without a
// retrainer process beside it.
//
// (!) TWO CLOCKS AND THEY ARE NOT THE SAME THING.
//   compute  steady_clock around detector.step() alone. What 42.9's T7 measured
//            (median 11 us, p99 28 us, worst 326 us) -- and T7 INCLUDED its trace
//            write, so it is an upper bound on a smaller quantity.
//   period   steady_clock between tick starts. The rate the group is asked to
//            hold. Reported beside the compute time, never instead of it.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "sentinel/Detector.hpp"

namespace {

std::vector<unsigned char> slurp(const char* path)
{
    std::vector<unsigned char> out;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { return out; }
    unsigned char buf[65536];
    size_t got = 0U;
    while ((got = std::fread(buf, 1U, sizeof(buf), f)) > 0U) {
        out.insert(out.end(), buf, buf + got);
    }
    std::fclose(f);
    return out;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: tick_rate model hz ticks\n");
        return 2;
    }
    const std::vector<unsigned char> model = slurp(argv[1]);
    const double hz = std::atof(argv[2]);
    const uint32_t ticks = static_cast<uint32_t>(std::atoi(argv[3]));
    if (model.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }

    static Sentinel::Detector detector;
    if (detector.load(model.data(), static_cast<Sentinel::U32>(model.size()))
        != Sentinel::LoadStatus::OK) {
        std::fprintf(stderr, "model refused\n"); return 3;
    }

    const auto period = std::chrono::nanoseconds(
        static_cast<int64_t>(1e9 / hz));
    std::vector<int64_t> compute;
    compute.reserve(ticks);

    Sentinel::F32 values[Sentinel::Config::MAX_CHANNELS];
    const Sentinel::U32 channels = detector.model().nChannels;
    auto next = std::chrono::steady_clock::now();

    for (uint32_t t = 0U; t < ticks; ++t) {
        for (Sentinel::U32 c = 0U; c < channels; ++c) {
            const uint32_t mix = ((t * 2654435761U) + (c * 40503U)) & 0xFFFFU;
            values[c] = static_cast<Sentinel::F32>(mix) / 65535.0F;
        }
        const auto t0 = std::chrono::steady_clock::now();
        detector.step(values, true);
        const auto t1 = std::chrono::steady_clock::now();
        compute.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

        next += period;
        std::this_thread::sleep_until(next);
    }

    std::sort(compute.begin(), compute.end());
    const size_t n = compute.size();
    const double med = static_cast<double>(compute[n / 2]) / 1000.0;
    const double p99 = static_cast<double>(compute[(n * 99U) / 100U]) / 1000.0;
    const double worst = static_cast<double>(compute[n - 1U]) / 1000.0;
    std::printf("%.0f %u %.3f %.3f %.3f\n", hz, ticks, med, p99, worst);
    return 0;
}
