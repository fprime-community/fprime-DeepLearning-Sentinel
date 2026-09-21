// E2: the isolation gate. docs/MODELS.md 47.9 X4 and X5, 47.9.1, 47.9.2.
//
// (!) IT LINKS THE FLIGHT CORE, NOT A RESTATEMENT OF IT. `Sentinel::Detector` is the
// same object the F' component runs, the same one the golden vectors pin, and the same
// one 42.9 measured at median 11 us and worst 326 us. A harness that timed a copy would
// be measuring the copy. Same argument TestbedRun.cpp makes.
//
// Four arms, interleaved in blocks, per 47.9.1:
//   A   idle        detector alone
//   B   process     retrainer thrashing in a SEPARATE process, one domain
//   B2  saturate    the same, one domain per core -- no headroom (47.9.2)
//   C   thread      retrainer thrashing on a domain IN THIS process
//
// X4 adjudicates on the WORSE of B and B2. X5 prices C against B.
#include "sentinel/Detector.hpp"
#include "sentinel_thrash.h"

#include <caml/memory.h>

#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>

namespace {

const uint32_t CHANNELS       = 8U;
const uint32_t BLOCK          = 20000U;
const uint32_t BLOCKS_PER_ARM = 10U;     // first is warm-up and is discarded
const uint32_t N_ARMS         = 4U;

const char* ARM_NAME[N_ARMS] = { "A_idle", "B_process", "B2_saturate", "C_thread" };

// A deterministic, cheap input table. Precomputed so the generator is not inside the
// timed region -- what is being timed is Detector::step and nothing else.
std::vector<float> makeInputs(uint32_t cycles)
{
    std::vector<float> v(static_cast<size_t>(cycles) * CHANNELS);
    uint32_t s = 12345U;
    for (size_t i = 0; i < v.size(); ++i) {
        s = (s * 1103515245U) + 12345U;                       // LCG, reproducible
        v[i] = static_cast<float>((s >> 16) & 0x7FFFU) / 32768.0F;
    }
    return v;
}

// A busy-wait, NOT a sleep. docs/MODELS.md 47.13.2: the first X6 probe used
// std::this_thread::sleep_for and compared the clock against the REQUESTED duration,
// so it measured sleep_for's wake-up latency rather than the instrument. A busy-wait
// ends when this same clock says it has, so requested and actual are one quantity.
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
    if (argc < 3) {
        std::fprintf(stderr, "usage: e2_harness <model.bin> <out.csv> [probe_ms]\n");
        return 2;
    }
    const uint32_t probeMs = (argc > 3) ? static_cast<uint32_t>(std::atoi(argv[3])) : 0U;

    // -- the model, the same one 42.9 fitted on the testbed's healthy telemetry -------
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
    const Sentinel::LoadStatus st =
        detector.load(model.data(), static_cast<Sentinel::U32>(model.size()));
    if (st != Sentinel::LoadStatus::OK) {
        std::fprintf(stderr, "model refused: %d\n", static_cast<int>(st));
        return 3;
    }

    const uint32_t perArm = BLOCK * BLOCKS_PER_ARM;
    const std::vector<float> inputs = makeInputs(perArm);

    // Preallocated: no allocation inside the timed region.
    std::vector<uint32_t> ns(static_cast<size_t>(perArm) * N_ARMS, 0U);
    int32_t thrashCount[N_ARMS] = { 0, 0, 0, 0 };

    const uint32_t cores = std::thread::hardware_concurrency();
    std::fprintf(stderr, "cores %u, %u cycles/arm, blocks of %u, probe %u ms\n",
                 cores, perArm, BLOCK, probeMs);

    // X6 runs FIRST (47.9.1), through the REAL measurement path, in THIS binary and
    // THIS run -- so the instrument that produced every figure below is the instrument
    // that was validated, rather than a sibling of it.
    if (probeMs > 0U) {
        const long long inject = static_cast<long long>(probeMs) * 1000000LL;
        std::vector<long long> recovered;
        std::vector<uint32_t> cleanNs;
        uint32_t ps = 999U;
        for (uint32_t i = 0U; i < 2000U; ++i) {
            float values[Sentinel::Config::MAX_CHANNELS];
            for (uint32_t c = 0U; c < CHANNELS; ++c) {
                ps = (ps * 1103515245U) + 12345U;
                values[c] = static_cast<float>((ps >> 16) & 0x7FFFU) / 32768.0F;
            }
            const bool doInject = ((i > 0U) && ((i % 250U) == 0U));
            const auto t0 = std::chrono::steady_clock::now();
            detector.step(values, true);
            if (doInject) { stallNs(inject); }
            const auto t1 = std::chrono::steady_clock::now();
            const long long v =
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            if (doInject) { recovered.push_back(v); }
            else { cleanNs.push_back(static_cast<uint32_t>(v)); }
        }
        std::sort(cleanNs.begin(), cleanNs.end());
        const long long base = static_cast<long long>(cleanNs[cleanNs.size() / 2U]);
        long long worstPct = 0;
        for (size_t k = 0; k < recovered.size(); ++k) {
            const long long delta = recovered[k] - base;
            long long e = ((delta - inject) * 100LL) / inject;
            if (e < 0) { e = -e; }
            if (e > worstPct) { worstPct = e; }
        }
        std::fprintf(stderr, "X6: %zu busy-wait injections of %u ms, clean median %lld ns, "
                     "worst error %lld%% (band 10%%) -> %s\n",
                     recovered.size(), probeMs, base, worstPct,
                     (worstPct <= 10) ? "HELD" : "FAILED");
        std::printf("probe,%u,%lld,%lld\n", probeMs, base, worstPct);
        if (worstPct > 10) {
            std::fprintf(stderr, "X6 FAILED: X4 and X5 are withdrawn (47.9.1). Stopping.\n");
            return 5;
        }
    }

    uint32_t done[N_ARMS] = { 0U, 0U, 0U, 0U };

    for (uint32_t b = 0U; b < BLOCKS_PER_ARM; ++b) {
        for (uint32_t a = 0U; a < N_ARMS; ++a) {
            pid_t child = 0;

            if ((a == 1U) || (a == 2U)) {                 // arms B and B2
                const char* n = (a == 1U) ? "1" : "10";
                // Long enough to outlast the block; the child stops itself and writes
                // its counter, so the parent never has to guess that it was running.
                child = fork();
                if (child == 0) {
                    execl("./oxcaml/_build/e2_thrash", "e2_thrash", n, "60000",
                          (a == 1U) ? "/tmp/e2_b.count" : "/tmp/e2_b2.count", nullptr);
                    _exit(127);
                }
                // let the child boot its runtime and get up to speed
                std::this_thread::sleep_for(std::chrono::milliseconds(400));
            } else if (a == 3U) {                          // arm C
                if (sentinel_retrainer_boot() != 0) {
                    std::fprintf(stderr, "arm C: runtime unavailable\n");
                    return 4;
                }
                (void)sentinel_thrash_start_domain();
                // (!) Release the runtime so THIS domain does not obstruct the
                // stop-the-world barrier the thrashing domain needs. Without it the
                // thrasher blocks forever and the arm looks quiet for the wrong
                // reason -- which is exactly what 47.9.2's counter check exists for.
                caml_enter_blocking_section();
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }

            for (uint32_t i = 0U; i < BLOCK; ++i) {
                const uint32_t idx = done[a] + i;
                const float* row = &inputs[static_cast<size_t>(idx) * CHANNELS];
                float values[Sentinel::Config::MAX_CHANNELS];
                for (uint32_t c = 0U; c < CHANNELS; ++c) { values[c] = row[c]; }

                const auto s0 = std::chrono::steady_clock::now();
                detector.step(values, true);
                const auto s1 = std::chrono::steady_clock::now();

                ns[(static_cast<size_t>(a) * perArm) + idx] = static_cast<uint32_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count());
            }
            done[a] += BLOCK;

            if ((a == 1U) || (a == 2U)) {
                kill(child, SIGKILL);
                int status = 0;
                waitpid(child, &status, 0);
                // The child was killed mid-run, so its file holds the count from the
                // PREVIOUS block. Read whatever is there; the last block's child is
                // allowed to finish below.
                {
                    const char* path = (a == 1U) ? "/tmp/e2_b.count" : "/tmp/e2_b2.count";
                    std::FILE* cf = std::fopen(path, "r");
                    if (cf != nullptr) {
                        int v = 0;
                        if (std::fscanf(cf, "%d", &v) == 1) { thrashCount[a] = v; }
                        std::fclose(cf);
                    }
                }
            } else if (a == 3U) {
                caml_leave_blocking_section();
                thrashCount[a] = sentinel_thrash_stop();
            }
        }
    }

    // -- emit, discarding the first block of each arm as warm-up ----------------------
    std::FILE* out = std::fopen(argv[2], "w");
    std::fprintf(out, "arm,cycle,ns\n");
    for (uint32_t a = 0U; a < N_ARMS; ++a) {
        for (uint32_t i = BLOCK; i < perArm; ++i) {
            std::fprintf(out, "%s,%u,%u\n", ARM_NAME[a], i,
                         ns[(static_cast<size_t>(a) * perArm) + i]);
        }
    }
    std::fclose(out);

    for (uint32_t a = 0U; a < N_ARMS; ++a) {
        std::fprintf(stderr, "thrash counter %-12s %d\n", ARM_NAME[a], thrashCount[a]);
    }
    return 0;
}
