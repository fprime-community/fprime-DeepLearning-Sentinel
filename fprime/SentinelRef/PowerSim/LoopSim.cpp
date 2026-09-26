// D85 / `docs/MODELS.md` 78: the retraining loop in simulation, running the flown code.
//
// It links what flies: `PowerPlant` (the testbed's physics), `Sentinel::Detector`
// (the flight core the Monitor runs), and `Sentinel::RetrainLoop` (the retrainer's
// onboard logic, the same class the F' component runs) over the generated cycle
// object. Nothing is re-implemented here; this file only wires them tick by tick, as
// `TestbedRun.cpp` wired the plant and the core for 42.
//
//   loopsim key=value ...     (all ticks are AFTER the spin-up)
//
//   out=DIR  flying=MODEL.bin  seed=S  ticks=N  spinup=U
//   balance=HOUSEKEEPING,SHUNT_SOC               D85's orbit-balanced operating point
//   ageing=START,TAU,SOLAR_LOSS,EMIS_LOSS        healthy ageing (optional)
//   fault=START,RATE                             RESISTANCE_RISE (optional)
//   guard=G budget=B schedule=E nominal_ppm=P    the retrainer (EXPERIMENTAL B, E)
//   retrain=0|1                                  run the retrainer at all
//   swap=TICK,MODEL.bin                          a human-approved swap: at TICK the
//                                                detector reloads MODEL.bin, exactly
//                                                as RELOAD_MODEL does (the warm-up
//                                                restarts), and the retrainer stops
//
// Writes, into DIR:
//   trace.f32       the plant's channels, rows x 8, float32 -- the telemetry the
//                   ground receives
//   ticks.csv       per tick: emitted, crossing, limit_yellow, window_admits,
//                   admitted, candidate
//   candidates.csv  per candidate: index, tick, first data tick, last data tick,
//                   steps, static_crc32, file
//   cand_K.bin      each candidate model file
//   summary.txt     the scenario, and the first any-colour crossing tick
//
// (!) DETERMINISTIC. Same arguments, same bytes: the plant is counter-seeded, the core
// and the cycle are deterministic (Objective.md 11 rule 5). A swap run therefore
// reproduces its unswapped run exactly up to the swap tick, which is what lets the
// ground gate run BETWEEN the two.
//
// (!) No timing figure is produced (stop 35).
#include "PowerPlant.hpp"
#include "RetrainLoop.hpp"
#include "sentinel/Detector.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace Testbed;

namespace {

// YELLOW, transcribed from PowerSim.fpp and guarded by tests/test_powersim_limits.py
// (the same table TestbedRun.cpp carries).
const double YELLOW_LOW[PLANT_CHANNELS] = {-2.0, -10.0, 0.5, 27.0, -5.0, -35.0, 0.0, 0.30};
const double YELLOW_HIGH[PLANT_CHANNELS] = {300.0, 10.0, 9.5, 32.0, 40.0, 30.0, 1.0, 1.0};

struct Args {
    std::string out, flying, swapModel;
    unsigned seed = 1u, ticks = 0u, spinup = 0u;
    bool balance = false; double housekeeping = 2.35, shunt = 1.0;
    bool ageing = false; unsigned ageStart = 0u; double ageTau = 1.0, solarLoss = 0.0, emisLoss = 0.0;
    bool fault = false; unsigned faultStart = 0u; double faultRate = 0.0;
    unsigned guard = 260u, schedule = 6550u; int budget = 1, nominalPpm = 1830;
    bool retrain = true; bool swap = false; unsigned swapTick = 0u;
};

std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> out;
    size_t a = 0;
    while (true) {
        const size_t b = s.find(',', a);
        out.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) { break; }
        a = b + 1;
    }
    return out;
}

bool parse(int argc, char** argv, Args& a)
{
    for (int i = 1; i < argc; ++i) {
        const std::string kv(argv[i]);
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) { return false; }
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        const std::vector<std::string> p = split(v);
        if (k == "out") { a.out = v; }
        else if (k == "flying") { a.flying = v; }
        else if (k == "seed") { a.seed = (unsigned)std::strtoul(v.c_str(), nullptr, 10); }
        else if (k == "ticks") { a.ticks = (unsigned)std::strtoul(v.c_str(), nullptr, 10); }
        else if (k == "spinup") { a.spinup = (unsigned)std::strtoul(v.c_str(), nullptr, 10); }
        else if (k == "balance" && p.size() == 2) {
            a.balance = true; a.housekeeping = std::atof(p[0].c_str()); a.shunt = std::atof(p[1].c_str()); }
        else if (k == "ageing" && p.size() == 4) {
            a.ageing = true; a.ageStart = (unsigned)std::atoi(p[0].c_str()); a.ageTau = std::atof(p[1].c_str());
            a.solarLoss = std::atof(p[2].c_str()); a.emisLoss = std::atof(p[3].c_str()); }
        else if (k == "fault" && p.size() == 2) {
            a.fault = true; a.faultStart = (unsigned)std::atoi(p[0].c_str()); a.faultRate = std::atof(p[1].c_str()); }
        else if (k == "guard") { a.guard = (unsigned)std::atoi(v.c_str()); }
        else if (k == "budget") { a.budget = std::atoi(v.c_str()); }
        else if (k == "schedule") { a.schedule = (unsigned)std::atoi(v.c_str()); }
        else if (k == "nominal_ppm") { a.nominalPpm = std::atoi(v.c_str()); }
        else if (k == "retrain") { a.retrain = (v == "1"); }
        else if (k == "swap" && p.size() == 2) {
            a.swap = true; a.swapTick = (unsigned)std::atoi(p[0].c_str()); a.swapModel = p[1]; }
        else { return false; }
    }
    // With retrain=0 and no flying model this is a plain plant dump -- which is how the
    // flying model itself is fitted, on a healthy run.
    return !a.out.empty() && a.ticks > 0u && (!a.flying.empty() || !a.retrain);
}

bool slurp(const std::string& path, std::vector<unsigned char>& out)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { return false; }
    unsigned char buf[65536];
    size_t got = 0U;
    while ((got = std::fread(buf, 1U, sizeof(buf), f)) > 0U) { out.insert(out.end(), buf, buf + got); }
    std::fclose(f);
    return true;
}

bool anyYellow(const PowerPlant& p)
{
    for (uint32_t c = 0u; c < PLANT_CHANNELS; ++c) {
        const double v = p.value(c);
        if ((v < YELLOW_LOW[c]) || (v > YELLOW_HIGH[c]) || (v < PLANT_RED[c].low) || (v > PLANT_RED[c].high)) {
            return true;
        }
    }
    return false;
}

// Static: the loop and the detectors are large, and fixed-size by design.
Sentinel::RetrainLoop g_loop;
Sentinel::Detector g_detector;

}  // namespace

int main(int argc, char** argv)
{
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr, "usage: loopsim out=DIR flying=MODEL.bin ticks=N [key=value ...] "
                             "(see LoopSim.cpp's header)\n");
        return 2;
    }
    std::vector<unsigned char> flying, swapModel;
    const bool detecting = !a.flying.empty();
    if (detecting && !slurp(a.flying, flying)) { std::fprintf(stderr, "cannot read %s\n", a.flying.c_str()); return 2; }
    if (a.swap && !slurp(a.swapModel, swapModel)) {
        std::fprintf(stderr, "cannot read %s\n", a.swapModel.c_str()); return 2;
    }
    if (detecting && g_detector.load(flying.data(), (Sentinel::U32)flying.size()) != Sentinel::LoadStatus::OK) {
        std::fprintf(stderr, "the detector refused %s\n", a.flying.c_str()); return 1;
    }
    if (detecting && g_detector.model().nChannels != PLANT_CHANNELS) {
        std::fprintf(stderr, "the model is not %u channels\n", (unsigned)PLANT_CHANNELS); return 1;
    }

    if (a.retrain) {
        Sentinel::LoopConfig cfg = {};
        for (uint32_t c = 0u; c < PLANT_CHANNELS; ++c) {
            cfg.yellowLow[c] = (Sentinel::F32)YELLOW_LOW[c];
            cfg.yellowHigh[c] = (Sentinel::F32)YELLOW_HIGH[c];
        }
        cfg.guard = a.guard; cfg.budget = a.budget; cfg.schedule = a.schedule; cfg.nominalPpm = a.nominalPpm;
        if (!g_loop.configure(cfg)) { std::fprintf(stderr, "the retrainer refused its config\n"); return 1; }
        if (!g_loop.setFlying(flying.data(), (Sentinel::U32)flying.size())) {
            std::fprintf(stderr, "the retrainer refused the flying model\n"); return 1;
        }
        const Sentinel::I32 rc = g_loop.boot();
        if (rc != 0) { std::fprintf(stderr, "the retrainer did not boot: %d\n", (int)rc); return 1; }
    }

    PowerPlant plant;
    plant.reset(a.seed);
    if (a.balance) { plant.configureBalance(a.housekeeping, a.shunt); }
    if (a.ageing) { plant.configureAgeing(a.spinup + a.ageStart, a.ageTau, a.solarLoss, a.emisLoss); }
    for (unsigned t = 0u; t < a.spinup; ++t) { plant.step(false, 0.0); }

    std::FILE* trace = std::fopen((a.out + "/trace.f32").c_str(), "wb");
    std::FILE* ticks = std::fopen((a.out + "/ticks.csv").c_str(), "w");
    std::FILE* cands = std::fopen((a.out + "/candidates.csv").c_str(), "w");
    if (!trace || !ticks || !cands) { std::fprintf(stderr, "cannot write into %s\n", a.out.c_str()); return 2; }
    std::fprintf(ticks, "tick,emitted,crossing,limit_yellow,window_admits,admitted,candidate\n");
    std::fprintf(cands, "index,tick,first_data_tick,last_data_tick,steps,static_crc32,file\n");

    long firstYellow = -1;
    bool retraining = a.retrain;
    float row[PLANT_CHANNELS];
    for (unsigned t = 0u; t < a.ticks; ++t) {
        if (a.swap && t == a.swapTick) {
            // RELOAD_MODEL: the detector loads the approved file and its warm-up restarts.
            if (g_detector.load(swapModel.data(), (Sentinel::U32)swapModel.size()) != Sentinel::LoadStatus::OK) {
                std::fprintf(stderr, "the approved model was refused at tick %u\n", t); return 1;
            }
            retraining = false;
        }
        plant.step(a.fault && t >= a.faultStart, a.faultRate);
        for (uint32_t c = 0u; c < PLANT_CHANNELS; ++c) { row[c] = (float)plant.value(c); }
        std::fwrite(row, sizeof(float), PLANT_CHANNELS, trace);

        Sentinel::F32 values[PLANT_CHANNELS];
        for (uint32_t c = 0u; c < PLANT_CHANNELS; ++c) { values[c] = row[c]; }
        if (detecting) { g_detector.step(values, true); }
        const bool yellow = anyYellow(plant);
        if (yellow && firstYellow < 0) { firstYellow = (long)t; }

        Sentinel::LoopTick lt = {false, false, false, false, false, false, 0};
        if (retraining) {
            lt = g_loop.tick(values, true);
            if (lt.status != 0) { std::fprintf(stderr, "the retrainer failed at tick %u: %d\n", t, (int)lt.status); return 1; }
            if (lt.candidateReady) {
                const unsigned k = g_loop.candidates();
                char name[64];
                std::snprintf(name, sizeof(name), "cand_%u.bin", k);
                std::FILE* f = std::fopen((a.out + "/" + name).c_str(), "wb");
                if (!f) { return 2; }
                std::fwrite(g_loop.candidate(), 1, g_loop.candidateBytes(), f);
                std::fclose(f);
                // The candidate's data: the spans end GUARD ticks before the tick that
                // trained on them (RetrainLoop.cpp), so first..last is exact.
                const unsigned long long first = g_loop.candidateFirstTick() - (Sentinel::U64)Sentinel::RetrainLoop::SPAN - a.guard;
                const unsigned long long last = g_loop.candidateLastTick() - a.guard - 1u;
                std::fprintf(cands, "%u,%u,%llu,%llu,%u,%u,%s\n", k, t, first, last,
                             g_loop.candidateSteps(), g_loop.candidateCrc32(), name);
            }
        }
        std::fprintf(ticks, "%u,%d,%d,%d,%d,%d,%d\n", t, g_detector.emitted() ? 1 : 0,
                     g_detector.crossing() ? 1 : 0, yellow ? 1 : 0, lt.windowAdmits ? 1 : 0,
                     lt.admitted ? 1 : 0, lt.candidateReady ? 1 : 0);
    }
    std::fclose(trace); std::fclose(ticks); std::fclose(cands);

    std::FILE* s = std::fopen((a.out + "/summary.txt").c_str(), "w");
    if (!s) { return 2; }
    std::fprintf(s, "seed %u ticks %u spinup %u balance %d ageing %d fault %d,%u,%g swap %d,%u\n",
                 a.seed, a.ticks, a.spinup, a.balance ? 1 : 0, a.ageing ? 1 : 0, a.fault ? 1 : 0,
                 a.faultStart, a.faultRate, a.swap ? 1 : 0, a.swapTick);
    std::fprintf(s, "first_yellow %ld\ncandidates %u\nadmitted %llu\n", firstYellow,
                 a.retrain ? g_loop.candidates() : 0u,
                 (unsigned long long)(a.retrain ? g_loop.admittedTotal() : 0u));
    std::fclose(s);
    std::printf("loopsim: %u ticks, first yellow %ld, candidates %u\n", a.ticks, firstYellow,
                a.retrain ? g_loop.candidates() : 0u);
    return 0;
}
