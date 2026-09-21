// ======================================================================
// \title  SentinelRetrainTopology.cpp
// \brief the retrainer's own deployment: Retrainer, a rate group, nothing else
//
// (!) 47.15b names this shape as C5's discharger: "a minimal retrainer
// deployment -- its own Top/, a topology carrying the Retrainer and a rate group
// and nothing else, per D70 consequence 2".
// ======================================================================
#include <SentinelRetrain/Top/SentinelRetrainTopologyAc.hpp>

namespace SentinelRetrain {

// 1 Hz base clock, one divisor. RateGroupDriver skips a zero divisor and guards
// each output on isConnected, so one entry is enough for one rate group.
const Fw::TimeInterval rateGroupInterval(1, 0);
Svc::RateGroupDriver::DividerSet rateGroupDivisorsSet{{{1, 0}}};
Svc::ActiveRateGroup::ContextArray rateGroup_1HzContext(0);

void configureTopology() {
    rateGroupDriver.configure(rateGroupDivisorsSet);
    rateGroup_1Hz.configure(rateGroup_1HzContext);

    // (!) THE RUNTIME IS DELIBERATELY *NOT* STARTED HERE. This function runs on
    // the topology's main thread; rateGroup_1Hz is ACTIVE and calls schedIn on
    // its own task, and OCaml 5 grants the domain lock to whichever thread calls
    // caml_startup. Booting here aborts the process on the first tick with
    // "Fatal error: no domain lock held". Retrainer::schedIn_handler boots on its
    // own thread instead, which is where the domain then stays.
}

void setupTopology(const TopologyState& state) {
    initComponents(state);
    setBaseIds();
    connectComponents();
    configComponents(state);
    configureTopology();
    loadParameters();
    startTasks(state);
}

void startRateGroups() {
    timer.startTimer(rateGroupInterval);
}

void stopRateGroups() {
    timer.quit();
}

void teardownTopology(const TopologyState& state) {
    stopTasks(state);
    freeThreads(state);
    tearDownComponents(state);
    deinitComponents(state);
}

}  // namespace SentinelRetrain
