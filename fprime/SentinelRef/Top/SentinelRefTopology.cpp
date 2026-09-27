// ======================================================================
// \title  SentinelRefTopology.cpp
// \brief cpp file containing the topology instantiation code
//
// ======================================================================
// Provides access to autocoded functions
#include <SentinelRef/Top/SentinelRefTopologyAc.hpp>
// Note: Uncomment when using Svc:TlmPacketizer
//#include <SentinelRef/Top/SentinelRefPacketsAc.hpp>

// Necessary project-specified types
#include <Fw/Types/MallocAllocator.hpp>
#include <cstring>

// Public functions for use in main program are namespaced with deployment module SentinelRef
// This is also the namespace where the topology components are instantiated by FPP.
#include "SentinelRef/PowerSim/FppConstantsAc.hpp"

namespace SentinelRef {

// Instantiate a malloc allocator for cmdSeq buffer allocation
Fw::MallocAllocator mallocator;

//! The hub pool's backing allocator; declared in SentinelRefTopologyDefs.hpp.
Fw::MallocAllocator hubAllocator;

// Rate group timing: base clock interval and divisors are coupled to rate group names
Fw::TimeInterval rateGroupInterval(1, 0);  // 1Hz base clock; -t shortens it for host runs (D85)
Svc::RateGroupDriver::DividerSet rateGroupDivisorsSet{{{1, 0}, {2, 0}, {4, 0}}};
// Divisors: 1Hz, 0.5Hz, 0.25Hz

// Context tokens for rate group members (unused, set to zero)
Svc::ActiveRateGroup::ContextArray rateGroup_1HzContext(0);
Svc::ActiveRateGroup::ContextArray rateGroup_0_5HzContext(0);
Svc::ActiveRateGroup::ContextArray rateGroup_0_25HzContext(0);

enum TopologyConstants {
    COMM_PRIORITY = 34,
};

/**
 * \brief configure/setup components in project-specific way
 *
 * This is a *helper* function which configures/sets up each component requiring project specific input. This includes
 * allocating resources, passing-in arguments, etc. This function may be inlined into the topology setup function if
 * desired, but is extracted here for clarity.
 */
void configureTopology() {
    // Rate group driver needs a divisor list
    rateGroupDriver.configure(rateGroupDivisorsSet);

    // Rate groups require context arrays.
    rateGroup_1Hz.configure(rateGroup_1HzContext);
    rateGroup_0_5Hz.configure(rateGroup_0_5HzContext);
    rateGroup_0_25Hz.configure(rateGroup_0_25HzContext);

    // Command sequencer needs to allocate memory to hold contents of command sequences
    cmdSeq.allocateBuffer(0, mallocator, 5 * 1024);

    // PrmDb file name must be supplied by the using topology
    FileHandling::prmDb.configure("PrmDb.dat");

    // Sentinel needs its model file and the width of the channel set this
    // instance watches. The width is supplied even though the file declares its
    // own, because Level 1 has to run when no file can be read at all (D5).
    //
    // (!) EIGHT CHANNELS, BECAUSE THE TESTBED PUBLISHES EIGHT (docs/MODELS.md 42).
    // It was twelve -- the flown configuration of docs/MODELS.md 3 -- while this
    // deployment carried no source of channel values at all. PowerSim now
    // supplies them, and a width the model file does not match is refused by the
    // loader rather than silently truncated, which is what BAD_SHAPE is for.
    //
    // This deployment still ships no model.bin in the tree: runs/ is gitignored
    // and a trained model is an artifact cited by path, not a committed file. So
    // loadModel() reports NOT_LOADED unless one is placed beside the binary, the
    // component emits DegradedToBaseline(NO_MODEL_FILE) and goes on serving the
    // topology on the Level 1 baseline. That is the safe failure mode working,
    // not a deployment defect: a corrupt file, a missing file and a bad CRC all
    // land in the same place, which is the point of D5.
    sentinelMonitor.configure("SentinelModel.bin", Testbed::POWERSIM_CHANNELS);
    (void)sentinelMonitor.loadModel();
}

void setupTopology(const TopologyState& state) {
    // D85: -t shortens the base tick for host runs of the retraining loop (docs/MODELS.md 78,
    // E1). Apparatus only: absent or 0, the clock is the 1 Hz it always was.
    if (state.tickMicros != 0U) {
        rateGroupInterval.set(state.tickMicros / 1000000U, state.tickMicros % 1000000U);
    }
    if (state.loopPlant) {
        powerSim.configureLoopPlant(0.60, 0.92, 60000U, state.ageEmis > 0.0, state.ageEmis);
    }
    // Autocoded initialization. Function provided by autocoder.
    initComponents(state);
    // Autocoded id setup. Function provided by autocoder.
    setBaseIds();
    // Autocoded connection wiring. Function provided by autocoder.
    connectComponents();
    // Autocoded command registration. Function provided by autocoder.
    regCommands();
    // Autocoded configuration. Function provided by autocoder.
    configComponents(state);
    if (state.hostname != nullptr && state.port != 0) {
        comDriver.configure(state.hostname, state.port);
    }
    // Project-specific component configuration. Function provided above. May be inlined, if desired.
    configureTopology();
    // Autocoded parameter read from file. Function provided by autocoder.
    readParameters();
    // Autocoded parameter loading. Function provided by autocoder.
    loadParameters();
    // Autocoded task kick-off (active components). Function provided by autocoder.
    startTasks(state);
    // Initialize socket communication if and only if there is a valid specification
    if (state.hostname != nullptr && state.port != 0) {
        Os::TaskString name("ReceiveTask");
        // Uplink is configured for receive so a socket task is started
        comDriver.start(name, COMM_PRIORITY, Default::STACK_SIZE);
    }
}

void startRateGroups() {
    // Blocks until stopRateGroups() is called (e.g. from signal handler)
    timer.startTimer(rateGroupInterval);
}

void stopRateGroups() {
    timer.quit();
}

void teardownTopology(const TopologyState& state) {
    // Autocoded (active component) task clean-up. Functions provided by topology autocoder.
    stopTasks(state);
    freeThreads(state);

    // Other task clean-up.
    comDriver.stop();
    (void)comDriver.join();

    // Resource deallocation
    cmdSeq.deallocateBuffer(mallocator);

    tearDownComponents(state);
    deinitComponents(state);
}
};  // namespace SentinelRef
