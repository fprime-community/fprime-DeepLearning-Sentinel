// ======================================================================
// \title  MonitorTester.hpp
// \brief  Test harness for Sentinel::Monitor
//
// The matrix that matters here is Level 1: every one of the loader's twelve
// refusal codes must leave the component running the statistical baseline, with
// an event that names the code, and must never fail the topology (D5,
// Objective.md 14.10). The mutations are the same ones
// flight/test/RefusalTests.cpp uses on the same file, so the two suites cannot
// drift about what a refusal is.
// ======================================================================

#ifndef Sentinel_MonitorTester_HPP
#define Sentinel_MonitorTester_HPP

#include "Sentinel/Monitor/Monitor.hpp"
#include "Sentinel/Monitor/MonitorGTestBase.hpp"

namespace Sentinel {

class MonitorTester final : public MonitorGTestBase {
  public:
    // Room for the longest run any test drives. TheTopologyKeepsTicking drives
    // 200 ticks and each writes five telemetry channels, so the history has to
    // hold them or the tester asserts before the component can be judged.
    static const FwSizeType MAX_HISTORY_SIZE = 512;
    static const FwEnumStoreType TEST_INSTANCE_ID = 0;

    //! The queue the tester gives the component. Required since work item 10
    //! made `Monitor` queued (D32 consequence 2); the autocoded tester base
    //! reads it to size the instance's message queue. Matches the topology's
    //! `queue size 10` and `Monitor::DISPATCH_DEPTH`, so a test drains what a
    //! flying tick would drain.
    static const FwSizeType TEST_INSTANCE_QUEUE_DEPTH = 10;

    //! g1: 3 channels, warm-up 8 ticks, threshold 0.8371512591838837
    static const U32 G1_BYTES = 1276U;
    static const U32 G1_CHANNELS = 3U;

    MonitorTester();
    ~MonitorTester();

    // -- the tests ---------------------------------------------------------
    void testAGoodFileArmsTheModel();
    void testEveryRefusalCodeDegradesToTheBaseline();
    void testBaselineOnlyDegradesWithoutARefusal();
    void testAMissingFileDegrades();
    void testTheTopologyKeepsTickingAfterEveryRefusal();
    void testTelemetryUpdatesOnEveryTick();
    void testSilentUntilWarm();
    void testATickWithNoSampleCannotAlarm();
    void testTheWarningNamesTheChannelFromTheModelFile();
    void testTheBaselineWarningNamesASynthesisedChannel();
    void testTheWarningNamesTheFusedRulesChannel();

    // Work item 10 / docs/MODELS.md 73, HO3. Both directions.
    void testACommandedReloadLoadsTheNamedFile();
    void testARefusedReloadRestoresThePreviousModel();
    void testTheReloadCommandIsDispatchedInsideTheTick();
    void testAReloadThatChangesTheChannelWidthIsRefused();

    // D84 / docs/MODELS.md 77, SX14: the loop closed on the host.
    //! A candidate the RETRAINER wrote, offered by RELOAD_MODEL. Driven by
    //! tests/test_retrained_candidate_reloads_in_the_monitor.py through the
    //! environment -- SENTINEL_LOOP_DIR, _CANDIDATE, _CHANNELS, _EXPECT -- and
    //! skipped without it, because the candidate comes from another process.
    void testARetrainedCandidateIsCommandedIn();

  private:
    void connectPorts();
    void initComponents();

    // -- model-file helpers, mirroring flight/test/RefusalTests.cpp ---------
    void loadPristine();
    void restore();
    void resign();
    void resignParams();
    void writeU16At(U32 offset, U32 value);
    void writeU32At(U32 offset, U32 value);
    U32 paramBlockOffset() const;

    //! Write m_working to `path`, then configure and load it. Returns the mode.
    Mode loadWorkingAs(const char* path);

    //! Drive `count` ticks with `value` on every channel.
    void tick(U32 count, F32 value, bool valid = true);

    //! One tick with DISTINCT per-channel values. `tick` drives every channel
    //! with the same number, which makes the residual and derivative argmaxes
    //! agree by construction -- and the defect this file now guards only shows
    //! up when they disagree.
    void tickChannels(const F32* values, U32 n, bool valid = true);

    Monitor component;
    U8 m_original[G1_BYTES];
    U8 m_working[G1_BYTES];
    U32 m_length;
    bool m_haveVectors;
};

}  // namespace Sentinel

#endif
