// ======================================================================
// \title  MonitorTestMain.cpp
// \brief  Test main for Sentinel::Monitor
// ======================================================================

#include "Fw/Test/UnitTest.hpp"
#include "MonitorTester.hpp"

// -- the model path ---------------------------------------------------------

TEST(Nominal, AGoodFileArmsTheModel) {
    COMMENT("A verified model.bin arms the forecaster and says so once.");
    Sentinel::MonitorTester tester;
    tester.testAGoodFileArmsTheModel();
}

TEST(Nominal, TelemetryUpdatesOnEveryTick) {
    COMMENT("All five channels move every tick, so the ground can see the detector run.");
    Sentinel::MonitorTester tester;
    tester.testTelemetryUpdatesOnEveryTick();
}

TEST(Nominal, SilentUntilWarm) {
    COMMENT("Objective.md 11 rule 2: no output until sufficient history backs a warning.");
    Sentinel::MonitorTester tester;
    tester.testSilentUntilWarm();
}

TEST(Nominal, TheWarningNamesTheChannelFromTheModelFile) {
    COMMENT("Objective.md 11 rule 4: a warning an operator can evaluate names its channel.");
    Sentinel::MonitorTester tester;
    tester.testTheWarningNamesTheChannelFromTheModelFile();
}

// -- Level 1, the safe failure mode -----------------------------------------

TEST(Level1, EveryRefusalCodeDegradesToTheBaseline) {
    COMMENT("D5: all twelve refusal codes degrade with an event, none fails the topology.");
    Sentinel::MonitorTester tester;
    tester.testEveryRefusalCodeDegradesToTheBaseline();
}

TEST(Level1, BaselineOnlyDegradesWithoutARefusal) {
    COMMENT("Objective.md 14.10: baseline_only is Level 1 by request, not by failure.");
    Sentinel::MonitorTester tester;
    tester.testBaselineOnlyDegradesWithoutARefusal();
}

TEST(Level1, AMissingFileDegrades) {
    COMMENT("An absent file is an absence, not a refusal code.");
    Sentinel::MonitorTester tester;
    tester.testAMissingFileDegrades();
}

TEST(Level1, TheTopologyKeepsTickingAfterEveryRefusal) {
    COMMENT("Never fail the topology, measured over 200 ticks after a refusal.");
    Sentinel::MonitorTester tester;
    tester.testTheTopologyKeepsTickingAfterEveryRefusal();
}

TEST(Level1, TheBaselineWarningNamesASynthesisedChannel) {
    COMMENT("With no channel map, the index is the only honest name.");
    Sentinel::MonitorTester tester;
    tester.testTheBaselineWarningNamesASynthesisedChannel();
}

// -- the degenerate inputs --------------------------------------------------

TEST(OffNominal, ATickWithNoSampleCannotAlarm) {
    COMMENT("Upstream falling silent scores negative infinity; nothing measured never alarms.");
    Sentinel::MonitorTester tester;
    tester.testATickWithNoSampleCannotAlarm();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
