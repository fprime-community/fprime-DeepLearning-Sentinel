// ======================================================================
// \title  RetrainerTestMain.cpp
// \brief  Test main for Sentinel::Retrainer -- E1, the pipe (docs/MODELS.md 47)
//
// (!) ONE TEST PER PROCESS, DELIBERATELY. The OCaml runtime is process-global and gtest
// runs every case in one process, so no case can get a fresh runtime and a
// test-per-behaviour split would have each case depending on the order of the
// ones before it. One ordered lifecycle is the honest shape. See
// RetrainerTester.hpp for the finding this came from. LiveWindow (78.11) skips unless
// its runner sets it up, and that runner starts one process per run.
// ======================================================================
#include <cstdlib>

#include "Fw/Test/UnitTest.hpp"
#include "RetrainerTester.hpp"

TEST(Retrainer, TheWholeLifecycle) {
    COMMENT("E1's pipe as a fixture (X1, X3), then D85: configuration checked, an "
            "absent flying model degrades, samples queue and drain, gaps are counted.");
    Sentinel::RetrainerTester tester;
    tester.testTheWholeLifecycle();
}

TEST(Retrainer, LiveWindow) {
    COMMENT("docs/MODELS.md 78.11 HT1-HT4, through the component's own ports: the window "
            "fills from live samples, and no tick within SPAN + 2 x guard of a crossing is "
            "admitted. Run alone, by tests/test_retrainer_live_window.py.");
    const char* const live = std::getenv("SENTINEL_RETRAINER_UT_LIVE");
    const char* const flying = std::getenv("SENTINEL_RETRAINER_UT_FLYING");
    const char* const trace = std::getenv("SENTINEL_RETRAINER_UT_TRACE");
    const char* const candidate = std::getenv("SENTINEL_RETRAINER_UT_CANDIDATE");
    const char* const spike = std::getenv("SENTINEL_RETRAINER_UT_SPIKE");
    if ((live == nullptr) || (flying == nullptr) || (trace == nullptr) || (candidate == nullptr)) {
        GTEST_SKIP() << "run alone by tests/test_retrainer_live_window.py (one test per process)";
    }
    Sentinel::RetrainerTester tester;
    tester.testLiveWindow(flying, candidate, trace, (spike != nullptr) && (spike[0] == '1'));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
