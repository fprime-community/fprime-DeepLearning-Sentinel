// ======================================================================
// \title  RetrainerTestMain.cpp
// \brief  Test main for Sentinel::Retrainer -- E1, the pipe (docs/MODELS.md 47)
//
// (!) ONE TEST, DELIBERATELY. The OCaml runtime is process-global and gtest runs
// every case in one process, so no case can get a fresh runtime and a
// test-per-behaviour split would have each case depending on the order of the
// ones before it. One ordered lifecycle is the honest shape. See
// RetrainerTester.hpp for the finding this came from.
// ======================================================================
#include "Fw/Test/UnitTest.hpp"
#include "RetrainerTester.hpp"

TEST(Retrainer, TheWholeLifecycle) {
    COMMENT("E1's pipe as a fixture (X1, X3), then D85: configuration checked, an "
            "absent flying model degrades, samples queue and drain, gaps are counted.");
    Sentinel::RetrainerTester tester;
    tester.testTheWholeLifecycle();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
