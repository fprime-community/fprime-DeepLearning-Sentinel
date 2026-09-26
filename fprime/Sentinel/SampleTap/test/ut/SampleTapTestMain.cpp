// ======================================================================
// \title  SampleTapTestMain.cpp
// \brief  D85: the SampleTap
// ======================================================================
#include "SampleTapTester.hpp"

TEST(SampleTap, TheDetectorGetsExactlyWhatTheAdapterSent) {
    Sentinel::SampleTapTester tester;
    tester.testTheDetectorGetsExactlyWhatTheAdapterSent();
}

TEST(SampleTap, TheCopyCarriesASequence) {
    Sentinel::SampleTapTester tester;
    tester.testTheCopyCarriesASequence();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
