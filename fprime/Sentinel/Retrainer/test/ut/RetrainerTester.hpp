// ======================================================================
// \title  RetrainerTester.hpp
// \brief  Test harness for Sentinel::Retrainer (D85), and E1's pipe kept as a fixture
//
// (!) ONE TEST PER PROCESS, AND THE REASON IS A FINDING RATHER THAN A CONVENIENCE. OCaml
// module state is PROCESS-global and caml_startup runs once per process, so no second test
// case can get a fresh runtime; gtest runs every case in one process
// (docs/MODELS.md 47.13). One ordered lifecycle is the honest shape. The second case,
// LiveWindow (78.11), keeps that rule by skipping unless SENTINEL_RETRAINER_UT_LIVE is set,
// and tests/test_retrainer_live_window.py runs it alone with --gtest_filter.
// ======================================================================

#ifndef Sentinel_RetrainerTester_HPP
#define Sentinel_RetrainerTester_HPP

#include "Sentinel/Retrainer/Retrainer.hpp"
#include "Sentinel/Retrainer/RetrainerGTestBase.hpp"

namespace Sentinel {

class RetrainerTester final : public RetrainerGTestBase {
  public:
    static const FwSizeType MAX_HISTORY_SIZE = 512;
    static const FwEnumStoreType TEST_INSTANCE_ID = 0;

    RetrainerTester();
    ~RetrainerTester();

    void testTheWholeLifecycle();

    //! docs/MODELS.md 78.11 HT1-HT4: the window fills from live samples, and nothing
    //! within SPAN + 2 x guard of a crossing is admitted. Writes one row per sample to
    //! `tracePath`; tests/test_retrainer_live_window.py reads it.
    void testLiveWindow(const char* flying, const char* candidate, const char* tracePath,
                        bool spike);

  private:
    void connectPorts();
    void initComponents();
    void sample(U32 seq, F32 value);

    Retrainer component;
};

}  // namespace Sentinel

#endif
