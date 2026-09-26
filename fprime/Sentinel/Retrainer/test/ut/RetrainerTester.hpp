// ======================================================================
// \title  RetrainerTester.hpp
// \brief  Test harness for Sentinel::Retrainer (D85), and E1's pipe kept as a fixture
//
// (!) ONE TEST, AND THE REASON IS A FINDING RATHER THAN A CONVENIENCE. OCaml module
// state is PROCESS-global and caml_startup runs once per process, so no second test
// case can get a fresh runtime; gtest runs every case in one process
// (docs/MODELS.md 47.13). One ordered lifecycle is the honest shape.
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

  private:
    void connectPorts();
    void initComponents();
    void sample(U32 seq, F32 value);

    Retrainer component;
};

}  // namespace Sentinel

#endif
