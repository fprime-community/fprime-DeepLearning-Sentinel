// ======================================================================
// \title  SampleTapTester.hpp
// \brief  D85: the tap forwards the detector's input unchanged, then copies it
// ======================================================================
#ifndef Sentinel_SampleTapTester_HPP
#define Sentinel_SampleTapTester_HPP

#include "Sentinel/SampleTap/SampleTap.hpp"
#include "Sentinel/SampleTap/SampleTapGTestBase.hpp"

namespace Sentinel {

class SampleTapTester final : public SampleTapGTestBase {
  public:
    static const FwSizeType MAX_HISTORY_SIZE = 64;
    static const FwEnumStoreType TEST_INSTANCE_ID = 0;

    SampleTapTester();
    ~SampleTapTester();

    void testTheDetectorGetsExactlyWhatTheAdapterSent();
    void testTheCopyCarriesASequence();

  private:
    void connectPorts();
    void initComponents();

    SampleTap component;
};

}  // namespace Sentinel

#endif
