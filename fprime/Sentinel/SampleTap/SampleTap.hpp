// ======================================================================
// \title  SampleTap.hpp
// \brief  D85: forwards the detector's input unchanged, then copies it to the hub
// ======================================================================
#ifndef Sentinel_SampleTap_HPP
#define Sentinel_SampleTap_HPP

#include "Sentinel/SampleTap/SampleTapComponentAc.hpp"

namespace Sentinel {

class SampleTap final : public SampleTapComponentBase {
  public:
    explicit SampleTap(const char* const compName);
    ~SampleTap();

    //! Samples forwarded so far, for the tests.
    U32 forwarded() const { return m_seq; }

  private:
    void sampleIn_handler(FwIndexType portNum, Sentinel::ChannelVector& values,
                          bool valid) override;

    U32 m_seq;
};

}  // namespace Sentinel

#endif
