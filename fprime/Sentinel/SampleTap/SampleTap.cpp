// ======================================================================
// \title  SampleTap.cpp
// \brief  D85: the telemetry path to the retrainer. SampleTap.fpp carries the design.
// ======================================================================
#include "Sentinel/SampleTap/SampleTap.hpp"

namespace Sentinel {

SampleTap::SampleTap(const char* const compName)
    : SampleTapComponentBase(compName), m_seq(0U)
{
}

SampleTap::~SampleTap() {}

void SampleTap::sampleIn_handler(FwIndexType portNum, Sentinel::ChannelVector& values,
                                 bool valid)
{
    (void)portNum;
    // (!) THE DETECTOR FIRST, UNCHANGED. Same reference, same flag, before anything
    // else -- the copy below cannot affect what the Monitor is handed.
    if (this->isConnected_sampleOut_OutputPort(0)) {
        this->sampleOut_out(0, values, valid);
    }
    // Then the copy, fixed work, whatever is (or is not) on the far side.
    if (this->isConnected_sampleCopy_OutputPort(0)) {
        this->sampleCopy_out(0, values, valid, m_seq);
    }
    ++m_seq;
    this->tlmWrite_Forwarded(m_seq);
}

}  // namespace Sentinel
