// ======================================================================
// \title  SampleTapTester.cpp
// \brief  D85: the detector's input through the tap is byte-for-byte the adapter's
// ======================================================================
#include "SampleTapTester.hpp"

#include <cstring>

namespace Sentinel {

SampleTapTester::SampleTapTester()
    : SampleTapGTestBase("SampleTapTester", SampleTapTester::MAX_HISTORY_SIZE),
      component("SampleTap")
{
    this->initComponents();
    this->connectPorts();
}

SampleTapTester::~SampleTapTester() {}

void SampleTapTester::testTheDetectorGetsExactlyWhatTheAdapterSent()
{
    // Values chosen to include signed zero, a subnormal and a large magnitude, so a
    // conversion anywhere on the path would show.
    ChannelVector v;
    for (U32 c = 0U; c < 16U; ++c) {
        v[static_cast<FwSizeType>(c)] = static_cast<F32>(c) * 1.25F - 7.0F;
    }
    v[3] = -0.0F;
    v[4] = 1.0e-40F;
    v[5] = 3.0e38F;
    this->invoke_to_sampleIn(0, v, true);
    this->invoke_to_sampleIn(0, v, false);

    ASSERT_from_sampleOut_SIZE(2);
    for (U32 k = 0U; k < 2U; ++k) {
        const ChannelVector& got = this->fromPortHistory_sampleOut->at(k).values;
        for (U32 c = 0U; c < 16U; ++c) {
            const F32 a = got[static_cast<FwSizeType>(c)];
            const F32 b = v[static_cast<FwSizeType>(c)];
            ASSERT_EQ(0, std::memcmp(&a, &b, sizeof(F32))) << "channel " << c;
        }
    }
    ASSERT_TRUE(this->fromPortHistory_sampleOut->at(0).valid);
    ASSERT_FALSE(this->fromPortHistory_sampleOut->at(1).valid);
}

void SampleTapTester::testTheCopyCarriesASequence()
{
    ChannelVector v;
    for (U32 c = 0U; c < 16U; ++c) {
        v[static_cast<FwSizeType>(c)] = static_cast<F32>(c);
    }
    for (U32 k = 0U; k < 3U; ++k) {
        this->invoke_to_sampleIn(0, v, true);
    }
    ASSERT_from_sampleCopy_SIZE(3);
    for (U32 k = 0U; k < 3U; ++k) {
        ASSERT_EQ(k, this->fromPortHistory_sampleCopy->at(k).seq);
        ASSERT_TRUE(this->fromPortHistory_sampleCopy->at(k).valid);
    }
    ASSERT_EQ(3U, this->component.forwarded());
    ASSERT_TLM_Forwarded(2, 3U);
}

}  // namespace Sentinel
