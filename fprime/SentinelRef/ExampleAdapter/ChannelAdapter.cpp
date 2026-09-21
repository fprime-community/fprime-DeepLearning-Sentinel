// The whole adapter: hold the latest value of each channel, emit one vector per
// tick. A mission copies this and replaces `valueIn` with its own typed
// telemetry -- the conversion is the only thing this component exists to do.
#include "SentinelRef/ExampleAdapter/ChannelAdapter.hpp"

#include "Sentinel/Monitor/FppConstantsAc.hpp"
#include "sentinel/Config.hpp"

namespace Example {

namespace {
//! The same width guard `PowerSim.cpp:11-14` carries, for the same reason: the
//! two counts are declared in different files and a mission that widens one
//! should fail the build rather than the flight.
static_assert(static_cast<U32>(Sentinel::MAX_CHANNELS) ==
                  static_cast<U32>(::Sentinel::Config::MAX_CHANNELS),
              "Sentinel.MAX_CHANNELS and Config::MAX_CHANNELS must agree");
}  // namespace

ChannelAdapter::ChannelAdapter(const char* compName)
    : ChannelAdapterComponentBase(compName), m_values() {
    for (U32 c = 0U; c < static_cast<U32>(Sentinel::MAX_CHANNELS); ++c) {
        m_values[static_cast<FwSizeType>(c)] = 0.0F;
        m_seen[c] = false;
    }
}

void ChannelAdapter::configure(U32 published) {
    FW_ASSERT(published <= static_cast<U32>(Sentinel::MAX_CHANNELS),
              static_cast<FwAssertArgType>(published));
    m_published = published;
}

void ChannelAdapter::valueIn_handler(FwIndexType portNum, U32 index, F32 value) {
    static_cast<void>(portNum);
    // One bad index is not a reason to blind the detector: refuse it, say so,
    // and let the tick emit what did arrive.
    if (index >= m_published) {
        this->log_WARNING_LO_IndexRefused(index, m_published);
        return;
    }
    m_values[static_cast<FwSizeType>(index)] = value;
    m_seen[index] = true;
}

void ChannelAdapter::schedIn_handler(FwIndexType portNum, U32 context) {
    static_cast<void>(portNum);
    static_cast<void>(context);
    this->tlmWrite_PublishedChannels(m_published);
    if (!this->isConnected_channelOut_OutputPort(0)) {
        return;
    }
    // `valid` is false when any published channel was not measured this cycle.
    // `Monitor.fpp:27-28`: that scores negative infinity, and nothing measured
    // is never an alarm.
    bool valid = (m_published > 0U);
    for (U32 c = 0U; c < m_published; ++c) {
        if (!m_seen[c]) {
            valid = false;
        }
    }
    this->channelOut_out(0, m_values, valid);
}

}  // namespace Example
