// ======================================================================
// \title  HubCounter.cpp
// \brief  counts what a hub emits, and forwards everything unchanged
// ======================================================================
#include "SentinelRef/HubCounter/HubCounter.hpp"

namespace HubTap {

HubCounter::HubCounter(const char* const compName)
    : HubCounterComponentBase(compName), m_events(0U), m_channels(0U), m_ticks(0U),
      m_complete(0U)
{
    for (U32 i = 0U; i < CHANNELS; ++i) {
        m_perChannel[i] = 0U;
    }
}

HubCounter::~HubCounter() {}

void HubCounter::eventIn_handler(FwIndexType portNum, FwEventIdType id, Fw::Time& timeTag,
                                 const Fw::LogSeverity& severity, Fw::LogBuffer& args)
{
    m_events++;
    // Forwarded unchanged. A tap that alters what it taps is not measuring the
    // same thing the ground sees.
    if (this->isConnected_eventFwd_OutputPort(0)) {
        this->eventFwd_out(portNum, id, timeTag, severity, args);
    }
}

void HubCounter::tlmIn_handler(FwIndexType portNum, FwChanIdType id, Fw::Time& timeTag,
                               Fw::TlmBuffer& val)
{
    m_channels++;

    const U32 raw = static_cast<U32>(id);
    if ((raw >= RETRAIN_BASE) && ((raw - RETRAIN_BASE) < CHANNELS)) {
        const U32 offset = raw - RETRAIN_BASE;
        m_perChannel[offset]++;

        // (!) THE DENOMINATOR IS TAKEN FROM THE STREAM ITSELF, NOT FROM A CLOCK.
        // Offset 0 is the retrainer's SampleCount, which counts samples fed and
        // therefore ticks run at the SOURCE. Reading it here means the tick count
        // HB2b divides by is the source's own, not this process's guess at it --
        // and 70.5's first reading went wrong on exactly that kind of guess.
        if (offset == 0U) {
            U32 samples = 0U;
            Fw::TlmBuffer copy = val;
            copy.resetDeser();
            if (copy.deserializeTo(samples) == Fw::FW_SERIALIZE_OK) {
                const U32 ticks = samples / SAMPLES_PER_TICK;
                if (ticks > m_ticks) {
                    m_ticks = ticks;
                }
            }
        }
    } else {
        this->log_WARNING_LO_UnknownChannel(raw);
    }

    if (this->isConnected_tlmFwd_OutputPort(0)) {
        this->tlmFwd_out(portNum, id, timeTag, val);
    }
}

void HubCounter::schedIn_handler(FwIndexType portNum, U32 context)
{
    // A tick is COMPLETE when every one of the five channels has been seen at
    // least that many times. The minimum across the five is therefore the number
    // of ticks that delivered all of them, which is exactly HB2b's numerator.
    U32 least = m_perChannel[0];
    for (U32 i = 1U; i < CHANNELS; ++i) {
        if (m_perChannel[i] < least) {
            least = m_perChannel[i];
        }
    }
    m_complete = least;
    this->log_ACTIVITY_HI_Tallies(m_ticks, m_events, m_channels, m_complete);
}

}  // namespace HubTap
