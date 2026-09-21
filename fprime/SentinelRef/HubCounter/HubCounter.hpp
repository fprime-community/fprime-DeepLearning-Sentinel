// ======================================================================
// \title  HubCounter.hpp
// \brief  counts what a hub emits, where the hub emits it
//
// docs/MODELS.md 70.6 owed this. 70.5 could not close HB2b because the ground
// is downstream of Svc::TlmChan's packetisation and the GDS decoder, so an
// arrival count there measures the downlink as well as the crossing.
// ======================================================================
#ifndef HubTap_HubCounter_HPP
#define HubTap_HubCounter_HPP

#include "SentinelRef/HubCounter/HubCounterComponentAc.hpp"

namespace HubTap {

class HubCounter final : public HubCounterComponentBase {
  public:
    //! The retrainer's five telemetry channels occupy consecutive ids from its
    //! deployment base. Fixed at init and never grown (CPP-1).
    static const U32 CHANNELS = 5U;

    //! `SentinelRetrain`'s base id. The retrainer's SampleCount sits here and
    //! the other four follow. Stated as a constant rather than discovered,
    //! because a tap that guesses which ids it is counting is not a counter.
    static const U32 RETRAIN_BASE = 0x30000000U;

    //! What one source tick feeds, so SampleCount converts to a tick count.
    //! Mirrors `Retrain::Retrainer::SAMPLES_PER_TICK`.
    static const U32 SAMPLES_PER_TICK = 10U;

    explicit HubCounter(const char* const compName);
    ~HubCounter() override;

    //! Total telemetry points seen, for a test to read without a downlink.
    U32 channelsSeen() const { return m_channels; }
    //! Total events seen.
    U32 eventsSeen() const { return m_events; }
    //! Ticks that delivered all five channels.
    U32 completeTicks() const { return m_complete; }

  private:
    void eventIn_handler(FwIndexType portNum, FwEventIdType id, Fw::Time& timeTag,
                         const Fw::LogSeverity& severity, Fw::LogBuffer& args) override;
    void tlmIn_handler(FwIndexType portNum, FwChanIdType id, Fw::Time& timeTag,
                       Fw::TlmBuffer& val) override;
    void schedIn_handler(FwIndexType portNum, U32 context) override;

    U32 m_events;
    U32 m_channels;
    U32 m_ticks;                    //!< derived from the largest SampleCount seen
    U32 m_complete;                 //!< ticks that delivered all five
    U32 m_perChannel[CHANNELS];     //!< arrivals per channel offset
};

}  // namespace HubTap

#endif
