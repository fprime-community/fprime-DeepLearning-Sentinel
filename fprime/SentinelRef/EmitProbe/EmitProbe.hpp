// ======================================================================
// \title  EmitProbe.hpp
// \brief  LC2's detector-side log: the ticks the Monitor's detector emitted on
//
// docs/MODELS.md 78.11. Apparatus in SentinelRef's loop variant only; see EmitProbe.fpp.
// ======================================================================
#ifndef EmitTap_EmitProbe_HPP
#define EmitTap_EmitProbe_HPP

#include "SentinelRef/EmitProbe/EmitProbeComponentAc.hpp"
#include "Sentinel/Monitor/Monitor.hpp"
#include "Sentinel/SampleTap/SampleTap.hpp"

namespace EmitTap {

class EmitProbe final : public EmitProbeComponentBase {
  public:
    //! A heartbeat line every this many tap sequence numbers, so a script can wait on
    //! the ticks the plant actually reached rather than on a wall clock.
    static const U32 HEARTBEAT = 1000U;

    explicit EmitProbe(const char* const compName);
    ~EmitProbe() override;

    //! Topology setup. With `enabled` false the component does nothing at all.
    void configure(const Sentinel::Monitor* monitor, const Sentinel::SampleTap* tap,
                   bool enabled);

  private:
    void schedIn_handler(FwIndexType portNum, U32 context) override;

    const Sentinel::Monitor* m_monitor;
    const Sentinel::SampleTap* m_tap;
    bool m_enabled;
    bool m_haveSeq;
    U32 m_lastSeq;
    bool m_wasModel;
};

}  // namespace EmitTap

#endif
