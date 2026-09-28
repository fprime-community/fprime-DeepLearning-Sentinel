// ======================================================================
// \title  EmitProbe.cpp
// \brief  LC2's detector-side log (docs/MODELS.md 78.11)
// ======================================================================
#include "SentinelRef/EmitProbe/EmitProbe.hpp"

#include <Fw/Logger/Logger.hpp>

namespace EmitTap {

EmitProbe::EmitProbe(const char* const compName)
    : EmitProbeComponentBase(compName), m_monitor(nullptr), m_tap(nullptr), m_enabled(false),
      m_haveSeq(false), m_lastSeq(0U), m_wasModel(false)
{
}

EmitProbe::~EmitProbe() {}

void EmitProbe::configure(const Sentinel::Monitor* monitor, const Sentinel::SampleTap* tap,
                          bool enabled)
{
    m_monitor = monitor;
    m_tap = tap;
    m_enabled = enabled && (monitor != nullptr) && (tap != nullptr);
}

void EmitProbe::schedIn_handler(FwIndexType portNum, U32 context)
{
    (void)portNum;
    (void)context;
    if (!m_enabled) {
        return;
    }
    // The tap numbers the sample it forwards from 0, so the sample the Monitor just
    // stepped on is forwarded() - 1. No sample yet: nothing was stepped on a sample.
    const U32 forwarded = m_tap->forwarded();
    if (forwarded == 0U) {
        return;
    }
    const U32 seq = forwarded - 1U;
    // A tick on which the plant forwarded nothing new is reported, not guessed at:
    // the Monitor still steps (valid false), and LC2 is then NO VERDICT.
    if (m_haveSeq && (seq == m_lastSeq)) {
        Fw::Logger::log("PROBE_STALE seq %u\n", seq);
        return;
    }
    if (m_haveSeq && (seq != (m_lastSeq + 1U))) {
        Fw::Logger::log("PROBE_GAP seq %u after %u\n", seq, m_lastSeq);
    }
    if (!m_haveSeq) {
        Fw::Logger::log("PROBE_FIRST seq %u\n", seq);
    }
    m_haveSeq = true;
    m_lastSeq = seq;

    const bool model = (m_monitor->activeMode() == Sentinel::Mode::MODEL);
    if (model != m_wasModel) {
        Fw::Logger::log("PROBE_MODE seq %u model %u\n", seq, model ? 1U : 0U);
        m_wasModel = model;
    }
    if (model && m_monitor->detector().emitted()) {
        Fw::Logger::log("EMIT seq %u\n", seq);
    }
    if ((seq % HEARTBEAT) == 0U) {
        Fw::Logger::log("TICK seq %u\n", seq);
    }
}

}  // namespace EmitTap
