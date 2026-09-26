// ======================================================================
// \title  RetrainLoop.cpp
// \brief  D85: the retrainer's onboard logic. RetrainLoop.hpp carries the design.
// ======================================================================
#include "RetrainLoop.hpp"

#include "sentinel_cycle.h"
#include "sentinel_shadow.h"
#include "sentinel_window.h"

namespace Sentinel {

RetrainLoop::RetrainLoop()
    : m_config(), m_configured(false), m_booted(false), m_replica(), m_flying(),
      m_flyingBytes(0U), m_candidate(), m_candidateBytes(0U), m_ring(), m_ringValid(),
      m_head(0U), m_filled(0U), m_tick(0U), m_lastFlag(0U), m_everFlagged(false),
      m_window(), m_weights(), m_stepsSince(0U), m_admittedTotal(0U), m_candFirst(0U),
      m_candLast(0U), m_spanFirst(0U), m_candSteps(0U), m_candidates(0U)
{
}

bool RetrainLoop::configure(const LoopConfig& config)
{
    if ((config.guard > GUARD_MAX) || (config.guard < PREDICTIONS) || (config.budget <= 0)
        || (config.schedule == 0U) || (config.nominalPpm <= 0)) {
        return false;
    }
    for (U32 c = 0U; c < CHANNELS; ++c) {
        if (!(config.yellowLow[c] < config.yellowHigh[c])) {
            return false;
        }
    }
    m_config = config;
    m_configured = true;
    return true;
}

bool RetrainLoop::setFlying(const U8* bytes, U32 length)
{
    if ((bytes == nullptr) || (length == 0U) || (length > FILE_MAX_BYTES)) {
        return false;
    }
    for (U32 i = 0U; i < length; ++i) {
        m_flying[i] = bytes[i];
    }
    m_flyingBytes = length;
    // The replica judges every tick exactly as the detector does: same file, same core.
    if (m_replica.load(m_flying, m_flyingBytes) != LoadStatus::OK) {
        m_flyingBytes = 0U;
        return false;
    }
    return m_replica.model().nChannels == CHANNELS;
}

I32 RetrainLoop::warmFromFlying()
{
    // The shadow holds the flying file as the candidate's template; the warm start
    // copies its weights into the cycle and resets the optimiser (shadow_c.ml).
    I32 rc = sentinel_shadow_load(m_flying, m_flyingBytes);
    if (rc != SENTINEL_SHD_OK) {
        return rc;
    }
    return sentinel_shadow_warm();
}

I32 RetrainLoop::boot()
{
    if (!m_configured || (m_flyingBytes == 0U)) {
        return -1;
    }
    I32 rc = sentinel_cycle_boot();
    if (rc != SENTINEL_CYC_OK) {
        return rc;
    }
    rc = sentinel_cycle_init(7);
    if (rc != SENTINEL_CYC_OK) {
        return rc;
    }
    rc = sentinel_window_reset();
    if (rc != SENTINEL_WIN_OK) {
        return rc;
    }
    rc = sentinel_window_configure(m_config.nominalPpm);
    if (rc != SENTINEL_WIN_OK) {
        return rc;
    }
    rc = warmFromFlying();
    if (rc != SENTINEL_SHD_OK) {
        return rc;
    }
    m_booted = true;
    return 0;
}

U32 RetrainLoop::candidateCrc32() const
{
    if (m_candidateBytes < 48U) {
        return 0U;
    }
    return static_cast<U32>(m_candidate[44]) | (static_cast<U32>(m_candidate[45]) << 8)
         | (static_cast<U32>(m_candidate[46]) << 16) | (static_cast<U32>(m_candidate[47]) << 24);
}

I32 RetrainLoop::emitCandidate()
{
    I32 rc = sentinel_cycle_export(m_weights, N_PARAMS);
    if (rc != SENTINEL_CYC_OK) {
        return rc;
    }
    rc = sentinel_shadow_write(m_weights, N_PARAMS);
    if (rc != SENTINEL_SHD_OK) {
        return rc;
    }
    const I32 n = sentinel_shadow_export(m_candidate, FILE_MAX_BYTES);
    if (n <= 0) {
        return (n < 0) ? n : -1;
    }
    m_candidateBytes = static_cast<U32>(n);
    ++m_candidates;
    // A fresh warm start for the next candidate: every candidate is the flying model
    // fine-tuned on its own E admitted steps, which is D78's geometry and is what lets
    // the ground reproduce the control exactly.
    return warmFromFlying();
}

LoopTick RetrainLoop::tick(const F32* sample, bool valid)
{
    LoopTick out = {false, false, false, false, false, false, 0};
    if (!m_booted || (sample == nullptr)) {
        out.status = -1;
        return out;
    }

    // -- the detector's verdict, reproduced -------------------------------------
    m_replica.step(sample, valid);
    const bool warmed = (m_replica.steps() >= static_cast<U64>(m_replica.model().warmupSteps));
    out.emitted = m_replica.emitted();
    out.crossing = warmed && m_replica.crossing();

    // -- the engineering dictionary: any channel outside its yellow band ----------
    bool limit = false;
    if (valid) {
        for (U32 c = 0U; c < CHANNELS; ++c) {
            if ((sample[c] < m_config.yellowLow[c]) || (sample[c] > m_config.yellowHigh[c])) {
                limit = true;
            }
        }
    }
    out.limitAny = limit;

    // -- 56's rule over the trailing W --------------------------------------------
    I32 rc = sentinel_window_push(limit ? 1 : 0, out.emitted ? 1 : 0);
    if (rc < 0) {
        out.status = rc;
        return out;
    }
    const I32 admits = sentinel_window_admits();
    if (admits < 0) {
        out.status = admits;
        return out;
    }
    out.windowAdmits = (admits == 1);

    // -- the ring, and the guard band -------------------------------------------
    for (U32 c = 0U; c < CHANNELS; ++c) {
        m_ring[m_head][c] = valid ? sample[c] : 0.0F;
    }
    m_ringValid[m_head] = valid;
    m_head = (m_head + 1U) % RING;
    if (m_filled < RING) {
        ++m_filled;
    }
    ++m_tick;
    if (out.crossing || out.emitted || limit || !valid) {
        m_lastFlag = m_tick;
        m_everFlagged = true;
    }

    // The effective ring: SPAN rows with GUARD rows either side. Nothing flagged
    // anywhere in it, so nothing within GUARD ticks of a warning, a limit tick or a
    // missed sample -- before OR after the training span -- is trained on.
    const U32 extent = SPAN + (2U * m_config.guard);
    const bool quiet = (!m_everFlagged) || ((m_tick - m_lastFlag) >= static_cast<U64>(extent));
    const bool admit = warmed && out.windowAdmits && (m_filled >= extent) && quiet;

    if (admit) {
        // The training span: the middle SPAN rows of the effective ring, oldest first.
        // Row r of the span is the sample written (extent - guard - r) ticks ago.
        for (U32 r = 0U; r < SPAN; ++r) {
            const U32 back = extent - m_config.guard - r;      // 1 .. extent
            const U32 slot = (m_head + RING - back) % RING;
            for (U32 c = 0U; c < CHANNELS; ++c) {
                m_window[(r * CHANNELS) + c] = m_ring[slot][c];
            }
        }
        rc = sentinel_cycle_load(m_window, SPAN * CHANNELS);
        if (rc != SENTINEL_CYC_OK) {
            out.status = rc;
            return out;
        }
        rc = sentinel_cycle_run(m_config.budget, static_cast<I32>(WINDOW), 1);
        if (rc != SENTINEL_CYC_OK) {
            out.status = rc;
            return out;
        }
        const I32 steps = sentinel_cycle_steps();
        if (steps > 0) {
            if (m_stepsSince == 0U) {
                m_spanFirst = m_tick;
            }
            m_stepsSince += static_cast<U32>(steps);
            m_admittedTotal += 1U;
            out.admitted = true;
        }
        if (m_stepsSince >= m_config.schedule) {
            m_candFirst = m_spanFirst;
            m_candLast = m_tick;
            m_candSteps = m_stepsSince;
            rc = emitCandidate();
            m_stepsSince = 0U;
            if (rc < 0) {
                out.status = rc;
                return out;
            }
            out.candidateReady = true;
        }
    }
    return out;
}

}  // namespace Sentinel
