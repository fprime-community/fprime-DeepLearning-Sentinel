// ======================================================================
// \title  Retrainer.cpp
// \brief  E1, the pipe. docs/MODELS.md 47.
// ======================================================================
#include "SentinelRef/Retrainer/Retrainer.hpp"
#include "SentinelRef/Retrainer/FppConstantsAc.hpp"

#include "sentinel_retrainer.h"

namespace Retrain {

namespace {

//! Map the C header's status to the FPP enum. A switch rather than a cast, so a
//! code added on one side and not the other is a compile error rather than a
//! number nobody recognises -- the same argument `Sentinel::Status` makes.
RetrainerStatus toFpp(I32 status)
{
    switch (status) {
        case SENTINEL_RT_OK:               return RetrainerStatus::OK;
        case SENTINEL_RT_ERR_NOT_INIT:     return RetrainerStatus::ERR_NOT_INIT;
        case SENTINEL_RT_ERR_ALREADY_INIT: return RetrainerStatus::ERR_ALREADY_INIT;
        case SENTINEL_RT_ERR_CAPACITY:     return RetrainerStatus::ERR_CAPACITY;
        case SENTINEL_RT_ERR_OVERFLOW:     return RetrainerStatus::ERR_OVERFLOW;
        case SENTINEL_RT_ERR_SHORT_BUFFER: return RetrainerStatus::ERR_SHORT_BUFFER;
        case SENTINEL_RT_ERR_NO_RUNTIME:   return RetrainerStatus::ERR_NO_RUNTIME;
        default:                           return RetrainerStatus::ERR_INTERNAL;
    }
}

}  // namespace

Retrainer::Retrainer(const char* const compName)
    : RetrainerComponentBase(compName), m_armed(false), m_refusals(0U)
{
    for (U32 i = 0U; i < SAMPLES_PER_TICK; ++i) {
        m_samples[i] = static_cast<F64>(i + 1U);   // 1.0 .. 10.0; sum 55, mean 5.5
    }
    for (U32 i = 0U; i < EXPORT_WIDTH; ++i) {
        m_export[i] = 0.0;
    }
}

Retrainer::~Retrainer() {}

bool Retrainer::accept(I32 status)
{
    if (status == SENTINEL_RT_OK) {
        return true;
    }
    m_refusals++;
    this->log_WARNING_LO_CallRefused(toFpp(status));
    this->tlmWrite_LastStatus(toFpp(status));
    return false;
}

bool Retrainer::boot()
{
    // (!) A runtime that did not start is not a transient fault, so this does not
    // retry and the component goes inert rather than failing the topology.
    if (sentinel_retrainer_boot() != SENTINEL_RT_OK) {
        this->log_WARNING_HI_RuntimeUnavailable();
        m_armed = false;
        return false;
    }
    if (!this->accept(sentinel_retrainer_init(static_cast<U32>(RETRAINER_CAPACITY)))) {
        m_armed = false;
        return false;
    }
    m_armed = true;
    this->log_ACTIVITY_HI_RuntimeBooted(static_cast<U32>(RETRAINER_CAPACITY));
    this->tlmWrite_LastStatus(RetrainerStatus::OK);
    return true;
}

void Retrainer::schedIn_handler(FwIndexType portNum, U32 context)
{
    (void)portNum;
    (void)context;

    if (!m_armed) {
        return;   // inert, and silent: RuntimeUnavailable was raised once at boot
    }

    // Fixed work per cycle: one feed of a fixed block, one step, one export.
    // Objective.md 11 rule 5 -- no convergence criterion, no variable iteration.
    if (!this->accept(sentinel_retrainer_feed(m_samples, SAMPLES_PER_TICK))) {
        return;
    }
    if (!this->accept(sentinel_retrainer_step())) {
        return;
    }
    if (!this->accept(sentinel_retrainer_export(m_export, EXPORT_WIDTH))) {
        return;
    }

    const U32 count = static_cast<U32>(m_export[0]);
    this->tlmWrite_SampleCount(count);
    this->tlmWrite_Sum(m_export[1]);
    this->tlmWrite_Mean(m_export[2]);
    this->tlmWrite_LastStatus(RetrainerStatus::OK);
    this->log_ACTIVITY_LO_StepComplete(count, m_export[1], m_export[2]);
}

}  // namespace Retrain
