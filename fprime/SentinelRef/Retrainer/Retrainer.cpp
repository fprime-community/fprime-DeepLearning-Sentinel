// ======================================================================
// \title  Retrainer.cpp
// \brief  E1, the pipe. docs/MODELS.md 47.
// ======================================================================
#include "SentinelRef/Retrainer/Retrainer.hpp"
#include "SentinelRef/Retrainer/FppConstantsAc.hpp"

#include "sentinel_retrainer.h"
#include "sentinel_cycle.h"

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
    : RetrainerComponentBase(compName), m_armed(false), m_refusals(0U),
      m_cycleArmed(false), m_capacity(static_cast<U32>(RETRAINER_CAPACITY))
{
    for (U32 i = 0U; i < SAMPLES_PER_TICK; ++i) {
        m_samples[i] = static_cast<F64>(i + 1U);   // 1.0 .. 10.0; sum 55, mean 5.5
    }
    for (U32 i = 0U; i < EXPORT_WIDTH; ++i) {
        m_export[i] = 0.0;
    }
    // 61 / E5-c: a deterministic, bounded drive, filled once. Not telemetry --
    // a real deployment writes this from the window 56's gate admitted.
    for (U32 t = 0U; t < 250U; ++t) {
        for (U32 c = 0U; c < CYCLE_INS; ++c) {
            const U32 mix = ((t * 2654435761U) + (c * 40503U)) & 0xFFFFU;
            m_window[(t * CYCLE_INS) + c] = static_cast<F32>(mix) / 65535.0F;
        }
    }
}

Retrainer::~Retrainer() {}

void Retrainer::configure(U32 capacity)
{
    if (capacity > 0U) {
        m_capacity = capacity;
    }
}

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
    m_bootAttempted = true;
    // (!) A runtime that did not start is not a transient fault, so this does not
    // retry and the component goes inert rather than failing the topology.
    if (sentinel_retrainer_boot() != SENTINEL_RT_OK) {
        this->log_WARNING_HI_RuntimeUnavailable();
        m_armed = false;
        return false;
    }
    if (!this->accept(sentinel_retrainer_init(m_capacity))) {
        m_armed = false;
        return false;
    }
    m_armed = true;
    // 61 / E5-c: the cycle's runtime boots alongside E1's pipe. Both call
    // caml_startup, which runs at most once per process, so the second is a
    // no-op -- and BOTH surfaces live in one object for exactly that reason.
    m_cycleArmed = (sentinel_cycle_boot() == SENTINEL_CYC_OK)
                   && (sentinel_cycle_init(7) == SENTINEL_CYC_OK);
    this->log_ACTIVITY_HI_RuntimeBooted(m_capacity);
    this->tlmWrite_LastStatus(RetrainerStatus::OK);
    return true;
}

void Retrainer::schedIn_handler(FwIndexType portNum, U32 context)
{
    (void)portNum;
    (void)context;

    // (!) THE RUNTIME STARTS HERE, ON THIS THREAD, AND THE REASON IS OCaml 5's
    // domain lock. caml_startup grants the lock to its CALLER; an ActiveRateGroup
    // runs this handler on its own task, so a runtime started in the topology's
    // configureTopology() aborts the process with "Fatal error: no domain lock
    // held" on the first call in. Deferring the boot to the first tick makes the
    // thread that owns the domain the same one that uses it, for the life of the
    // process -- and D70 consequence 9's "exactly one domain, never Domain.spawn"
    // is unchanged. Found at docs/MODELS.md 65; E1 and 61 never saw it because a
    // unit-test process does all of this on one thread.
    if (!m_bootAttempted) {
        static_cast<void>(this->boot());
    }

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

    // 61 / E5-c: one retraining cycle per tick, at a FIXED step budget (D73).
    if (m_cycleArmed) {
        (void)sentinel_cycle_load(m_window, CYCLE_WINDOW);
        const I32 rc = sentinel_cycle_run(CYCLE_BUDGET, static_cast<I32>(CYCLE_T), 1);
        const I32 steps = (rc == SENTINEL_CYC_OK) ? sentinel_cycle_steps() : -1;
        this->tlmWrite_CycleSteps(steps);
    }

    const U32 count = static_cast<U32>(m_export[0]);
    this->tlmWrite_SampleCount(count);
    this->tlmWrite_Sum(m_export[1]);
    this->tlmWrite_Mean(m_export[2]);
    this->tlmWrite_LastStatus(RetrainerStatus::OK);
    this->log_ACTIVITY_LO_StepComplete(count, m_export[1], m_export[2]);
}

}  // namespace Retrain
