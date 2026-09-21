// ======================================================================
// \title  Retrainer.cpp
// \brief  E1, the pipe. docs/MODELS.md 47.
// ======================================================================
#include "SentinelRef/Retrainer/Retrainer.hpp"
#include "SentinelRef/Retrainer/FppConstantsAc.hpp"

#include "sentinel_retrainer.h"
#include "sentinel_cycle.h"
#include "sentinel_shadow.h"

#include <Os/File.hpp>

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
      m_cycleArmed(false), m_capacity(static_cast<U32>(RETRAINER_CAPACITY)),
      m_flyingPath(""), m_candidatePath(""), m_shadowArmed(false),
      m_candidateBytes(0U), m_candidateCrc32(0U)
{
    m_loss[0] = 0.0F;
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

void Retrainer::configureShadow(const char* flyingPath, const char* candidatePath)
{
    m_flyingPath = (flyingPath != nullptr) ? flyingPath : "";
    m_candidatePath = (candidatePath != nullptr) ? candidatePath : "";
}

//! Read the flying model file into m_modelFile. Returns its length, or 0.
//!
//! (!) IT CALLS NO OCaml ENTRY POINT, AND THAT IS DELIBERATE.
//! tests/test_ocaml_domain_lock_is_thread_pinned.py permits an OCaml entry point
//! in `boot` and `schedIn_handler` and nowhere else, because OCaml 5 pins the
//! domain lock to the thread that called caml_startup. Putting the
//! sentinel_shadow_load call in here would have meant widening that set to admit
//! a third name -- weakening the guard to fit the code, when the code can just as
//! easily fit the guard. So this reads bytes and the caller crosses the boundary.
U32 Retrainer::readFlying()
{
    if (m_flyingPath.length() == 0U) {
        return 0U;
    }
    Os::File file;
    if (file.open(m_flyingPath.toChar(), Os::File::OPEN_READ) != Os::File::OP_OK) {
        return 0U;
    }
    FwSizeType wanted = static_cast<FwSizeType>(MODEL_FILE_MAX_BYTES);
    const Os::File::Status readStatus = file.read(m_modelFile, wanted);
    file.close();
    if (readStatus != Os::File::OP_OK) {
        return 0U;
    }
    return static_cast<U32>(wanted);
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
    // 72 / E5-e, HO1: the flying file is read ONCE, here, not per tick. A file
    // this process cannot read leaves it ticking and cycling as before, simply
    // producing no candidate -- the same degrade-rather-than-die posture the
    // runtime failure above takes.
    const U32 flyingBytes = m_cycleArmed ? this->readFlying() : 0U;
    m_shadowArmed = (flyingBytes > 0U)
                    && (sentinel_shadow_load(m_modelFile, flyingBytes)
                        == SENTINEL_SHD_OK);
    this->log_ACTIVITY_HI_RuntimeBooted(m_capacity);
    this->tlmWrite_LastStatus(RetrainerStatus::OK);
    return true;
}

void Retrainer::refuseCandidate(CandidateStage::T stage, I32 code)
{
    m_refusals++;
    this->log_WARNING_LO_CandidateRefused(CandidateStage(stage), code);
}

//! Write the exported bytes out and report them. NO OCaml ENTRY POINT IS CALLED
//! HERE -- the bytes are already in this process's own buffer, so the domain
//! lock is not in question and the helper is free to exist.
void Retrainer::emitCandidate(U32 n, I32 steps)
{
    Os::File file;
    if (file.open(m_candidatePath.toChar(), Os::File::OPEN_CREATE,
                  Os::File::OVERWRITE) != Os::File::OP_OK) {
        this->refuseCandidate(CandidateStage::FILE_OPEN, 0);
        return;
    }
    FwSizeType size = static_cast<FwSizeType>(n);
    const Os::File::Status wrote = file.write(m_modelFile, size);
    file.close();
    if ((wrote != Os::File::OP_OK) || (static_cast<U32>(size) != n)) {
        this->refuseCandidate(CandidateStage::FILE_WRITE,
                              static_cast<I32>(size));
        return;
    }

    // The file's OWN static_crc32, read back from offset 44 of the bytes just
    // written (shadow59.ml:36,93). It crosses to the ground by a different path
    // from the file, so the ground checks the bytes it received against a number
    // that did not travel with them.
    m_candidateCrc32 =
        static_cast<U32>(m_modelFile[MODEL_STATIC_CRC_OFFSET])
        | (static_cast<U32>(m_modelFile[MODEL_STATIC_CRC_OFFSET + 1U]) << 8)
        | (static_cast<U32>(m_modelFile[MODEL_STATIC_CRC_OFFSET + 2U]) << 16)
        | (static_cast<U32>(m_modelFile[MODEL_STATIC_CRC_OFFSET + 3U]) << 24);
    m_candidateBytes = n;

    this->log_ACTIVITY_HI_CandidateWritten(n, m_candidateCrc32, steps, m_loss[0],
                                           static_cast<U32>(m_export[0]));
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

        // 72 / E5-e, HO1: the candidate leaves this process.
        //
        // (!) WRITTEN INLINE AND NOT IN A HELPER, because
        // tests/test_ocaml_domain_lock_is_thread_pinned.py asserts that every
        // OCaml entry point sits in `boot` or `schedIn_handler`. OCaml 5 pins the
        // domain lock to the thread that called caml_startup, and that assertion
        // is what keeps a future edit from moving one of these calls onto another
        // thread. A helper would satisfy the letter of it only by widening the
        // permitted set, which is the wrong direction.
        //
        // Fixed work per cycle, still: one export, one write, one export, one
        // file. No loop whose length depends on anything (Objective.md 11 rule 5).
        if (m_shadowArmed && (rc == SENTINEL_CYC_OK)) {
            const I32 exRc = sentinel_cycle_export(m_weights, CYCLE_N_PARAMS);
            if (exRc != SENTINEL_CYC_OK) {
                this->refuseCandidate(CandidateStage::CYCLE_EXPORT, exRc);
            } else {
                const I32 wrRc = sentinel_shadow_write(m_weights, CYCLE_N_PARAMS);
                if (wrRc != SENTINEL_SHD_OK) {
                    this->refuseCandidate(CandidateStage::SHADOW_WRITE, wrRc);
                } else {
                    const I32 n = sentinel_shadow_export(m_modelFile,
                                                         MODEL_FILE_MAX_BYTES);
                    if (n <= 0) {
                        this->refuseCandidate(CandidateStage::SHADOW_EXPORT, n);
                    } else {
                        (void)sentinel_shadow_loss(m_loss, 1U);
                        this->emitCandidate(static_cast<U32>(n), steps);
                    }
                }
            }
        }
    }

    const U32 count = static_cast<U32>(m_export[0]);
    this->tlmWrite_SampleCount(count);
    this->tlmWrite_Sum(m_export[1]);
    this->tlmWrite_Mean(m_export[2]);
    this->tlmWrite_LastStatus(RetrainerStatus::OK);
    this->log_ACTIVITY_LO_StepComplete(count, m_export[1], m_export[2]);
}

}  // namespace Retrain
