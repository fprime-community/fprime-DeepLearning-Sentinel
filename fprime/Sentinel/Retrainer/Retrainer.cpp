// ======================================================================
// \title  Retrainer.cpp
// \brief  D85: the retrainer's F' shell. Retrainer.hpp and RetrainLoop.hpp carry the
//         design.
// ======================================================================
#include "Sentinel/Retrainer/Retrainer.hpp"
#include "Sentinel/Retrainer/FppConstantsAc.hpp"

#include <Os/File.hpp>

namespace Sentinel {

namespace {

RetrainerStatus toFpp(I32 status)
{
    return (status == 0) ? RetrainerStatus::OK : RetrainerStatus::ERR_INTERNAL;
}

}  // namespace

Retrainer::Retrainer(const char* const compName)
    : RetrainerComponentBase(compName), m_loop(), m_configured(false), m_armed(false),
      m_bootAttempted(false), m_lock(), m_queue(), m_queueValid(), m_queueSeq(),
      m_queued(0U), m_overrun(0U), m_batch(), m_batchValid(), m_batchSeq(),
      m_haveSeq(false), m_nextSeq(0U), m_spanOffset(0U), m_received(0U), m_missed(0U), m_file(),
      m_flyingPath(""), m_candidatePath("")
{
}

Retrainer::~Retrainer() {}

void Retrainer::configureShadow(const char* flyingPath, const char* candidatePath)
{
    m_flyingPath = (flyingPath != nullptr) ? flyingPath : "";
    m_candidatePath = (candidatePath != nullptr) ? candidatePath : "";
}

bool Retrainer::configureLoop(const LoopConfig& config)
{
    m_configured = m_loop.configure(config);
    return m_configured;
}

//! Read the flying model file. No OCaml entry point: boot() crosses the boundary.
U32 Retrainer::readFlying()
{
    if (m_flyingPath.length() == 0U) {
        return 0U;
    }
    Os::File file;
    if (file.open(m_flyingPath.toChar(), Os::File::OPEN_READ) != Os::File::OP_OK) {
        return 0U;
    }
    FwSizeType wanted = static_cast<FwSizeType>(RetrainLoop::FILE_MAX_BYTES);
    const Os::File::Status readStatus = file.read(m_file, wanted);
    file.close();
    if (readStatus != Os::File::OP_OK) {
        return 0U;
    }
    return static_cast<U32>(wanted);
}

bool Retrainer::boot()
{
    m_bootAttempted = true;
    const U32 flyingBytes = this->readFlying();
    if (!m_configured || (flyingBytes == 0U) || !m_loop.setFlying(m_file, flyingBytes)) {
        this->log_WARNING_LO_CandidateRefused(CandidateStage::FLYING_READ,
                                               static_cast<I32>(flyingBytes));
        m_armed = false;
        return false;
    }
    // (!) THE RUNTIME STARTS HERE, ON THE RATE-GROUP THREAD, via RetrainLoop::boot.
    // caml_startup grants the domain lock to its caller, so booting in the topology's
    // configureTopology() aborts the process on the first call in (docs/MODELS.md 65).
    const I32 rc = m_loop.boot();
    if (rc != 0) {
        this->log_WARNING_HI_RuntimeUnavailable();
        this->tlmWrite_LastStatus(toFpp(rc));
        m_armed = false;
        return false;
    }
    m_armed = true;
    this->log_ACTIVITY_HI_RetrainerReady(RetrainLoop::CHANNELS, m_loop.schedule());
    this->tlmWrite_LastStatus(RetrainerStatus::OK);
    return true;
}

//! On the hub's receive thread. Queue only; never an OCaml call.
void Retrainer::sampleIn_handler(FwIndexType portNum, Sentinel::ChannelVector& values,
                                 bool valid, U32 seq)
{
    (void)portNum;
    m_lock.lock();
    if (m_queued < QUEUE) {
        for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
            m_queue[m_queued][c] = values[static_cast<FwSizeType>(c)];
        }
        m_queueValid[m_queued] = valid;
        m_queueSeq[m_queued] = seq;
        ++m_queued;
    } else {
        ++m_overrun;          // counted, and seen as a sequence gap by the next tick
    }
    m_lock.unLock();
}

void Retrainer::emitCandidate(U32 lastOffset)
{
    Os::File file;
    if (file.open(m_candidatePath.toChar(), Os::File::OPEN_CREATE,
                  Os::File::OVERWRITE) != Os::File::OP_OK) {
        this->log_WARNING_LO_CandidateRefused(CandidateStage::FILE_OPEN, 0);
        return;
    }
    FwSizeType size = static_cast<FwSizeType>(m_loop.candidateBytes());
    const Os::File::Status wrote = file.write(m_loop.candidate(), size);
    file.close();
    if ((wrote != Os::File::OP_OK) || (static_cast<U32>(size) != m_loop.candidateBytes())) {
        this->log_WARNING_LO_CandidateRefused(CandidateStage::FILE_WRITE, static_cast<I32>(size));
        return;
    }
    this->log_ACTIVITY_HI_CandidateWritten(m_loop.candidateBytes(), m_loop.candidateCrc32(),
                                           m_loop.candidateSteps(),
                                           static_cast<U32>(m_loop.candidateFirstTick()) + m_spanOffset,
                                           static_cast<U32>(m_loop.candidateLastTick()) + lastOffset);
}

void Retrainer::feed(const F32* values, bool valid, U32 offset)
{
    const U32 before = m_loop.stepsSinceCandidate();
    const LoopTick out = m_loop.tick(values, valid);
    if (out.status != 0) {
        this->log_WARNING_LO_CallRefused(toFpp(out.status));
        this->tlmWrite_LastStatus(toFpp(out.status));
        return;
    }
    if (out.admitted && (before == 0U)) {
        m_spanOffset = offset;       // this candidate's first admitted step
    }
    if (out.candidateReady) {
        this->emitCandidate(offset);
    }
}

void Retrainer::schedIn_handler(FwIndexType portNum, U32 context)
{
    (void)portNum;
    (void)context;
    if (!m_bootAttempted) {
        static_cast<void>(this->boot());
    }
    if (!m_armed) {
        return;   // inert, and silent: the refusal was reported once at boot
    }

    // Drain under the lock, then work outside it: the hub thread never waits on a
    // training step, and no OCaml call is made while the lock is held.
    m_lock.lock();
    const U32 n = m_queued;
    for (U32 i = 0U; i < n; ++i) {
        for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
            m_batch[i][c] = m_queue[i][c];
        }
        m_batchValid[i] = m_queueValid[i];
        m_batchSeq[i] = m_queueSeq[i];
    }
    m_queued = 0U;
    m_lock.unLock();

    // Fixed work: at most QUEUE samples, and at most QUEUE missed ticks fed for gaps.
    static const F32 ZEROS[Config::MAX_CHANNELS] = {};
    U32 lost = 0U;
    for (U32 i = 0U; i < n; ++i) {
        if (m_haveSeq && (m_batchSeq[i] != m_nextSeq)) {
            const U32 gap = m_batchSeq[i] - m_nextSeq;
            const U32 fill = (gap < QUEUE) ? gap : QUEUE;
            for (U32 k = 0U; k < fill; ++k) {
                this->feed(ZEROS, false, 0U);   // never admitted: a missed tick
            }
            lost += gap;
        }
        this->feed(m_batch[i], m_batchValid[i],
                   m_batchSeq[i] - static_cast<U32>(m_loop.ticks()));
        m_haveSeq = true;
        m_nextSeq = m_batchSeq[i] + 1U;
        ++m_received;
    }
    if (lost > 0U) {
        m_missed += lost;
        this->log_WARNING_LO_SamplesLost(lost);
    }
    this->tlmWrite_SamplesReceived(m_received);
    this->tlmWrite_Admitted(static_cast<U32>(m_loop.admittedTotal()));
    this->tlmWrite_Candidates(m_loop.candidates());
    this->tlmWrite_SamplesMissed(m_missed);
}

}  // namespace Sentinel
