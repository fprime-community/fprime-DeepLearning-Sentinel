// ======================================================================
// \title  RetrainLoop.hpp
// \brief  D85: the retrainer's onboard logic, in plain C++ -- one class that the
//         F' component, the host simulation and the tests all run
//
// Until D85 the Retrainer trained a fixed synthetic drive, took `admit = 1` for
// granted, never warm-started, and stopped cycling at tick 410. This class is what
// makes it learn from the spacecraft's own telemetry, and ONLY from recent healthy
// telemetry:
//
//   sample --> ring (SPAN + 2 GUARD rows) --> replica of the flight core --+
//                                                                          |
//     56's rule (G-limit, G-rate, G-suff over the trailing 6,550)  <-------+
//     + the guard band: no detector CROSSING, on any channel, anywhere in the
//       ring -- so nothing within GUARD ticks of a warning, before or after the
//       training span, is ever trained on
//          |
//          v admitted
//     B optimiser steps on the ring's middle SPAN rows (input + future block)
//          |
//          v every E admitted steps
//     a candidate: the flying file with the trained weights; then the cycle is
//     reset to the flying weights, so every candidate is a fresh warm start
//     (D76, D78's geometry) and the ground can reproduce its control exactly
//
// (!) THE REPLICA IS HOW THE RETRAINER KNOWS WHAT THE DETECTOR DID, WITHOUT TOUCHING
// THE DETECTOR. Monitor has no data output and its warning event is throttled after
// ten. But the flight core is deterministic (Objective.md 11 rule 5): the same
// model and the same samples give the same `emitted()` and `crossing()` bit for bit.
// So this process runs its own copy -- 603 KB here, none in the detector's process.
//
// (!) FIXED MEMORY, FIXED WORK. Every buffer is a member sized at compile time from
// the generated shape; `tick` does a bounded amount of work whatever arrives
// (CPP-1, Objective.md 11 rule 5). Every OCaml entry point is called from `boot`
// and `tick` only, which the component calls from its boot and its rate-group
// handler -- the thread that owns the OCaml domain (docs/MODELS.md 65.10).
// ======================================================================
#ifndef Sentinel_RetrainLoop_HPP
#define Sentinel_RetrainLoop_HPP

#include "sentinel/Config.hpp"
#include "sentinel/Detector.hpp"
#include "sentinel/Types.hpp"
#include "sentinel_cycle_shape.h"

namespace Sentinel {

//! What a mission sets once, at topology setup.
struct LoopConfig {
    //! The engineering dictionary's YELLOW band per channel. 56's G-limit is "any
    //! channel outside any band", and yellow lies inside red, so yellow is enough.
    F32 yellowLow[Config::MAX_CHANNELS];
    F32 yellowHigh[Config::MAX_CHANNELS];
    //! Guard band, ticks, on each side of the training span. <= GUARD_MAX.
    U32 guard;
    //! EXPERIMENTAL: optimiser steps per admitted tick (D73's flight N is owed).
    I32 budget;
    //! EXPERIMENTAL: admitted steps per candidate.
    U32 schedule;
    //! The flying model's calibrated nominal emission rate, parts per million
    //! (56's G-rate: k x this x W).
    I32 nominalPpm;
};

//! What one tick did, for telemetry and for the simulation's trace.
struct LoopTick {
    bool emitted;          //!< the replica emitted this tick
    bool crossing;         //!< the replica's score crossed its threshold this tick
    bool limitAny;         //!< a channel was outside its yellow band
    bool windowAdmits;     //!< 56's rule admitted the trailing window
    bool admitted;         //!< a training step was taken this tick
    bool candidateReady;   //!< a candidate was completed this tick
    I32 status;            //!< 0, or the first negative status an entry point returned
};

class RetrainLoop {
  public:
    static const U32 CHANNELS = SENTINEL_CYCLE_CHANNELS;
    static const U32 PREDICTIONS = SENTINEL_CYCLE_PREDICTIONS;
    static const U32 WINDOW = SENTINEL_CYCLE_WINDOW;
    static const U32 SPAN = WINDOW + PREDICTIONS;        //!< input rows + target rows
    static const U32 N_PARAMS = SENTINEL_CYCLE_N_PARAMS;
    static const U32 GUARD_MAX = 512U;
    static const U32 RING = SPAN + (2U * GUARD_MAX);

    //! The largest model file budgeted for; same derivation as Retrainer.hpp.
    static const U32 FILE_MAX_BYTES =
        64U + (20U * Config::MAX_CHANNELS) + (4U * N_PARAMS) + 96U + (8U * Config::MAX_CHANNELS);

    RetrainLoop();

    //! Once, before `boot`. Returns false (and changes nothing) on a bad config.
    bool configure(const LoopConfig& config);

    //! The flying model file, once, before `boot`: loaded into the replica and kept
    //! as the template and warm-start source for every candidate. No OCaml call.
    bool setFlying(const U8* bytes, U32 length);

    //! Boots the OCaml side and warm-starts the cycle. Returns 0 or a negative status.
    I32 boot();

    //! One tick. `sample` is CHANNELS wide; `valid` is false when a sample was missed.
    LoopTick tick(const F32* sample, bool valid);

    //! The last completed candidate, and what it was trained on.
    const U8* candidate() const { return m_candidate; }
    U32 candidateBytes() const { return m_candidateBytes; }
    U32 candidateCrc32() const;
    U64 candidateFirstTick() const { return m_candFirst; }
    U64 candidateLastTick() const { return m_candLast; }
    U32 candidateSteps() const { return m_candSteps; }
    U32 candidates() const { return m_candidates; }

    U32 schedule() const { return m_config.schedule; }
    U64 ticks() const { return m_tick; }
    U64 admittedTotal() const { return m_admittedTotal; }
    U32 stepsSinceCandidate() const { return m_stepsSince; }
    const Detector& replica() const { return m_replica; }

  private:
    I32 warmFromFlying();
    I32 emitCandidate();

    LoopConfig m_config;
    bool m_configured;
    bool m_booted;

    Detector m_replica;
    U8 m_flying[FILE_MAX_BYTES];
    U32 m_flyingBytes;
    U8 m_candidate[FILE_MAX_BYTES];
    U32 m_candidateBytes;

    F32 m_ring[RING][Config::MAX_CHANNELS];
    bool m_ringValid[RING];
    U32 m_head;            //!< next slot to write
    U32 m_filled;
    U64 m_tick;
    U64 m_lastFlag;        //!< tick of the last crossing, invalid sample or limit tick
    bool m_everFlagged;

    F32 m_window[SPAN * CHANNELS];
    F32 m_weights[N_PARAMS];
    U32 m_stepsSince;
    U64 m_admittedTotal;
    U64 m_candFirst;
    U64 m_candLast;
    U64 m_spanFirst;       //!< first tick of the current candidate's training
    U32 m_candSteps;
    U32 m_candidates;
};

}  // namespace Sentinel

#endif
