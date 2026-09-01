// Level 1: the statistical baseline the component runs when the model cannot be.
//
// Objective.md 14.10 and D5 make this mandatory and say why: it is not primarily
// about how much data a mission has, it is the loader's safe failure mode. A
// corrupt file, a version mismatch, a failed CRC or a radiation bit-flip must
// degrade to this with an event and an active-tier telemetry channel, and never
// fail the topology.
//
// The rule is `rstd`, the floor every Phase 1 result was measured against. Per
// channel, over a trailing right-inclusive window of `Config::BASELINE_WINDOW`
// samples including the current one:
//
//     n      = max(count of finite samples in the window, 1)
//     spread = sqrt(max(S2/n - (S1/n)^2, 0))
//     score  = spread / max(scale[c], EPSILON)
//
// then the maximum across channels and one alarm when it is `>=` the threshold.
//
// (!) THIS IS TRANSCRIBED FROM THE RULE, NOT FROM `baselines.py`, AND THAT IS
// DELIBERATE. `baselines._rolling` accumulates its prefix sums in float32 and
// differences them to recover a second moment, which is catastrophic
// cancellation: measured 7.6584e+00 of error on a true sigma of 3.0, and 2,852
// spurious exact zeros on this project's own fixture. This core matches
// `src/sentinel_models/baseline_reference.py`, which agrees with `numpy.nanstd`
// to 3.5e-10. See `docs/DECISIONS.md` D37 and `docs/MODELS.md` 20.6. The harness
// repair is `docs/MODELS.md` 21 and has not run yet, so the two are knowingly
// different numbers today.
//
// Warn-only, like the rest of the core: a score and a flag, no side effects.
#ifndef SENTINEL_BASELINE_HPP
#define SENTINEL_BASELINE_HPP

#include "sentinel/Config.hpp"
#include "sentinel/Types.hpp"

namespace Sentinel {

class Baseline {
  public:
    //! The scale divisor's floor, `src/sentinel_models/baselines.py:31`.
    static const F64 EPSILON;

    Baseline();

    //! Set the channel width and clear everything. Refuses a width past
    //! `Config::MAX_CHANNELS` by clamping to zero, which leaves the baseline
    //! inert rather than reading past an array.
    void configure(U32 nChannels);

    //! The `BASELINE_SCALE` parameter (D34). Copied, not aliased: the component
    //! owns its parameters and this class owns its arithmetic.
    void setScale(const F64* scale, U32 count);

    //! The `BASELINE_THRESHOLD` parameter (D34). F64, and compared as F64.
    void setThreshold(F64 threshold);

    //! Clear the ring and restart the warm-up.
    void reset();

    //! One rate-group tick. `values` is `nChannels` wide. `valid` false scores
    //! negative infinity -- nothing measured is never an alarm, the same rule
    //! `Detector::step` applies on the model path.
    void step(const F32* values, bool valid);

    // -- what the tick produced -------------------------------------------
    F64 score() const { return m_score; }
    bool crossing() const { return m_crossing; }
    bool emitted() const { return m_emitted; }
    U64 steps() const { return m_steps; }
    U32 nChannels() const { return m_channels; }

    //! The channel that took the maximum. Ties keep the lowest index, which is
    //! what the strict `>` scan gives and what `Detector.cpp:100-106` does.
    //! This is what the warning event names.
    U32 peakChannel() const { return m_peak; }

    //! Per-channel scores, `nChannels` wide, valid after `step`.
    const F64* channelScores() const { return m_scores; }

    //! Warm-up is the window, `baselines.py:96-98` -- 120 ticks, not the model
    //! path's 2,350. A degraded component therefore starts emitting 2,230 ticks
    //! earlier than a healthy one, which is a real operational consequence and
    //! is pre-registered as prediction C10.
    //!
    //! Strictly greater, to match the rest of the core rather than by accident.
    //! `Detector.cpp:113` tests `m_steps >= warmupSteps` against the count of
    //! *previous* steps, because its increment comes after the check; this class
    //! increments before, so the equivalent test is `>`. Both then agree with
    //! `harness.py:160-167`, which prefixes the scored window with `warmup_steps`
    //! samples and drops exactly that many outputs, so the first surviving score
    //! is the one produced by call `window + 1`.
    bool warmed() const { return m_steps > static_cast<U64>(Config::BASELINE_WINDOW); }

  private:
    // The ring. `m_present` is one bit per channel per slot, so a 16-channel
    // window costs 240 bytes rather than 1,920.
    F32 m_ring[Config::BASELINE_WINDOW][Config::MAX_CHANNELS];
    U16 m_present[Config::BASELINE_WINDOW];

    F64 m_scale[Config::MAX_CHANNELS];
    F64 m_scores[Config::MAX_CHANNELS];
    F64 m_threshold;

    U64 m_steps;
    U32 m_channels;
    U32 m_peak;
    F64 m_score;
    bool m_crossing;
    bool m_emitted;
};

}  // namespace Sentinel

#endif  // SENTINEL_BASELINE_HPP
