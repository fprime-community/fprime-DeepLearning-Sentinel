// The whole per-tick pipeline: forecast, residual, EWMA, and three decision
// layers -- one that emits, and two that are computed and reported beside it.
//
// WHAT THIS UNIT IMPLEMENTS, as of the port registered at `docs/MODELS.md` 39:
//
//   1. THE FORECAST. Per channel, the mean of the up-to-ten predictions made at
//      t-1 .. t-10, the residual |actual - forecast|, and its EWMA at span 105.
//      Unchanged, and every committed `.vec` vector still pins it at 1e-5.
//
//   2. D25's STATIC QUANTILE, WHICH IS WHAT STILL EMITS. The maximum smoothed
//      residual across channels against one global threshold calibrated as the
//      99.9th percentile over the mission's own anomaly-masked nominal window.
//      No persistence filter beyond N=1, no error_buffer dilation, no k-of-n.
//      `crossing()` and `emitted()` are this rule and nothing else.
//
//   3. telemanom's DYNAMIC THRESHOLD (`DynamicThreshold`), which D62 made
//      mandatory for the port because it is the only rule measured to reach a
//      flyable alarm rate on a non-stationary regime (D48). It solves once per
//      70-tick segment over a trailing 2,170-sample window and reports through
//      `dynamicEmitted()`.
//
//   4. THE FIRST DERIVATIVE (`DerivativeStream`), trailing-standardised, which
//      fused with the residual reached **EVAL 17 of 19 where the frozen arm
//      reached 4** at the same alarm rate (D65). `fused()` and `fusedScore()`.
//
// (!) 3 AND 4 DO NOT DRIVE THE WARNING, AND THAT IS A DECISION RATHER THAN AN
// OVERSIGHT (39.8). **D65 re-opens the decision layer and names no flight
// configuration.** Promoting either to the emission would make this port the
// adoption decision, and it would move every committed golden vector, which 39.3
// said would not happen. The core computes all three; which one the component
// raises an event on is the component's to choose and the owner's to decide.
//
// **Warn-only.** The detector produces scores and flags. It commands nothing,
// writes nothing, and has no side effects (Objective.md 11 rule 3).
#ifndef SENTINEL_DETECTOR_HPP
#define SENTINEL_DETECTOR_HPP

#include "sentinel/Config.hpp"
#include "sentinel/DerivativeStream.hpp"
#include "sentinel/DynamicThreshold.hpp"
#include "sentinel/Ewma.hpp"
#include "sentinel/ModelFile.hpp"
#include "sentinel/Status.hpp"
#include "sentinel/Types.hpp"

namespace Sentinel {

class Detector {
  public:
    Detector();

    //! Load a model file. Until this returns OK the detector is inert: `step` does
    //! nothing and `emitted` is false, so a corrupt file can never produce a
    //! warning (D5, Objective.md 14.10).
    LoadStatus load(const U8* data, U32 length);

    //! Clear the GRU state, the prediction ring and the EWMA together, and
    //! restart the warm-up. This is what a restart looks like to the arithmetic.
    void reset();

    //! One rate-group tick. `values` is `nChannels` wide; `valid` is false when
    //! any watched channel was not measured this cycle, which scores -infinity --
    //! nothing measured is never an alarm (`detectors.py:394-399`).
    void step(const F32* values, bool valid);

    // -- what the tick produced -------------------------------------------
    F32 score() const { return m_score; }
    bool crossing() const { return m_crossing; }
    bool emitted() const { return m_emitted; }
    U64 steps() const { return m_steps; }

    const F32* hidden(U32 layer) const { return m_hidden[layer]; }
    const F32* headOutput() const { return m_head; }
    const F32* forecast() const { return m_forecast; }
    const F32* residual() const { return m_residual; }
    const F32* smoothed() const { return m_smoothed; }
    const Model& model() const { return m_model; }

    // -- the new streams: computed and reported, never in the flag (39.8) ---
    //
    // (!) `crossing()` and `emitted()` above are STILL D25's static quantile.
    // D62 requires the port to carry the dynamic threshold and it now does --
    // it runs every tick, is held to the reference at 1e-5 and its emission
    // flag is exact -- but WHICH rule drives the component's warning event is
    // an adoption decision, and D65 explicitly names no flight configuration.
    // Promoting it here would also move every committed `.vec` vector, which
    // `docs/MODELS.md` 39.3 said would not happen. The component chooses;
    // the core computes both.
    const DynamicThreshold& threshold() const { return m_threshold; }
    const DerivativeStream& derivative() const { return m_derivative; }

    //! True on the tick telemanom's rule decided a warning, at a segment's end.
    bool dynamicEmitted() const { return m_threshold.emitted(); }

    //! Arm 2's fused statistic for one channel, `max(z_residual, z_derivative)`
    //! -- the ablation, which is the arm that reached EVAL 17 of 19 (D65).
    F64 fused(U32 channel) const;

    //! The largest `fused` across channels this tick, and the channel that took
    //! it. A warning names a channel (Objective.md 11 rule 4).
    F64 fusedScore() const { return m_fusedScore; }
    U32 fusedChannel() const { return m_fusedChannel; }

  private:
    Model m_model;

    // -- carried state ----------------------------------------------------
    F32 m_hidden[Config::MAX_LAYERS][Config::MAX_HIDDEN];
    F32 m_ring[Config::MAX_PREDICTIONS][Config::MAX_OUTPUTS];
    Ewma m_ewma;
    DynamicThreshold m_threshold;
    DerivativeStream m_derivative;
    U64 m_steps;

    // -- per-tick scratch, reused every cycle -----------------------------
    F32 m_input[Config::MAX_INPUTS];
    F32 m_projected[Config::MAX_GATE_WIDTH];
    F32 m_recurrent[Config::MAX_GATE_WIDTH];
    F32 m_head[Config::MAX_OUTPUTS];
    F32 m_forecast[Config::MAX_CHANNELS];
    F32 m_residual[Config::MAX_CHANNELS];
    F32 m_smoothed[Config::MAX_CHANNELS];

    // -- the emission -----------------------------------------------------
    F32 m_score;
    bool m_crossing;
    bool m_emitted;
    F64 m_fusedScore;
    U32 m_fusedChannel;
};

}  // namespace Sentinel

#endif  // SENTINEL_DETECTOR_HPP
