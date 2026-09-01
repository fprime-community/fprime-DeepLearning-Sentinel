// The whole per-tick pipeline: forecast, residual, EWMA, max, one compare.
//
// The decision layer is frozen (D25) and this is a transcription of it, not a
// design: per channel the EWMA of |actual - forecast|, the maximum across
// channels, and one global threshold calibrated as the 99.9th percentile of that
// statistic over the mission's own anomaly-masked nominal window. No persistence
// filter beyond N=1, no error_buffer dilation, no k-of-n.
//
// **Warn-only.** The detector produces a score and a flag. It commands nothing,
// writes nothing, and has no side effects (Objective.md 11 rule 3).
#ifndef SENTINEL_DETECTOR_HPP
#define SENTINEL_DETECTOR_HPP

#include "sentinel/Config.hpp"
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

  private:
    Model m_model;

    // -- carried state ----------------------------------------------------
    F32 m_hidden[Config::MAX_LAYERS][Config::MAX_HIDDEN];
    F32 m_ring[Config::MAX_PREDICTIONS][Config::MAX_OUTPUTS];
    Ewma m_ewma;
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
};

}  // namespace Sentinel

#endif  // SENTINEL_DETECTOR_HPP
