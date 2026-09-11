// The first derivative of the raw value, standardised against a trailing window.
//
// D65's finding, and the reason this exists: fused with the smoothed residual it
// reaches **EVAL 17 of 19 where the frozen arm reaches 4**, at the frozen arm's
// own 0.6820%, a strict superset, reproduced identically on two independent
// reads. `docs/MODELS.md` 36.4 found the signal; 37.4 warned it had only been
// shown under a normalisation that looks at the whole test array; 38.15 reached
// it causally, which is what made it a result.
//
// The stream, from `scripts/decision_layer_arms.py:224`:
//
//     dx[t] = |x[t] - x[t-1]|,  with x[-1] := x[0] so dx[0] is exactly zero
//     z_d   = (dx[t] - mean) / max(sd, 1e-12)   over a trailing 2,100 window
//
// and the fused statistic is `max(z_residual, z_derivative)` -- the **ablation**,
// which is the arm that won. D65's P2.4 was refuted in the informative direction:
// dropping the horizon-disagreement stream is *better*, 30/38 against 25/38,
// because that stream spends alarm budget for less than it returns. There is no
// third stream here for that reason.
//
// (!) COMPUTED AND REPORTED, NEVER IN THE FLAG (39.8). `Detector::crossing` and
// `Detector::emitted` stay driven by the dynamic threshold alone. **D65 re-opens
// the decision layer and names no flight configuration**, so wiring the fused
// rule into the warning would make this port the adoption decision, which is not
// its to make. The cost is measured now; adopting it later is a one-line change
// and a D-entry.
//
// COST, and N7 registers it: one subtraction, one absolute value, two F64
// accumulator updates and one divide, per channel per tick. No second `z` sweep,
// no pruning pass, no dilation. **Cheaper than the threshold it rides beside in
// compute and equal to it in memory** -- both need a trailing window of the same
// depth, which is D65 consequence 4 as corrected.
#ifndef SENTINEL_DERIVATIVE_STREAM_HPP
#define SENTINEL_DERIVATIVE_STREAM_HPP

#include "sentinel/Config.hpp"
#include "sentinel/TrailingWindow.hpp"
#include "sentinel/Types.hpp"

namespace Sentinel {

class DerivativeStream {
  public:
    DerivativeStream();

    void configure(U32 nChannels);
    void reset();

    //! One tick of raw values, `nChannels` wide.
    void step(const F32* values);

    //! `|x[t] - x[t-1]|` for one channel, this tick.
    F64 delta(U32 channel) const;

    //! The trailing-standardised derivative, `z_d`.
    F64 z(U32 channel) const;

    U64 steps() const { return m_window.steps(); }
    U32 nChannels() const { return m_channels; }

    //! True once the standardising window is full. Below it the moments are of a
    //! shorter window, and a `z` taken from four samples is not evidence -- the
    //! component's own warm-up (`warmup_steps`, 2,350) outlasts this either way.
    bool warmed() const { return m_window.full(); }

  private:
    TrailingWindow m_window;
    F32 m_previous[Config::MAX_CHANNELS];
    F32 m_delta[Config::MAX_CHANNELS];
    U32 m_channels;
    bool m_seenFirst;
};

}  // namespace Sentinel

#endif  // SENTINEL_DERIVATIVE_STREAM_HPP
