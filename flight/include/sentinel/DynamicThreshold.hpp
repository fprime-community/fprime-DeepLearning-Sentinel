// telemanom's nonparametric dynamic threshold, streamed.
//
// D62 made this mandatory for the port -- it is the only rule measured to reach a
// flyable alarm rate on a non-stationary regime (D48), where D25's static
// quantile cannot be brought to one at any multiplier. `docs/MODELS.md` 39
// registers the transcription; every constant is in `Config.hpp` beside its
// `src/sentinel_models/telemanom.py` line.
//
// THE RULE, and it is solved once per segment rather than once per tick:
//
//   every STRIDE ticks, over the trailing SOLVE_WINDOW samples of one channel,
//     mu, sigma          the window's moments
//     for each z in 2.5 .. 11.5 step 0.5, skipping any z above the window's
//       own reach (max - mu)/sigma, because a candidate above the maximum
//       exceeds nothing:
//         eps = mu + z*sigma
//         score = (dmu/mu + dsigma/sigma) / (|sequences|^2 + covered)
//     keep the best-scoring eps, ties to the larger z -- the quieter choice
//     if nothing qualifies, eps = mu + 12*sigma and NOTHING is reported.
//     Silence, not a guess.
//
// (!) TWO DEPARTURES FROM THE REFERENCE, BOTH REGISTERED AND NEITHER SILENT.
//
// 1. DILATION IS FORWARD ONLY (39.5). `telemanom._buffered` widens every
//    exceedance run by ERROR_BUFFER - 1 = 99 steps on BOTH sides. The backward
//    half marks timesteps that have already been emitted, and a warn-only
//    component cannot go back and say something about a tick it has already
//    passed in silence. There is no un-emit. The price of dropping it is
//    registered as N6 and is NOT YET MEASURED -- it needs one bucket read.
//
// 2. THE PER-TICK RATIO LAGS BY ONE SEGMENT, AND THE EMISSION DOES NOT.
//    `channel_ratios` divides a segment's errors by the eps solved from a window
//    that includes that segment; in flight the segment's ticks are already past
//    when its eps exists. So `ratio()` uses the eps held from the previous solve
//    and is a reported score, while `emitted()` fires at the segment's end with
//    the eps actually solved for it -- which is the reference's OWN flight model,
//    `telemanom.py:420-431`: "a flight component cannot emit retroactively; it
//    emits when the batch containing the crossing is processed, at the end of the
//    segment". `last_emission` is what the harness measures lead time from, so
//    this is the quantity 37.7a's 0 of 10 positive leads was measured on.
//
// Warn-only. A score and a flag; it commands nothing (Objective.md 11 rule 3).
#ifndef SENTINEL_DYNAMIC_THRESHOLD_HPP
#define SENTINEL_DYNAMIC_THRESHOLD_HPP

#include "sentinel/Config.hpp"
#include "sentinel/TrailingWindow.hpp"
#include "sentinel/Types.hpp"

namespace Sentinel {

class DynamicThreshold {
  public:
    //! A half-open run of samples inside the solve window, oldest-first indices.
    struct Sequence {
        U16 lo;
        U16 hi;
    };

    DynamicThreshold();

    //! Set the channel width and clear. Refuses an oversized width by going
    //! inert, as every unit in this core does.
    void configure(U32 nChannels);

    void reset();

    //! One tick of smoothed error, `nChannels` wide. Solves at the end of every
    //! `Config::STRIDE`-th tick and is O(1) otherwise.
    void step(const F32* smoothed);

    //! True on the tick a segment's solve decided a warning. Not sticky: it is
    //! the emission instant, not the alarm's extent.
    bool emitted() const { return m_emitted; }

    //! The channel that emitted, valid on a tick where `emitted()` is true.
    U32 peakChannel() const { return m_peak; }

    //! `e_s[t] / eps` for one channel, with the eps held from the last solve.
    //! Below 1.0 is quiet. Zero before the first solve.
    F64 ratio(U32 channel) const;

    //! The largest `ratio` across channels this tick.
    F64 score() const { return m_score; }

    //! The threshold last solved for a channel.
    F64 epsilon(U32 channel) const;

    //! `z_residual`: the smoothed error standardised against its own trailing
    //! 2,100 window, which is arm 2's first stream
    //! (`scripts/decision_layer_arms.py:224`). Reported, never in the flag.
    F64 zResidual(U32 channel) const;

    //! Solves completed since `reset`.
    U32 solves() const { return m_solves; }

    U64 steps() const { return m_window.steps(); }
    U32 nChannels() const { return m_channels; }

    //! Ticks into the current segment, `0 .. Config::STRIDE - 1`.
    U32 segmentPosition() const { return m_sinceSolve; }

  private:
    //! Solve one channel over the current window. Returns the chosen eps and
    //! fills `m_seq`/`m_kept` with its surviving sequences.
    F64 solve(U32 channel, U32 filled, U32& sequenceCount);

    //! Exceedance runs at `eps`, dilated FORWARD only and merged.
    U32 runsAt(U32 channel, U32 filled, F64 eps);

    //! telemanom's pruning ladder. Marks `m_kept` for `count` sequences.
    void pruneLadder(U32 channel, U32 filled, F64 eps, U32 count);

    TrailingWindow m_window;

    F64 m_eps[Config::MAX_CHANNELS];
    F32 m_latest[Config::MAX_CHANNELS];

    // Scratch, shared across channels because channels are solved one at a time.
    // Sizing it per channel would cost MAX_CHANNELS times as much for no benefit.
    Sequence m_seq[Config::MAX_SEQUENCES];
    bool m_kept[Config::MAX_SEQUENCES];
    F32 m_peaks[Config::MAX_SEQUENCES];
    U16 m_order[Config::MAX_SEQUENCES];

    U32 m_channels;
    U32 m_sinceSolve;
    U32 m_solves;
    U32 m_peak;
    F64 m_score;
    bool m_emitted;
};

}  // namespace Sentinel

#endif  // SENTINEL_DYNAMIC_THRESHOLD_HPP
