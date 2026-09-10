// A trailing, right-inclusive window per channel, with its mean and standard
// deviation carried in F64 -- the primitive both new streams are built on.
//
// Two things need it and they need different halves of it. telemanom's dynamic
// threshold needs the window's **contents**: it re-solves `eps = mu + z*sigma`
// over 19 candidates, recomputes both moments with the candidate points removed,
// and finds and prunes exceedance runs inside the window, none of which a running
// moment can supply. The derivative stream needs only the **moments**, to
// standardise `|x[t] - x[t-1]|` against the same span
// (`scripts/decision_layer_arms.py:224`). One class serves both, and
// `docs/MODELS.md` 39.6 registers two instances of it as layout L1.
//
// (!) THE MOMENTS ARE CARRIED, NOT RECOMPUTED, AND THAT DEPARTS FROM `Baseline`.
// `Baseline.cpp` recomputes its 120-sample window every tick and says why: "a
// running add/subtract drifts without bound over a mission and a fixed recompute
// does not". At 120 x 16 that costs 1,920 operations; at 2,100 x 16 it would cost
// 33,600, twice over, against the model path's own 70,080 -- and
// `docs/MODELS.md` 39 registers the derivative at O(1) per tick (N7).
//
// So the drift was measured rather than assumed, on the regime D37's defect
// showed up in: **1,000,000 ticks, window 2,100, N(1000, 3) float32 samples,
// running add/subtract against an exact recompute of the same window.**
//
//     worst relative drift, mean   0.000e+00
//     worst relative drift, sd     4.246e-10
//
// The mean's is exactly zero because every sample leaves by the same subtraction
// that admitted it. The second moment carries a residue five orders below the
// 1e-5 contract N1 sets. **This is a measurement over 1e6 ticks and is not a
// proof for all time**: if a mission runs long enough to matter, the ring is
// already resident and an exact refresh costs one pass.
//
// (!) AND IT IS F64 FROM THE FIRST ACCUMULATION, WHICH IS THE WHOLE OF D37.
// `baselines.py:39-51` records the defect: float32 prefix sums differenced to
// recover a small second moment is catastrophic cancellation -- 7.6584e+00 of
// error against a true sigma of 3.0. Samples arrive as F32 and are promoted
// before they touch an accumulator, never after.
#ifndef SENTINEL_TRAILING_WINDOW_HPP
#define SENTINEL_TRAILING_WINDOW_HPP

#include "sentinel/Config.hpp"
#include "sentinel/Types.hpp"

namespace Sentinel {

class TrailingWindow {
  public:
    //! The standard-deviation floor, `scripts/decision_layer_arms.py:62`'s
    //! `np.maximum(sd, 1e-12)`. A dead channel scores zero, never infinity.
    static const F64 EPSILON;

    TrailingWindow();

    //! Set the channel width and clear. A width past `Config::MAX_CHANNELS`
    //! clamps to zero, which leaves the window inert rather than reading past an
    //! array -- the same refusal `Baseline::configure` makes.
    void configure(U32 nChannels);

    //! Clear the ring, the accumulators and the step count.
    void reset();

    //! One tick, `nChannels` wide. Non-finite samples are excluded from the
    //! accumulators and from the count, exactly as `Baseline` excludes them.
    void push(const F32* values);

    // -- the moments, O(1) --------------------------------------------------
    F64 mean(U32 channel) const;
    F64 sd(U32 channel) const;

    //! `(value - mean) / max(sd, EPSILON)`, which is `zstat`
    //! (`scripts/decision_layer_arms.py:60-62`) for one sample of one channel.
    F64 z(U32 channel, F64 value) const;

    // -- the contents, for the threshold ------------------------------------
    //! Sample `index` of `channel`'s window, **oldest first**: index 0 is the
    //! earliest sample still inside the window and `filled() - 1` is the newest.
    //! Time order, not ring order, because every rule that reads this cares
    //! which sample came first.
    F32 at(U32 channel, U32 index) const;

    //! Samples currently in the window, `<= Config::ERROR_WINDOW`.
    U32 filled() const { return m_filled; }

    //! Ticks since `reset`, which keeps counting past the window.
    U64 steps() const { return m_steps; }

    U32 nChannels() const { return m_channels; }

    //! True once the window has as many samples as it is wide. Below it the
    //! moments are of a shorter window and are honest about being so -- the
    //! reference does the same, `trailing_stats` dividing by `cnt` rather than
    //! by `span` (`scripts/decision_layer_arms.py:56`).
    bool full() const { return m_filled >= Config::ERROR_WINDOW; }

  private:
    //! Channel-major, unlike `Baseline`'s slot-major ring. The threshold walks
    //! one channel's whole window 19 times per re-solve and never walks a slot
    //! across channels, so this is the order those reads want.
    F32 m_ring[Config::MAX_CHANNELS][Config::ERROR_WINDOW];

    F64 m_sum[Config::MAX_CHANNELS];
    F64 m_sumSquares[Config::MAX_CHANNELS];
    U32 m_count[Config::MAX_CHANNELS];

    U64 m_steps;
    U32 m_channels;
    U32 m_filled;
    U32 m_head;      //!< ring slot the next sample goes into
};

}  // namespace Sentinel

#endif  // SENTINEL_TRAILING_WINDOW_HPP
