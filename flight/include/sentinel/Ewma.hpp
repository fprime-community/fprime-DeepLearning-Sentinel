// The per-channel exponential moving average, span 105, as `telemanom.ewma` does it.
#ifndef SENTINEL_EWMA_HPP
#define SENTINEL_EWMA_HPP

#include "sentinel/Config.hpp"
#include "sentinel/Types.hpp"

namespace Sentinel {

//! Bias-corrected (pandas' `adjust=True`), computed in F64, emitted as F32.
//!
//! The reference (`telemanom.py:151-198`) opens `np.asarray(values,
//! dtype=np.float64)` and closes `.astype(np.float32)`. The precision is not
//! incidental: a decision layer run in F32 throughout would be a different
//! detector, so F64 here is a requirement of the flight target and not a
//! convenience (`docs/MODEL_FILE.md` 9).
//!
//! **The denominator is computed recursively rather than in closed form.** The
//! reference evaluates `(1 - decay^(t+1)) / alpha` from an absolute step counter.
//! In flight that counter grows without bound and every tick pays a `pow`. The
//! recursion `D[t] = 1 + decay * D[t-1]`, `D[-1] = 0`, is the same sequence --
//! `D[0] = (1 - decay)/alpha = 1`, `D[1] = 1 + decay` -- for one multiply and one
//! add, with no counter and no `pow`. It converges to `1/alpha`, which is 53.0 at
//! span 105. The deviation between the two forms is measured and reported by
//! `flight/test/GoldenVectors.cpp` rather than assumed to be zero.
class Ewma {
  public:
    Ewma();

    //! `alpha = 2 / (span + 1)`; span 105 gives 2/106.
    void configure(U32 span, U32 nChannels);
    void reset();

    //! One tick. `residual` and `out` are `nChannels` wide.
    void update(const F32* residual, F32* out) const;
    void advance(const F32* residual, F32* out);

    F64 decay() const { return m_decay; }
    F64 denominator() const { return m_denominator; }

  private:
    F64 m_numerator[Config::MAX_CHANNELS];
    F64 m_denominator;
    F64 m_decay;
    U32 m_channels;
};

}  // namespace Sentinel

#endif  // SENTINEL_EWMA_HPP
