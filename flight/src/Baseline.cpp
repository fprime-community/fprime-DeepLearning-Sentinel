#include "sentinel/Baseline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Sentinel {

// `src/sentinel_models/baselines.py:31`.
const F64 Baseline::EPSILON = 1e-12;

namespace {

//! One bit per channel per ring slot. `Config::MAX_CHANNELS` is 16, so a U16
//! mask holds the whole window row; the shift is bounded by that.
inline U16 bitFor(U32 channel) {
    return static_cast<U16>(1U << channel);
}

}  // namespace

Baseline::Baseline()
    : m_ring(),
      m_present(),
      m_scale(),
      m_scores(),
      m_threshold(0.0),
      m_steps(0U),
      m_channels(0U),
      m_peak(0U),
      m_score(0.0),
      m_crossing(false),
      m_emitted(false) {
    // A scale of 1.0 rather than 0.0, so a baseline configured but never given
    // parameters divides by one instead of by EPSILON. `BASELINE_SCALE`'s FPP
    // default is the same value for the same reason (D34).
    for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
        m_scale[c] = 1.0;
    }
    reset();
}

void Baseline::configure(U32 nChannels) {
    // CPP-4: this is reachable from a model file, which is uplinked. Refuse by
    // going inert, never by asserting.
    m_channels = (nChannels <= Config::MAX_CHANNELS) ? nChannels : 0U;
    reset();
}

void Baseline::setScale(const F64* scale, U32 count) {
    if (scale == nullptr) {
        return;
    }
    const U32 limit = (count <= Config::MAX_CHANNELS) ? count : Config::MAX_CHANNELS;
    for (U32 c = 0U; c < limit; ++c) {
        m_scale[c] = scale[c];
    }
}

void Baseline::setThreshold(F64 threshold) {
    m_threshold = threshold;
}

void Baseline::reset() {
    // Cleared to the compile-time maxima, not to the configured width, so a
    // reconfiguration can never expose a stale sample from a wider instance.
    for (U32 slot = 0U; slot < Config::BASELINE_WINDOW; ++slot) {
        for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
            m_ring[slot][c] = 0.0F;
        }
        m_present[slot] = 0U;
    }
    for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
        m_scores[c] = 0.0;
    }
    m_steps = 0U;
    m_peak = 0U;
    m_score = -std::numeric_limits<F64>::infinity();
    m_crossing = false;
    m_emitted = false;
}

void Baseline::step(const F32* values, bool valid) {
    if ((m_channels == 0U) || (values == nullptr)) {
        m_crossing = false;
        m_emitted = false;
        return;
    }

    // -- write this tick into the ring -------------------------------------
    //
    // A non-finite sample is excluded from the sums as well as from the count.
    // `_rolling` sums `nan_to_num(values)` unmasked, so a NaN contributes 0 --
    // identical -- but an infinity would contribute +/-3.4e38 while not
    // incrementing the count. Bundle values are NaN where unobserved and never
    // infinite, so the two agree on every input the harness can produce; the
    // departure is recorded in `baseline_reference.py` and pinned by test.
    const U32 slot = static_cast<U32>(m_steps % static_cast<U64>(Config::BASELINE_WINDOW));
    U16 mask = 0U;
    for (U32 c = 0U; c < m_channels; ++c) {
        const bool finite = std::isfinite(values[c]);
        m_ring[slot][c] = finite ? values[c] : 0.0F;
        if (finite) {
            mask = static_cast<U16>(mask | bitFor(c));
        }
    }
    m_present[slot] = mask;
    ++m_steps;

    // -- recompute the window ----------------------------------------------
    //
    // Recomputed every tick rather than carried as a running sum. A running
    // add/subtract drifts without bound over a mission and a fixed recompute
    // does not, and at 120 x 16 F64 multiply-adds it costs 1,920 operations
    // against the model path's 70,080 -- 2.7%, and the model path is not
    // running when this one is. CPP-34: both loops are bounded by compile-time
    // constants the header has already validated.
    const U32 filled = (m_steps < static_cast<U64>(Config::BASELINE_WINDOW))
                           ? static_cast<U32>(m_steps)
                           : Config::BASELINE_WINDOW;

    for (U32 c = 0U; c < m_channels; ++c) {
        F64 sum = 0.0;
        F64 sumSquares = 0.0;
        F64 count = 0.0;
        const U16 bit = bitFor(c);
        for (U32 i = 0U; i < filled; ++i) {
            if ((m_present[i] & bit) != 0U) {
                const F64 sample = static_cast<F64>(m_ring[i][c]);
                sum += sample;
                sumSquares += sample * sample;
                count += 1.0;
            }
        }
        count = std::max(count, 1.0);   // `baselines.py:50`, the count floor
        const F64 mean = sum / count;
        const F64 second = sumSquares / count;
        // `baselines.py:55`, the variance floor. Cancellation cannot drive this
        // negative the way `_rolling`'s float32 prefix sums do (D37), but the
        // floor is part of the rule and is transcribed with it.
        const F64 variance = std::max(second - (mean * mean), 0.0);
        const F64 spread = std::sqrt(variance);
        const F64 divisor = (m_scale[c] > EPSILON) ? m_scale[c] : EPSILON;
        m_scores[c] = spread / divisor;
    }

    // -- reduce and compare -------------------------------------------------
    if (valid) {
        U32 peak = 0U;
        F64 largest = m_scores[0];
        for (U32 c = 1U; c < m_channels; ++c) {
            // Strict `>`, so a tie keeps the lowest index, as the model path does.
            if (m_scores[c] > largest) {
                largest = m_scores[c];
                peak = c;
            }
        }
        m_peak = peak;
        m_score = largest;
    } else {
        m_peak = 0U;
        m_score = -std::numeric_limits<F64>::infinity();
    }

    // `harness.py:169-172` compares in float64 and with `>=`, not `>`.
    m_crossing = (m_score >= m_threshold);
    m_emitted = m_crossing && warmed();
}

}  // namespace Sentinel
