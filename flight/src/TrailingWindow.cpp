#include "sentinel/TrailingWindow.hpp"

#include <cmath>

namespace Sentinel {

const F64 TrailingWindow::EPSILON = 1e-12;

TrailingWindow::TrailingWindow()
    : m_ring{}, m_sum{}, m_sumSquares{}, m_count{},
      m_steps(0U), m_channels(0U), m_filled(0U), m_head(0U) {}

void TrailingWindow::configure(U32 nChannels) {
    // Refuse by going inert, never by reading past an array. `ModelFile` has
    // already refused an oversized model with TOO_LARGE before this is reached;
    // this is the second line of the same defence.
    m_channels = (nChannels <= Config::MAX_CHANNELS) ? nChannels : 0U;
    reset();
}

void TrailingWindow::reset() {
    for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
        m_sum[c] = 0.0;
        m_sumSquares[c] = 0.0;
        m_count[c] = 0U;
        for (U32 i = 0U; i < Config::ERROR_WINDOW; ++i) {
            m_ring[c][i] = 0.0F;
        }
    }
    m_steps = 0U;
    m_filled = 0U;
    m_head = 0U;
}

void TrailingWindow::push(const F32* values) {
    if ((m_channels == 0U) || (values == nullptr)) {
        return;
    }

    const bool wrapped = (m_filled >= Config::ERROR_WINDOW);

    for (U32 c = 0U; c < m_channels; ++c) {
        // Remove the sample this slot is about to overwrite. It leaves by the
        // same subtraction that admitted it, which is why the mean's drift
        // measured exactly zero over 1e6 ticks (see the header).
        if (wrapped) {
            const F64 leaving = static_cast<F64>(m_ring[c][m_head]);
            if (std::isfinite(leaving) && (m_count[c] > 0U)) {
                m_sum[c] -= leaving;
                m_sumSquares[c] -= leaving * leaving;
                --m_count[c];
            }
        }

        // F64 BEFORE the accumulator, never after. D37.
        const F32 sample = values[c];
        const bool finite = std::isfinite(sample);
        m_ring[c][m_head] = finite ? sample : 0.0F;
        if (finite) {
            const F64 promoted = static_cast<F64>(sample);
            m_sum[c] += promoted;
            m_sumSquares[c] += promoted * promoted;
            ++m_count[c];
        }
    }

    m_head = (m_head + 1U) % Config::ERROR_WINDOW;
    if (!wrapped) {
        ++m_filled;
    }
    ++m_steps;
}

F64 TrailingWindow::mean(U32 channel) const {
    if ((channel >= m_channels) || (m_count[channel] == 0U)) {
        return 0.0;
    }
    return m_sum[channel] / static_cast<F64>(m_count[channel]);
}

F64 TrailingWindow::sd(U32 channel) const {
    if ((channel >= m_channels) || (m_count[channel] == 0U)) {
        return 0.0;
    }
    const F64 n = static_cast<F64>(m_count[channel]);
    const F64 mu = m_sum[channel] / n;
    const F64 second = m_sumSquares[channel] / n;
    // The variance floor, `scripts/decision_layer_arms.py:58`'s
    // `np.maximum(s2/cnt - mu*mu, 0.0)`. Population sd, not sample: the
    // reference divides by `cnt`, and so does `numpy.nanstd`.
    const F64 variance = (second - (mu * mu) > 0.0) ? (second - (mu * mu)) : 0.0;
    return std::sqrt(variance);
}

F64 TrailingWindow::z(U32 channel, F64 value) const {
    const F64 spread = sd(channel);
    const F64 divisor = (spread > EPSILON) ? spread : EPSILON;
    return (value - mean(channel)) / divisor;
}

F32 TrailingWindow::at(U32 channel, U32 index) const {
    if ((channel >= m_channels) || (index >= m_filled)) {
        return 0.0F;
    }
    // Oldest first. `m_head` is where the next sample goes, so it is also the
    // oldest slot once the ring has wrapped; before that the oldest is slot 0.
    const U32 oldest = (m_filled >= Config::ERROR_WINDOW) ? m_head : 0U;
    return m_ring[channel][(oldest + index) % Config::ERROR_WINDOW];
}

}  // namespace Sentinel
