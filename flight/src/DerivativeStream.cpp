#include "sentinel/DerivativeStream.hpp"

#include <cmath>

namespace Sentinel {

DerivativeStream::DerivativeStream()
    : m_window(), m_previous{}, m_delta{}, m_channels(0U), m_seenFirst(false) {}

void DerivativeStream::configure(U32 nChannels) {
    m_channels = (nChannels <= Config::MAX_CHANNELS) ? nChannels : 0U;
    m_window.configure(m_channels, Config::ERROR_WINDOW, Config::ERROR_WINDOW);
    reset();
}

void DerivativeStream::reset() {
    m_window.reset();
    for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
        m_previous[c] = 0.0F;
        m_delta[c] = 0.0F;
    }
    m_seenFirst = false;
}

void DerivativeStream::step(const F32* values) {
    if ((m_channels == 0U) || (values == nullptr)) {
        return;
    }

    for (U32 c = 0U; c < m_channels; ++c) {
        // `np.diff(x, prepend=x[0])`: the first difference is against the first
        // sample itself, so `dx[0]` is exactly zero rather than a step from
        // nothing. Getting this wrong puts a spurious excursion at every reset.
        const F32 previous = m_seenFirst ? m_previous[c] : values[c];
        const F32 difference = values[c] - previous;
        m_delta[c] = std::fabs(difference);
        m_previous[c] = values[c];
    }
    m_seenFirst = true;

    m_window.push(m_delta);
}

F64 DerivativeStream::delta(U32 channel) const {
    return (channel < m_channels) ? static_cast<F64>(m_delta[channel]) : 0.0;
}

F64 DerivativeStream::z(U32 channel) const {
    if (channel >= m_channels) {
        return 0.0;
    }
    return m_window.z(channel, static_cast<F64>(m_delta[channel]));
}

}  // namespace Sentinel
