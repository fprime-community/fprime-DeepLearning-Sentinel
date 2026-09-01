#include "sentinel/Ewma.hpp"

namespace Sentinel {

Ewma::Ewma() : m_numerator(), m_denominator(0.0), m_decay(0.0), m_channels(0U) {}

void Ewma::configure(U32 span, U32 nChannels) {
    const F64 alpha = 2.0 / (static_cast<F64>(span) + 1.0);
    m_decay = 1.0 - alpha;
    m_channels = nChannels;
    reset();
}

void Ewma::reset() {
    for (U32 i = 0U; i < Config::MAX_CHANNELS; ++i) {
        m_numerator[i] = 0.0;
    }
    m_denominator = 0.0;
}

void Ewma::update(const F32* residual, F32* out) const {
    // Const preview of the next tick, for callers that want the value without
    // moving the state. `advance` is the one that commits.
    const F64 denominator = 1.0 + (m_decay * m_denominator);
    for (U32 c = 0U; c < m_channels; ++c) {
        const F64 numerator = static_cast<F64>(residual[c]) + (m_decay * m_numerator[c]);
        out[c] = static_cast<F32>(numerator / denominator);
    }
}

void Ewma::advance(const F32* residual, F32* out) {
    m_denominator = 1.0 + (m_decay * m_denominator);
    for (U32 c = 0U; c < m_channels; ++c) {
        m_numerator[c] = static_cast<F64>(residual[c]) + (m_decay * m_numerator[c]);
        out[c] = static_cast<F32>(m_numerator[c] / m_denominator);
    }
}

}  // namespace Sentinel
