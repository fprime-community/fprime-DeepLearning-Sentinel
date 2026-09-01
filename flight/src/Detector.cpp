#include "sentinel/Detector.hpp"

#include <cmath>
#include <limits>

#include "sentinel/Gru.hpp"

namespace Sentinel {

Detector::Detector()
    : m_model(), m_hidden(), m_ring(), m_ewma(), m_steps(0U), m_input(),
      m_projected(), m_recurrent(), m_head(), m_forecast(), m_residual(),
      m_smoothed(), m_score(0.0F), m_crossing(false), m_emitted(false) {}

LoadStatus Detector::load(const U8* data, U32 length) {
    const LoadStatus status = m_model.load(data, length);
    if (status != LoadStatus::OK) {
        return status;
    }
    m_ewma.configure(m_model.ewmaSpan, m_model.nChannels);
    reset();
    return LoadStatus::OK;
}

void Detector::reset() {
    for (U32 layer = 0U; layer < Config::MAX_LAYERS; ++layer) {
        for (U32 i = 0U; i < Config::MAX_HIDDEN; ++i) {
            m_hidden[layer][i] = 0.0F;
        }
    }
    for (U32 slot = 0U; slot < Config::MAX_PREDICTIONS; ++slot) {
        for (U32 i = 0U; i < Config::MAX_OUTPUTS; ++i) {
            m_ring[slot][i] = 0.0F;
        }
    }
    m_ewma.reset();
    m_steps = 0U;
    m_score = 0.0F;
    m_crossing = false;
    m_emitted = false;
}

void Detector::step(const F32* values, bool valid) {
    if (!m_model.loaded() || (values == nullptr)) {
        m_crossing = false;
        m_emitted = false;
        return;
    }

    const U32 channels = m_model.nChannels;
    const U32 predictions = m_model.nPredictions;

    // Normalisation is identity under D2, and the slot exists because a mission
    // whose data is not ESA-preprocessed will need it.
    for (U32 c = 0U; c < channels; ++c) {
        m_input[c] = (values[c] - m_model.normOffset[c]) * m_model.normScale[c];
    }
    for (U32 c = channels; c < m_model.nInputs; ++c) {
        m_input[c] = 0.0F;                     // n_exogenous is 0 until D6
    }

    // -- the forward pass, one tick, state carried ------------------------
    for (U32 layer = 0U; layer < m_model.nLayers; ++layer) {
        const F32* source = (layer == 0U) ? m_input : m_hidden[layer - 1U];
        Gru::step(m_model, layer, source, m_hidden[layer], m_projected, m_recurrent);
    }
    Gru::head(m_model, m_hidden[m_model.nLayers - 1U], m_head);

    // -- the forecast: the mean of the up-to-l_p predictions made at t-1..t-l_p
    //
    // Summed with j ascending and accumulated in F64, because
    // `windows.aggregate_predictions` does both. A running accumulator would
    // receive the same terms in the opposite order and round differently, which
    // is why the last l_p head outputs are held instead -- 4,800 bytes at the
    // flown shape to stay faithful to the reference.
    const U32 count = (m_steps < static_cast<U64>(predictions))
                          ? static_cast<U32>(m_steps) : predictions;
    const F64 divisor = (count == 0U) ? 1.0 : static_cast<F64>(count);
    for (U32 c = 0U; c < channels; ++c) {
        F64 total = 0.0;
        for (U32 j = 0U; j < count; ++j) {
            const U64 source = m_steps - 1U - static_cast<U64>(j);
            const U32 slot = static_cast<U32>(source % static_cast<U64>(predictions));
            total += static_cast<F64>(m_ring[slot][(j * channels) + c]);
        }
        m_forecast[c] = static_cast<F32>(total / divisor);
    }

    for (U32 c = 0U; c < channels; ++c) {
        m_residual[c] = std::fabs(m_input[c] - m_forecast[c]);
    }

    m_ewma.advance(m_residual, m_smoothed);

    // -- reduce and compare ------------------------------------------------
    if (!valid) {
        // Nothing measured is never an alarm (`detectors.py:394-399`).
        m_score = -std::numeric_limits<F32>::infinity();
    } else {
        F32 largest = m_smoothed[0];
        for (U32 c = 1U; c < channels; ++c) {
            if (m_smoothed[c] > largest) {
                largest = m_smoothed[c];
            }
        }
        m_score = largest;
    }

    // `harness.py:169-172` compares in float64, and with `>=` rather than `>`.
    m_crossing = (static_cast<F64>(m_score) >= m_model.threshold);

    // Silent until validated (Objective.md 11 rule 2). A reset restarts this.
    const bool warmed = (m_steps >= static_cast<U64>(m_model.warmupSteps));
    m_emitted = m_crossing && warmed && (m_model.baselineOnly == 0U);

    // The head output made at t covers t+1 .. t+l_p, so it is written after the
    // forecast is read -- its slot is the one holding t-l_p, which has just been
    // consumed for the last time.
    const U32 slot = static_cast<U32>(m_steps % static_cast<U64>(predictions));
    for (U32 i = 0U; i < m_model.nOutputs; ++i) {
        m_ring[slot][i] = m_head[i];
    }
    ++m_steps;
}

}  // namespace Sentinel
