#include "sentinel/DynamicThreshold.hpp"

#include <cmath>

namespace Sentinel {

namespace {

//! Population mean and standard deviation of a channel's window, F64 throughout.
void moments(const TrailingWindow& window, U32 channel, U32 filled,
             F64& mu, F64& sigma, F64& maximum) {
    F64 sum = 0.0;
    F64 sumSquares = 0.0;
    maximum = -1.0e308;
    for (U32 i = 0U; i < filled; ++i) {
        const F64 sample = static_cast<F64>(window.at(channel, i));
        sum += sample;
        sumSquares += sample * sample;
        if (sample > maximum) {
            maximum = sample;
        }
    }
    const F64 n = (filled > 0U) ? static_cast<F64>(filled) : 1.0;
    mu = sum / n;
    const F64 variance = (sumSquares / n) - (mu * mu);
    sigma = std::sqrt((variance > 0.0) ? variance : 0.0);
}

}  // namespace

DynamicThreshold::DynamicThreshold()
    : m_window(), m_eps{}, m_latest{}, m_seq{}, m_kept{}, m_peaks{}, m_order{},
      m_channels(0U), m_sinceSolve(0U), m_solves(0U), m_peak(0U),
      m_score(0.0), m_emitted(false) {}

void DynamicThreshold::configure(U32 nChannels) {
    m_channels = (nChannels <= Config::MAX_CHANNELS) ? nChannels : 0U;
    // 2,170 of CONTENTS for the sweep, 2,100 of MOMENTS for z_residual: the
    // threshold judges a window that includes the segment, arm 2 standardises
    // over `span = error_window` (`scripts/decision_layer_arms.py:224`).
    m_window.configure(m_channels, Config::SOLVE_WINDOW, Config::ERROR_WINDOW);
    reset();
}

void DynamicThreshold::reset() {
    m_window.reset();
    for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
        m_eps[c] = 0.0;
        m_latest[c] = 0.0F;
    }
    m_sinceSolve = 0U;
    m_solves = 0U;
    m_peak = 0U;
    m_score = 0.0;
    m_emitted = false;
}

U32 DynamicThreshold::runsAt(U32 channel, U32 filled, F64 eps) {
    // `telemanom._runs` then `_buffered`, with the backward pad removed (39.5).
    // Merging is the same: a run whose dilated start touches the previous run's
    // dilated end extends it rather than starting a new one.
    const U32 pad = Config::ERROR_BUFFER - 1U;
    U32 count = 0U;
    U32 i = 0U;
    while (i < filled) {
        if (static_cast<F64>(m_window.at(channel, i)) >= eps) {
            U32 lo = i;
            while ((i < filled)
                   && (static_cast<F64>(m_window.at(channel, i)) >= eps)) {
                ++i;
            }
            U32 hi = i + pad;                     // FORWARD ONLY. `lo` is not moved.
            if (hi > filled) {
                hi = filled;
            }
            if ((count > 0U) && (lo <= m_seq[count - 1U].hi)) {
                if (hi > m_seq[count - 1U].hi) {
                    m_seq[count - 1U].hi = static_cast<U16>(hi);
                }
            } else if (count < Config::MAX_SEQUENCES) {
                m_seq[count].lo = static_cast<U16>(lo);
                m_seq[count].hi = static_cast<U16>(hi);
                ++count;
            } else {
                break;                            // CPP-34: bounded, and it says so
            }
        } else {
            ++i;
        }
    }

    // `_buffered` drops any run of one sample or fewer after dilation.
    U32 kept = 0U;
    for (U32 s = 0U; s < count; ++s) {
        if (static_cast<U32>(m_seq[s].hi) > (static_cast<U32>(m_seq[s].lo) + 1U)) {
            m_seq[kept] = m_seq[s];
            ++kept;
        }
    }
    return kept;
}

void DynamicThreshold::pruneLadder(U32 channel, U32 filled, F64 eps, U32 count) {
    if (count == 0U) {
        return;
    }
    for (U32 s = 0U; s < count; ++s) {
        F32 peak = -3.4e38F;
        for (U32 i = m_seq[s].lo; i < m_seq[s].hi; ++i) {
            const F32 sample = m_window.at(channel, i);
            if (sample > peak) {
                peak = sample;
            }
        }
        m_peaks[s] = peak;
        m_order[s] = static_cast<U16>(s);
    }

    // Descending by peak. Insertion sort: `count` is small in every realistic
    // window and the sort is not on the per-tick path. Stable, so ties keep the
    // order `numpy.argsort` would give before its reversal.
    for (U32 a = 1U; a < count; ++a) {
        const U16 key = m_order[a];
        const F32 value = m_peaks[key];
        U32 b = a;
        while ((b > 0U) && (m_peaks[m_order[b - 1U]] < value)) {
            m_order[b] = m_order[b - 1U];
            --b;
        }
        m_order[b] = key;
    }

    // The largest error that stayed BELOW the threshold closes the ladder.
    F64 normalMax = 0.0;
    for (U32 i = 0U; i < filled; ++i) {
        const F64 sample = static_cast<F64>(m_window.at(channel, i));
        if ((sample < eps) && (sample > normalMax)) {
            normalMax = sample;
        }
    }

    for (U32 s = 0U; s < count; ++s) {
        m_kept[s] = true;
    }

    // Walk down the ladder. A relative drop below PRUNING_P says everything
    // above it is not separated from what lies beneath, so it is marked; a drop
    // at or above it clears the marks and the walk starts again. What survives
    // is whatever follows the last qualifying drop.
    U32 marked[Config::MAX_SEQUENCES];
    U32 markedCount = 0U;
    for (U32 i = 0U; i < count; ++i) {
        const F64 here = static_cast<F64>(m_peaks[m_order[i]]);
        const F64 next = (i + 1U < count)
                             ? static_cast<F64>(m_peaks[m_order[i + 1U]])
                             : normalMax;
        const F64 relative = (here != 0.0) ? ((here - next) / here) : 0.0;
        if (relative < static_cast<F64>(Config::PRUNING_P)) {
            marked[markedCount] = m_order[i];
            ++markedCount;
        } else {
            markedCount = 0U;
        }
    }
    for (U32 i = 0U; i < markedCount; ++i) {
        m_kept[marked[i]] = false;
    }
}

F64 DynamicThreshold::solve(U32 channel, U32 filled, U32& sequenceCount) {
    sequenceCount = 0U;

    F64 mu = 0.0;
    F64 sigma = 0.0;
    F64 maximum = 0.0;
    moments(m_window, channel, filled, mu, sigma, maximum);

    const F64 ceiling = mu + (static_cast<F64>(Config::Z_CEILING) * sigma);
    if (!std::isfinite(mu) || !std::isfinite(sigma) || (sigma == 0.0)) {
        return ceiling;                            // silence, not a guess
    }

    const F64 reach = (maximum - mu) / sigma;
    F64 bestScore = -1.0e308;
    F64 bestEps = ceiling;
    U32 bestCount = 0U;
    Sequence best[Config::MAX_SEQUENCES];
    bool bestKept[Config::MAX_SEQUENCES];

    for (U32 k = 0U; k < Config::Z_CANDIDATES; ++k) {
        const F64 z = static_cast<F64>(Config::Z_FLOOR)
                    + (static_cast<F64>(k) * static_cast<F64>(Config::Z_STEP));
        if (z > reach) {
            continue;                              // exceeds nothing, by construction
        }
        const F64 eps = mu + (z * sigma);

        // The remainder's moments, and whether anything cleared the bar at all.
        F64 sum = 0.0;
        F64 sumSquares = 0.0;
        U32 remaining = 0U;
        U32 above = 0U;
        for (U32 i = 0U; i < filled; ++i) {
            const F64 sample = static_cast<F64>(m_window.at(channel, i));
            if (sample >= eps) {
                ++above;
            } else {
                sum += sample;
                sumSquares += sample * sample;
                ++remaining;
            }
        }
        if ((above == 0U) || (remaining == 0U)) {
            continue;
        }

        const U32 count = runsAt(channel, filled, eps);
        if (count == 0U) {
            continue;
        }

        const F64 n = static_cast<F64>(remaining);
        const F64 remainderMu = sum / n;
        const F64 remainderVar = (sumSquares / n) - (remainderMu * remainderMu);
        const F64 remainderSigma = std::sqrt((remainderVar > 0.0) ? remainderVar : 0.0);

        const F64 dMu = (mu != 0.0) ? ((mu - remainderMu) / mu) : 0.0;
        const F64 dSigma = (sigma - remainderSigma) / sigma;
        F64 covered = 0.0;
        for (U32 s = 0U; s < count; ++s) {
            covered += static_cast<F64>(m_seq[s].hi) - static_cast<F64>(m_seq[s].lo);
        }
        const F64 sequences = static_cast<F64>(count);
        const F64 score = (dMu + dSigma) / ((sequences * sequences) + covered);

        // `>=`, not `>`: ties go to the larger z, which is the quieter choice.
        if (score >= bestScore) {
            pruneLadder(channel, filled, eps, count);
            bestScore = score;
            bestEps = eps;
            bestCount = count;
            for (U32 s = 0U; s < count; ++s) {
                best[s] = m_seq[s];
                bestKept[s] = m_kept[s];
            }
        }
    }

    for (U32 s = 0U; s < bestCount; ++s) {
        m_seq[s] = best[s];
        m_kept[s] = bestKept[s];
    }
    sequenceCount = bestCount;
    return bestEps;
}

void DynamicThreshold::step(const F32* smoothed) {
    m_emitted = false;
    if ((m_channels == 0U) || (smoothed == nullptr)) {
        return;
    }

    for (U32 c = 0U; c < m_channels; ++c) {
        m_latest[c] = smoothed[c];
    }
    m_window.push(smoothed);

    // The reported score uses the eps held from the last solve; see the header.
    F64 best = 0.0;
    U32 peak = 0U;
    for (U32 c = 0U; c < m_channels; ++c) {
        const F64 r = ratio(c);
        if (r > best) {
            best = r;
            peak = c;
        }
    }
    m_score = best;
    m_peak = peak;

    ++m_sinceSolve;
    if (m_sinceSolve < Config::STRIDE) {
        return;
    }
    m_sinceSolve = 0U;

    // -- segment end: solve, and decide whether this tick says anything --------
    const U32 filled = m_window.filled();
    const U32 offset = (filled > Config::STRIDE) ? (filled - Config::STRIDE) : 0U;

    bool emitted = false;
    U32 emitter = 0U;
    for (U32 c = 0U; c < m_channels; ++c) {
        U32 count = 0U;
        const F64 eps = solve(c, filled, count);
        m_eps[c] = eps;

        // `telemanom.py:420-431`: a surviving sequence must intersect the segment
        // being judged, AND the segment must itself carry a crossing. The second
        // is what stops a sequence dilated forward from an earlier crossing from
        // emitting again in a segment that is quiet.
        bool crossingInSegment = false;
        for (U32 i = offset; i < filled; ++i) {
            if (static_cast<F64>(m_window.at(c, i)) >= eps) {
                crossingInSegment = true;
                break;
            }
        }
        if (!crossingInSegment) {
            continue;
        }
        for (U32 s = 0U; s < count; ++s) {
            const U32 lo = (m_seq[s].lo > offset) ? m_seq[s].lo : offset;
            const U32 hi = (m_seq[s].hi < filled) ? m_seq[s].hi : filled;
            if (m_kept[s] && (hi > lo)) {
                if (!emitted) {
                    emitter = c;
                }
                emitted = true;
                break;
            }
        }
    }

    ++m_solves;
    m_emitted = emitted;
    if (emitted) {
        m_peak = emitter;
    }
}

F64 DynamicThreshold::ratio(U32 channel) const {
    if ((channel >= m_channels) || (m_eps[channel] <= 0.0)) {
        return 0.0;
    }
    return static_cast<F64>(m_latest[channel]) / m_eps[channel];
}

F64 DynamicThreshold::zResidual(U32 channel) const {
    if (channel >= m_channels) {
        return 0.0;
    }
    return m_window.z(channel, static_cast<F64>(m_latest[channel]));
}

F64 DynamicThreshold::epsilon(U32 channel) const {
    return (channel < m_channels) ? m_eps[channel] : 0.0;
}

}  // namespace Sentinel
