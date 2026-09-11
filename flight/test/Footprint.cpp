// The static memory footprint, measured against `docs/MODELS.md` 19 prediction F1.
//
// The prediction was committed before this code existed. Whatever this prints is
// what goes in the work-item report.
#include <cstdio>

#include "sentinel/Detector.hpp"
#include "TestSupport.hpp"

namespace SentinelTest {
int failures = 0;
}

using namespace Sentinel;

int main() {
    std::printf("== footprint ==\n");

    const U32 detector = static_cast<U32>(sizeof(Detector));
    const U32 model = static_cast<U32>(sizeof(Model));
    const U32 weights = static_cast<U32>(sizeof(Model::weights));

    std::printf("    sizeof(Detector)           %9u B  = %.1f KiB\n",
                detector, static_cast<double>(detector) / 1024.0);
    std::printf("    sizeof(Model)              %9u B\n", model);
    std::printf("      of which weight arena    %9u B  (%u float32)\n",
                weights, Format::MAX_PARAMETERS);
    std::printf("    carried state + scratch    %9u B\n", detector - model);
    std::printf("    predicted (MODELS.md 19)     312,642 B declared members\n");

    // The weight arena is the prediction that can be checked exactly: it is
    // arithmetic on the Config maxima and owes nothing to the compiler.
    SentinelTest::checkEqualU32(Format::MAX_PARAMETERS, 75360U,
                                "MAX_PARAMETERS at the compile-time maxima");
    SentinelTest::checkEqualU32(weights, 301440U, "weight arena bytes");

    // (!) 19's F1 IS A RECORD AND IS NOT EDITED. It measured the detector that
    // existed on 2026-09-01 -- forward pass, EWMA, one static compare -- and
    // held at 312,112 B, 530 B under. That object no longer exists: the port
    // added telemanom's dynamic threshold and the derivative stream
    // (`docs/MODELS.md` 39), so the bound below is 39's N3 and F1's is printed
    // above as the history it is.
    //
    // (!) AND N3 IS MISSED. It predicted 581,488 B +/- 64 and the measurement is
    // 603,024 -- **+21,536 B, +3.70%**, outside its own 1% band. Itemised the way
    // F1's 530 B was, because a prediction that fails without an account is just
    // a number that moved:
    //
    //     +8,960   the rings are SOLVE_WINDOW deep, not ERROR_WINDOW. 39.6 sized
    //              both at 2,100 when the window the threshold solves over is
    //              2,170 -- 2,100 of history plus the 70 it judges. Two rings,
    //              70 samples, 16 channels, 4 bytes.
    //    +12,168   the pruning ladder's scratch, which 39.6 did not itemise at
    //              all: m_seq, m_kept, m_peaks and m_order at MAX_SEQUENCES =
    //              1,085, plus eps and the latest sample per channel. Shared
    //              across channels, so this is once and not sixteen times.
    //       +408   the second moment-accumulator set carried on each ring, so
    //              one ring serves the threshold's 2,170 contents and arm 2's
    //              2,100 moments.
    //
    // The band is missed, not moved: N3 fails and 39 gains its OBSERVED entry
    // saying so.
    //
    // (!) AND THE MEASUREMENT MOVED AFTERWARDS, WHICH A 2% BAND HID. This check
    // read `detector < 615000U` -- "within 2% of the measured 603,024 B" -- until
    // 2026-09-11. `sizeof` on a fixed toolchain is exact, so a 2% band tolerates
    // about 12 KB of drift, and 8 B of it happened: `99fffd2` added
    // `U32 m_peakChannel` so rule 4 could name a channel, and padding took the
    // object from 603,024 to 603,032. N3's verdict is unchanged -- +3.70% either
    // way -- but the figure five documents quoted was a day stale and nothing
    // could have caught it. Exact from here: a compile-time constant gets an
    // equality, and moving it is a decision somebody makes on purpose.
    std::printf("    predicted (MODELS.md 39, N3)  581,488 B  -- MISSED by +3.70%%\n");
    SentinelTest::checkEqualU32(detector, 603032U,
                                "sizeof(Detector) is exactly the measured 603,032 B");

    // What the flown model actually uses of that budget.
    const U32 flownWeights = 71160U * 4U;
    std::printf("    flown model weights        %9u B  = %.1f KiB\n",
                flownWeights, static_cast<double>(flownWeights) / 1024.0);
    std::printf("    maxima headroom            %9u B  (%.1f%%)\n",
                weights - flownWeights,
                100.0 * static_cast<double>(weights - flownWeights)
                      / static_cast<double>(flownWeights));

    return SentinelTest::report("footprint");
}
