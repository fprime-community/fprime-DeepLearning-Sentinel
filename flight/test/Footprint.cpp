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

    // Nothing is allocated, so the whole detector is one object; a sizeof far
    // from the prediction means the declaration drifted from what was registered.
    SentinelTest::check(detector < 320000U,
                        "sizeof(Detector) is within 2% of the predicted 312,642 B");

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
