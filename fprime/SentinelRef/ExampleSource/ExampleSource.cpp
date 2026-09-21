// ======================================================================
// \title  ExampleSource.cpp
// \brief  A deterministic synthetic feed for the worked adoption chain
// ======================================================================
#include "SentinelRef/ExampleSource/ExampleSource.hpp"
#include "SentinelRef/ExampleSource/FppConstantsAc.hpp"

namespace Example {

namespace {
//! Numerical Recipes' LCG constants. The point is reproducibility, not quality:
//! this is a fixture, and a fixture that differs between runs is not one.
const U32 LCG_A = 1664525U;
const U32 LCG_C = 1013904223U;
const U32 DEFAULT_SEED = 12345U;
}  // namespace

ExampleSource::ExampleSource(const char* const compName)
    : ExampleSourceComponentBase(compName), m_state(DEFAULT_SEED), m_ticks(0U) {}

ExampleSource::~ExampleSource() {}

void ExampleSource::configure(U32 seed) {
    m_state = seed;
    m_ticks = 0U;
}

F32 ExampleSource::next() {
    // Unsigned overflow is defined and is the whole mechanism.
    m_state = (LCG_A * m_state) + LCG_C;
    // Top 16 bits, which are the well-behaved ones in an LCG, mapped to [-1, 1).
    const U32 top = (m_state >> 16U) & 0xFFFFU;
    return (static_cast<F32>(top) / 32768.0F) - 1.0F;
}

void ExampleSource::schedIn_handler(FwIndexType portNum, U32 context) {
    static_cast<void>(portNum);
    static_cast<void>(context);

    // Fixed work per tick: one value per channel, in index order. A mission's
    // own source emits whatever its telemetry path delivers; what matters for
    // the example is that every channel lands before the adapter's schedIn.
    if (this->isConnected_valueOut_OutputPort(0)) {
        for (U32 c = 0U; c < static_cast<U32>(EXAMPLE_SOURCE_CHANNELS); ++c) {
            this->valueOut_out(0, c, this->next());
        }
    }
    m_ticks++;
    this->tlmWrite_Ticks(m_ticks);
}

}  // namespace Example
