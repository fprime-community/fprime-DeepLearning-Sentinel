// ======================================================================
// \title  ExampleSource.hpp
// \brief  A deterministic synthetic feed for the worked adoption chain
//
// (!) NOT A SENSOR, AND NOT A MODEL OF ONE. It exists so a reader can follow
// source -> adapter -> Monitor in one place without reading the physics testbed
// (`PowerSim/`). A mission replaces this component with its own channels, and
// that replacement IS the adoption step.
//
// (!) NOT INSTANCED. `fprime/SentinelRef/Top/` wires `powerSim` into
// `sentinelMonitor.channelsIn`, which is a single `sync` port; a second producer
// would make the testbed's run non-deterministic. See `ExampleSource.fpp`.
//
// Deterministic from a seed, so the pattern is the same on every run and on
// every machine: the integer recurrence below is the same shape `cycle_c.ml`'s
// `seed_from` uses, chosen for exactly that reason -- no floating-point state,
// nothing platform-dependent.
// ======================================================================

#ifndef Example_ExampleSource_HPP
#define Example_ExampleSource_HPP

#include "SentinelRef/ExampleSource/ExampleSourceComponentAc.hpp"

namespace Example {

class ExampleSource final : public ExampleSourceComponentBase {
  public:
    explicit ExampleSource(const char* const compName);
    ~ExampleSource();

    //! Set the seed. Called at topology setup if at all; the default is
    //! deterministic on its own, so a deployment that never calls this still
    //! produces the same run every time.
    void configure(U32 seed);

  private:
    //! One tick: emit one value per channel, in index order, then the count.
    void schedIn_handler(FwIndexType portNum, U32 context) override;

    //! The recurrence. Integer state, float32 output in [-1, 1).
    F32 next();

    U32 m_state;
    U32 m_ticks;
};

}  // namespace Example

#endif
