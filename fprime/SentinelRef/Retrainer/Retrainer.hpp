// ======================================================================
// \title  Retrainer.hpp
// \brief  E1, the pipe: an F' component that calls into an OxCaml static library
//
// Test apparatus, not product. `fprime/library.cmake` exports `Sentinel/Monitor`
// and nothing here, so a mission adopting Sentinel does not inherit an OCaml
// runtime. See `docs/MODELS.md` 47 and `docs/DECISIONS.md` D70.
//
// It carries NO machine learning. E1 exists to prove a chain -- compile, link,
// OCaml runtime startup, the C boundary, F' integration -- and the payload is
// arithmetic whose answer can be checked by hand for exactly that reason
// (`docs/MODELS.md` 47.2).
//
// (!) NOTHING HERE MAY THROW, AND NOTHING BELOW IT CAN. CPP-25 forbids
// exceptions and this deployment builds -fno-exceptions. `retrainer.ml` catches
// every OCaml exception and returns a status code, so no exception can reach this
// translation unit -- which is what `docs/MODELS.md` 47.6 means by structurally
// impossible rather than merely avoided.
//
// (!) AND NOTHING HERE ALLOCATES AFTER INIT. CPP-1. The accumulator is sized once,
// inside OCaml, at RETRAINER_CAPACITY; the sample and export buffers are members
// of fixed size. The OCaml runtime's own heap is a separate matter and is exactly
// what E2 and E3 exist to measure -- it is NOT claimed here to be allocation-free.
// ======================================================================

#ifndef Retrain_Retrainer_HPP
#define Retrain_Retrainer_HPP

#include "SentinelRef/Retrainer/RetrainerComponentAc.hpp"

namespace Retrain {

class Retrainer final : public RetrainerComponentBase {
  public:
    //! How many samples one tick feeds. Fixed, so the per-cycle work is fixed
    //! (Objective.md 11 rule 5).
    static const U32 SAMPLES_PER_TICK = 10U;

    //! The export buffer's width: count, sum, mean.
    static const U32 EXPORT_WIDTH = 3U;

    // 61 / E5-c. The retraining cycle's fixed extents, claimed once (CPP-1).
    //
    // (!) CYCLE_BUDGET IS NOT THE FLIGHT BUDGET. docs/MODELS.md 55.4 stop 23
    // reserves that derivation; this is a unit test's value, chosen so a tick is
    // cheap. What is under test is that the count is FIXED, not that it is right.
    static const U32 CYCLE_T = 8U;
    static const U32 CYCLE_INS = 16U;
    static const U32 CYCLE_WINDOW = 250U * 16U;   // the OCaml side's full extent
    static const I32 CYCLE_BUDGET = 1;

    explicit Retrainer(const char* const compName);
    ~Retrainer();

    //! Boot the OCaml runtime and size the accumulator. Safe to call before the
    //! topology is running. Returns true if the component is armed; a false
    //! return leaves it inert rather than failing the topology, which is the same
    //! degrade-rather-than-die posture `Sentinel::Monitor` takes on a bad model
    //! file. Never throws, never asserts on the runtime's behaviour.
    bool boot();

    //! Whether boot() succeeded, for the topology and the tests.
    bool armed() const { return m_armed; }

  private:
    //! One tick: feed a fixed block, step, read the accumulator back.
    void schedIn_handler(FwIndexType portNum, U32 context) override;

    //! Emit CallRefused for a non-OK status and record it. Returns true if the
    //! status was OK.
    bool accept(I32 status);

    bool m_armed;
    U32  m_refusals;

    //! 61 / E5-c. The window the cycle reads. THIS COMPONENT OWNS IT; the OCaml
    //! side sees a CAML_BA_EXTERNAL view that does not outlive the call.
    F32  m_window[CYCLE_WINDOW];

    //! Whether the cycle's runtime came up. Separate from m_armed: E1's pipe and
    //! 61's cycle boot independently and either may be absent.
    bool m_cycleArmed;
    //! Fixed-size, member-owned. The OCaml side wraps these in a Bigarray with
    //! CAML_BA_EXTERNAL, so their storage is never owned or moved by the GC.
    F64  m_samples[SAMPLES_PER_TICK];
    F64  m_export[EXPORT_WIDTH];
};

}  // namespace Retrain

#endif
