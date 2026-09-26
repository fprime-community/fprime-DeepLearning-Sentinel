// ======================================================================
// \title  RetrainerTester.hpp
// \brief  Test harness for Sentinel::Retrainer -- E1, the pipe, and 72's candidate
//
// (!) THIS IS THE PROCESS E1 RUNS IN. `docs/MODELS.md` 47.9 X1 asks for an F'
// component linked against the OxCaml library and running in its own process,
// returning a hand-checkable result. A unit-test binary is that process: it hosts
// the OCaml runtime, drives the component through F's own generated port
// machinery, and reads the telemetry and events back out.
//
// What it does NOT establish is the separate-process ISOLATION D70 consequence 2
// requires. That is E2, it is measured and not asserted, and it is the experiment
// that can end the approach.
// ======================================================================

#ifndef Sentinel_RetrainerTester_HPP
#define Sentinel_RetrainerTester_HPP

#include "Sentinel/Retrainer/Retrainer.hpp"
#include "Sentinel/Retrainer/RetrainerGTestBase.hpp"

namespace Sentinel {

class RetrainerTester final : public RetrainerGTestBase {
  public:
    static const FwSizeType MAX_HISTORY_SIZE = 512;
    static const FwEnumStoreType TEST_INSTANCE_ID = 0;

    RetrainerTester();
    ~RetrainerTester();

    //! (!) ONE TEST, AND THE REASON IS A FINDING RATHER THAN A CONVENIENCE.
    //! OCaml module-level state is PROCESS-global: `retrainer.ml`'s accumulator
    //! lives in a module-level ref, `caml_startup` runs once per process, and a
    //! second component instance in the same process therefore re-attaches to the
    //! same accumulator -- `init` refuses it with ERR_ALREADY_INIT, correctly.
    //!
    //! That is CONSISTENT with the architecture rather than in tension with it:
    //! `docs/DECISIONS.md` D70 consequence 2 puts the retrainer in its own OS
    //! process, so one process means one retrainer and process-global state is the
    //! right shape. What it rules out is a test-per-behaviour structure, because
    //! gtest runs every case in one process and no case can get a fresh runtime.
    //!
    //! Discovered by this test failing, not by reading the manual, which is what
    //! E1 is for. Recorded at `docs/MODELS.md` 47.13.
    void testTheWholeLifecycle();

  private:
    void connectPorts();
    void initComponents();

    Retrainer component;
};

}  // namespace Sentinel

#endif
