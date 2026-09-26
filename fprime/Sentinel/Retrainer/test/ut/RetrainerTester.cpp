// ======================================================================
// \title  RetrainerTester.cpp
// \brief  E1, the pipe. docs/MODELS.md 47.
// ======================================================================
#include "RetrainerTester.hpp"

#include <unistd.h>

#include <cstdlib>

namespace Sentinel {

RetrainerTester::RetrainerTester()
    : RetrainerGTestBase("RetrainerTester", RetrainerTester::MAX_HISTORY_SIZE),
      component("Retrainer")
{
    this->initComponents();
    this->connectPorts();
}

RetrainerTester::~RetrainerTester() {}

// (!) connectPorts() and initComponents() are NOT defined here. `UT_AUTO_HELPERS`
// generates both into RetrainerTesterHelpers.cpp; defining them again is a
// redefinition, and defining one of them wrongly is how a tester ends up driving a
// component whose ports are half-connected. They are declared in the header and
// supplied by the autocoder, which is what MonitorTester does too.

void RetrainerTester::testTheWholeLifecycle()
{
    // -- D84 / docs/MODELS.md 77, SX14: the candidate, when the loop guard asks --
    // tests/test_retrained_candidate_reloads_in_the_monitor.py sets
    // SENTINEL_LOOP_DIR to a directory holding a flying file at this build's
    // shape. The candidate is written beside it, for the Monitor's RELOAD_MODEL.
    // Without it the lifecycle below is unchanged and no candidate is attempted.
    // It rides inside THIS test because the OCaml runtime is process-global and
    // cannot be booted twice (the finding recorded in the header).
    const char* const loopDir = std::getenv("SENTINEL_LOOP_DIR");
    if (loopDir != nullptr) {
        ASSERT_EQ(0, ::chdir(loopDir)) << loopDir;
        this->component.configureShadow("loop_flying.bin", "loop_candidate.bin");
    }

    // -- boot: the OCaml runtime starts inside this F' process ----------------
    ASSERT_TRUE(this->component.boot());
    ASSERT_TRUE(this->component.armed());
    ASSERT_EVENTS_RuntimeBooted_SIZE(1);
    ASSERT_EVENTS_RuntimeUnavailable_SIZE(0);
    ASSERT_EVENTS_CallRefused_SIZE(0);

    // -- one tick: the hand-checked numbers ------------------------------------
    // 1.0 .. 10.0. Count 10, sum 55, mean 5.5 -- all three exact in F64, so there
    // is no tolerance for a transcription error to hide in.
    this->invoke_to_schedIn(0, 0);
    ASSERT_TLM_SampleCount_SIZE(1);
    ASSERT_TLM_SampleCount(0, 10U);
    ASSERT_TLM_Sum(0, 55.0);
    ASSERT_TLM_Mean(0, 5.5);
    // 61 / E5-c, FC5: the tick drove a float32 retraining cycle through this
    // port, and the step count it telemeters is D73's budget EXACTLY. A value
    // other than CYCLE_BUDGET is a defect, not a measurement.
    ASSERT_TLM_CycleSteps_SIZE(1);
    ASSERT_TLM_CycleSteps(0, Retrainer::CYCLE_BUDGET);
    ASSERT_EVENTS_StepComplete_SIZE(1);
    ASSERT_EVENTS_CallRefused_SIZE(0);
    if (loopDir != nullptr) {
        // The cycle's own weights, at this build's shape, left the process as a
        // candidate file -- and nothing on that path refused.
        ASSERT_TRUE(this->component.shadowArmed());
        ASSERT_EVENTS_CandidateWritten_SIZE(1);
        ASSERT_EVENTS_CandidateRefused_SIZE(0);
        ASSERT_GT(this->component.candidateBytes(), 0U);
    } else {
        ASSERT_FALSE(this->component.shadowArmed());
        ASSERT_EVENTS_CandidateWritten_SIZE(0);
    }

    // -- a second tick: the OCaml accumulator kept its state across the return --
    // This is what makes it a boundary test rather than a function call.
    this->invoke_to_schedIn(0, 0);
    ASSERT_TLM_SampleCount_SIZE(2);
    ASSERT_TLM_SampleCount(1, 20U);
    ASSERT_TLM_CycleSteps_SIZE(2);
    ASSERT_TLM_CycleSteps(1, Retrainer::CYCLE_BUDGET);
    ASSERT_TLM_Sum(1, 110.0);
    ASSERT_TLM_Mean(1, 5.5);
    ASSERT_EVENTS_CallRefused_SIZE(0);

    // -- to the brim: capacity 64, ten per tick, so six ticks fit ---------------
    for (U32 i = 0U; i < 4U; ++i) {
        this->invoke_to_schedIn(0, 0);
    }
    ASSERT_TLM_SampleCount_SIZE(6);
    ASSERT_TLM_SampleCount(5, 60U);
    ASSERT_EVENTS_CallRefused_SIZE(0);

    // -- and over it: REFUSED with a status code, not thrown -------------------
    // (!) If an OCaml exception could cross the boundary this is where it would,
    // and the process would be in undefined behaviour under -fno-exceptions
    // rather than here asserting. docs/MODELS.md 47.6, prediction X3.
    this->invoke_to_schedIn(0, 0);
    ASSERT_EVENTS_CallRefused_SIZE(1);
    ASSERT_EVENTS_CallRefused(0, RetrainerStatus::ERR_OVERFLOW);

    // The refusal changed nothing: no seventh telemetry point, and the component
    // is still armed and still serving the topology.
    ASSERT_TLM_SampleCount_SIZE(6);
    ASSERT_TRUE(this->component.armed());
}

}  // namespace Sentinel
