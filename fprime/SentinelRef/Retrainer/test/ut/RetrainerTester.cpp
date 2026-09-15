// ======================================================================
// \title  RetrainerTester.cpp
// \brief  E1, the pipe. docs/MODELS.md 47.
// ======================================================================
#include "RetrainerTester.hpp"

namespace Retrain {

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
    ASSERT_EVENTS_StepComplete_SIZE(1);
    ASSERT_EVENTS_CallRefused_SIZE(0);

    // -- a second tick: the OCaml accumulator kept its state across the return --
    // This is what makes it a boundary test rather than a function call.
    this->invoke_to_schedIn(0, 0);
    ASSERT_TLM_SampleCount_SIZE(2);
    ASSERT_TLM_SampleCount(1, 20U);
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

}  // namespace Retrain
