# (!) THIS COMPONENT IS NOT IN THE `SentinelRef` MODULE, AND THE BUILD IS WHY.
# Same trap `PowerSim.fpp` records: a component declared in the deployment's own
# namespace has its autocoded names placed in the same scope as the deployment's
# INSTANCE names, F' builds at -Wshadow -Werror, and the topology then will not
# compile. Apparatus gets its own namespace for the same reason product does.
#
# E1, the pipe (`docs/MODELS.md` 47.9, predictions X1 to X3). This component exists
# to prove a chain -- compile, link, OCaml runtime startup, the C boundary, and F'
# integration -- and for no other purpose. It carries NO machine learning.
module Retrain {

    @ How many samples the E1 accumulator is sized for. Fixed at init and never
    @ grown: CPP-1 forbids allocation after initialisation, and the OCaml side
    @ refuses a feed that would exceed it rather than reallocating.
    constant RETRAINER_CAPACITY = 64

    @ The OCaml side's verdict, mirroring `sentinel_retrainer.h` exactly.
    @ (!) EVERY ONE OF THESE IS A STATUS CODE AND NOT AN EXCEPTION. F' forbids
    @ exceptions (CPP-25) and this deployment builds -fno-exceptions, so an OCaml
    @ exception crossing the boundary would be undefined behaviour. `retrainer.ml`
    @ catches every exception and returns one of these instead, which is what
    @ `docs/MODELS.md` 47.6 means by structurally impossible.
    enum RetrainerStatus: U8 {
        OK               = 0
        ERR_NOT_INIT     = 1
        ERR_ALREADY_INIT = 2
        ERR_CAPACITY     = 3
        ERR_OVERFLOW     = 4
        ERR_SHORT_BUFFER = 5
        ERR_INTERNAL     = 6
        ERR_NO_RUNTIME   = 7
    }

    @ The E1 pipe: an F' component that calls into an OxCaml static library.
    @
    @ (!) THIS IS TEST APPARATUS AND IT LIVES IN THE DEPLOYMENT, NOT THE LIBRARY.
    @ `fprime/library.cmake` exports `Sentinel/Monitor` and nothing here. A mission
    @ adopting Sentinel does not inherit an OCaml runtime, and will not until E1 to
    @ E4 have all passed and a decision beyond D70 says so.
    @
    @ (!) AND IT RUNS IN ITS OWN PROCESS, WHICH IS A REQUIREMENT AND NOT A CHOICE.
    @ D70 consequence 2: OCaml 5's minor collector is stop-the-world across all
    @ domains, so a retrainer sharing a process with the detector would stall it at
    @ a GC barrier even if the retrainer's own code allocated nothing. E2 measures
    @ whether the process boundary is enough.
    passive component Retrainer {

        # ----------------------------------------------------------------------
        # Ports
        # ----------------------------------------------------------------------

        @ The rate-group tick. Passive with a sync Svc.Sched input, which is what
        @ F's own component-and-port-selection.md prescribes for cyclic work and
        @ what `Sentinel.Monitor` already does (D32).
        sync input port schedIn: Svc.Sched

        # ----------------------------------------------------------------------
        # Telemetry
        # ----------------------------------------------------------------------

        @ Samples accumulated since init
        telemetry SampleCount: U32

        @ The accumulator's running sum, F64 -- the shape
        @ `flight/src/TrailingWindow.cpp` already uses, never differenced float32
        @ prefix sums, which is D37's defect
        telemetry Sum: F64

        @ The accumulator's mean, or 0 when nothing has been fed
        telemetry Mean: F64

        @ The OCaml side's last verdict
        telemetry LastStatus: RetrainerStatus

        @ Optimiser steps the last retraining cycle took (docs/MODELS.md 61).
        @ D73 fixes it per cycle, so a value other than the budget is a defect
        @ and not a measurement.
        telemetry CycleSteps: I32

        # ----------------------------------------------------------------------
        # Events
        # ----------------------------------------------------------------------

        @ The OCaml runtime started and the five registered closures resolved
        event RuntimeBooted(
                             capacity: U32
                           ) \
            severity activity high \
            id 0 \
            format "OxCaml retrainer booted, accumulator sized {}"

        @ The runtime could not be started, or its closures were absent. The
        @ component does nothing further; it raises no other event and it never
        @ retries, because a runtime that did not start is not a transient fault.
        event RuntimeUnavailable() \
            severity warning high \
            id 1 \
            format "OxCaml retrainer runtime unavailable; component is inert"

        @ One step completed and the accumulator was read back across the boundary
        event StepComplete(
                            count: U32
                            sum: F64
                            mean: F64
                          ) \
            severity activity low \
            id 2 \
            format "OxCaml retrainer step: {} samples, sum {}, mean {}" \
            throttle 10

        @ The OCaml side refused a call and said why. One event over all seven
        @ refusal codes, the way `Sentinel.Monitor`'s ModelRefused covers its own.
        event CallRefused(
                           status: RetrainerStatus
                         ) \
            severity warning low \
            id 3 \
            format "OxCaml retrainer refused a call: {}"

        # ----------------------------------------------------------------------
        # Framework ports
        # ----------------------------------------------------------------------
        #
        # (!) NO COMMAND PORTS, AND NO PARAMETERS. This component declares no
        # parameter, so F' does not require a command recv port, and it therefore
        # declares none at all -- which also means the -Wshadow collision the
        # header of this file describes cannot arise here. Objective.md 11 rule 3
        # is about the product and this is apparatus, but a piece of apparatus
        # that commands nothing is one less thing to reason about.

        @ Port for requesting the current time
        time get port timeCaller

        @ Enables event handling
        import Fw.Event

        @ Enables telemetry channels handling
        import Fw.Channel

    }
}
