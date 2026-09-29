# (!) THIS COMPONENT IS IN THE LIBRARY'S `Sentinel` MODULE SINCE D84, AND NOT IN A
# DEPLOYMENT'S. `PowerSim.fpp` records the trap that kept it out of `SentinelRef`'s:
# a component declared in the deployment's own namespace has its autocoded names in
# the same scope as the deployment's INSTANCE names, F' builds at -Wshadow -Werror,
# and the topology then will not compile. It lived in its own `Retrain` module until
# D84 exported it; a library exports ONE namespace, so it now shares `Sentinel` with
# the Monitor, and no deployment is named that.
#
# E1, the pipe (`docs/MODELS.md` 47.9, predictions X1 to X3), then 61's float32
# retraining cycle and 72's candidate file. Its window is a deterministic drive and
# not telemetry: what it proves is the chain -- compile, link, OCaml runtime startup,
# the C boundary, F' integration, a candidate at the mission's shape -- and it trains
# on nothing a mission has flown.
module Sentinel {

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

    @ Where on the candidate path a refusal happened (72 / E5-e, HO1). A named
    @ stage rather than a bare code, so an operator reading the event knows which
    @ crossing failed without consulting three headers.
    enum CandidateStage: U8 {
        @ sentinel_cycle_export, reading the cycle's weights out
        CYCLE_EXPORT  = 0
        @ sentinel_shadow_write, the weights into the model file
        SHADOW_WRITE  = 1
        @ sentinel_shadow_export, the file out into this process
        SHADOW_EXPORT = 2
        @ Os::File::open on the candidate path
        FILE_OPEN     = 3
        @ Os::File::write, short or refused
        FILE_WRITE    = 4
        @ D85: the flying model file could not be read or was refused by the replica
        FLYING_READ   = 5
        @ D85: the training loop (window rule, cycle, warm start) returned a failure
        LOOP          = 6
    }

    @ The retrainer: an F' component that calls into an OxCaml object.
    @
    @ (!) EXPORTED OPT-IN, OFF BY DEFAULT (D84). `fprime/library.cmake` registers
    @ it only when SENTINEL_WITH_RETRAINER is ON, so a mission that does nothing
    @ inherits no OCaml runtime. Its status is D83's, exactly: the chosen
    @ retraining implementation, host-verified, not flight-qualified. E2 returned
    @ NO VERDICT and E4 has never run.
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

        @ The rate-group tick. Every OCaml entry point runs on THIS thread, through
        @ RetrainLoop: OCaml 5 pins the domain lock to the thread that booted the
        @ runtime (docs/MODELS.md 65.10), and the boot happens here, on the first tick.
        sync input port schedIn: Svc.Sched

        @ D85: the detector's samples, from the SampleTap in the detector's process,
        @ over the hub. The handler runs on the hub's receive thread, so it only
        @ queues -- under a short lock, into a fixed queue -- and never calls OCaml.
        sync input port sampleIn: SentinelTappedSample

        # ----------------------------------------------------------------------
        # Telemetry -- five channels, in the id range HubCounter counts
        # ----------------------------------------------------------------------

        @ Samples received from the tap (one per detector tick)
        telemetry SamplesReceived: U32

        @ Training steps taken on admitted, healthy windows since boot
        telemetry Admitted: U32

        @ Candidates written since boot
        telemetry Candidates: U32

        @ The last status the loop returned
        telemetry LastStatus: RetrainerStatus

        @ Samples the sequence numbers say were lost in transit, fed to the loop as
        @ missed (invalid) ticks
        telemetry SamplesMissed: U32

        # ----------------------------------------------------------------------
        # Events
        # ----------------------------------------------------------------------

        @ The OCaml runtime started, the replica loaded the flying model, and the cycle
        @ warm-started from its weights
        event RetrainerReady(
                              channels: U32
                              schedule: U32
                            ) \
            severity activity high \
            id 0 \
            format "Retrainer ready: {} channels, a candidate every {} admitted steps"

        @ The runtime could not be started, or its closures were absent. The component
        @ does nothing further and never retries.
        event RuntimeUnavailable() \
            severity warning high \
            id 1 \
            format "OxCaml retrainer runtime unavailable; component is inert"

        @ The loop refused a call and said why
        event CallRefused(
                           status: RetrainerStatus
                         ) \
            severity warning low \
            id 3 \
            format "OxCaml retrainer refused a call: {}" \
            throttle 10

        @ A candidate model file was written, with what the ground gate needs to judge
        @ it: its own static_crc32, the steps it was trained for, and the first and last
        @ ticks of the data it was trained on (docs/DECISIONS.md D85).
        @
        @ The ticks are the TAP'S SEQUENCE NUMBERS -- the detector's ticks since the tap
        @ began -- not this component's own count, which a long gap shortens (a gap feeds
        @ at most QUEUE missed ticks). So the ground finds the exact rows in its own
        @ archive whatever was lost on the way. Counted from 1: tick k is the sample
        @ with sequence number k - 1, the convention LoopSim's candidates.csv shares.
        @
        @ (!) THIS IS NOT A CERTIFICATION. Nothing onboard scores a candidate; the ground
        @ gate does, and a human approves every swap (Objective.md 11 rule 1).
        event CandidateWritten(
                                sizeBytes: U32
                                crc32: U32
                                steps: U32
                                firstTick: U32
                                lastTick: U32
                              ) \
            severity activity high \
            id 4 \
            format "Candidate written: {} B, static_crc32 {}, {} steps on ticks {}..{}" \
            throttle 5

        @ The candidate could not be built or written
        event CandidateRefused(
                                stage: CandidateStage
                                code: I32
                              ) \
            severity warning low \
            id 5 \
            format "Candidate refused at {}: code {}" \
            throttle 5

        @ Samples were lost between the tap and here
        event SamplesLost(
                           missed: U32
                         ) \
            severity warning low \
            id 6 \
            format "{} samples lost in transit; fed to the loop as missed ticks" \
            throttle 5

        # ----------------------------------------------------------------------
        # Framework ports
        # ----------------------------------------------------------------------
        #
        # (!) NO COMMAND PORTS, AND NO PARAMETERS. It commands nothing and is commanded
        # by nothing: a candidate reaches the detector only through the ground gate and
        # a human's RELOAD_MODEL (Objective.md 11 rules 1 and 3).

        @ Port for requesting the current time
        time get port timeCaller

        @ Enables event handling
        import Fw.Event

        @ Enables telemetry channels handling
        import Fw.Channel

    }
}
