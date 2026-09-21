module Sentinel {

    @ The widest channel set a Monitor instance can watch. Must equal
    @ Config::MAX_CHANNELS in flight/include/sentinel/Config.hpp, which
    @ Monitor.cpp asserts at compile time. Objective.md 10.2 fix 2 models 8 to 12
    @ channels in one subsystem and scales by adding instances, not by widening.
    constant MAX_CHANNELS = 16

    @ One tick's worth of watched channel values, in the model's channel order
    array ChannelVector = [MAX_CHANNELS] F32 format "{f}"

    @ Per-channel scale for the Level 1 baseline. A PrmDb-style parameter rather
    @ than a model.bin field, because a bad header CRC makes that file
    @ unreadable in exactly the case Level 1 exists for (D34)
    array ChannelScale = [MAX_CHANNELS] F64 default [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ] format "{f}"

    @ Upstream delivers one vector of channel values per rate-group tick.
    @ Objective.md decision 3 is resolved by D33: direct port wiring rather than
    @ a telemetry-path tap, because Fw.Tlm carries a serialized TlmBuffer and
    @ model.bin's CHANNELS record has no per-channel type tag to deserialize it
    @ with. A mission supplies a small adapter -- F's own "Passive Adapter
    @ Pattern" -- converting its typed telemetry into this vector.
    @
    @ `valid` is false when any watched channel was not measured this cycle,
    @ which scores negative infinity: nothing measured is never an alarm.
    port ChannelSample(
                        ref values: ChannelVector
                        valid: bool
                      )

    @ Which detector is running. This is the active-tier telemetry channel
    @ Objective.md 14.10 and D5 require, and D30 consequence 6 wires
    @ model.bin's baseline_only flag to it.
    enum Mode: U8 {
        @ The GRU forecaster and the frozen decision layer
        MODEL = 0
        @ Level 1: the statistical baseline, after a refusal or baseline_only
        BASELINE = 1
    }

    @ The loader's verdict. Mirrors Sentinel::LoadStatus in
    @ flight/include/sentinel/Status.hpp value for value -- CPP-23 wants the
    @ ground-facing type modelled in FPP, and Monitor.cpp asserts the two agree.
    @ Named ModelLoadStatus rather than LoadStatus because the core already owns
    @ that name in this namespace.
    enum ModelLoadStatus: U8 {
        OK = 0
        BAD_MAGIC = 1
        BAD_VERSION = 2
        BAD_HEADER_CRC = 3
        BAD_STATIC_CRC = 4
        BAD_PARAM_CRC = 5
        BAD_ARCH = 6
        BAD_GATE_ORDER = 7
        BAD_SHAPE = 8
        TOO_LARGE = 9
        TRUNCATED = 10
        BAD_NORM_POLICY = 11
        @ D68's code, mirrored here 2026-09-17. It was added to the core's
        @ LoadStatus and not to this enum, where 12 already meant NOT_LOADED --
        @ so a file refused for an unknown param_version would have been reported
        @ to the ground as "never read, nothing refused". See docs/MODELS.md 20.13.
        BAD_PARAM_VERSION = 12
        @ Not a loader code: the file was never read, so nothing was refused.
        @ (!) IT SITS ABOVE EVERY CORE CODE AND MUST STAY THERE. Monitor.cpp
        @ asserts that at compile time, so the next code the core gains collides
        @ with the build rather than with the ground's reading of a refusal.
        NOT_LOADED = 13
    }

    @ Why the component is running the baseline instead of the model
    enum DegradeReason: U8 {
        @ The loader refused the file; the accompanying event carries the code
        MODEL_REFUSED = 0
        @ The file loaded and set baseline_only, which is Level 1 by request
        BASELINE_ONLY_SET = 1
        @ No model file has been loaded at all
        NO_MODEL_FILE = 2
    }

    @ Cross-channel telemetry monitor: warns before a limit trips, and never commands
    @
    @ (!) QUEUED, NOT PASSIVE, AND D32 CONSEQUENCE 2 AUTHORISED IT BEFORE IT WAS
    @ NEEDED: "Work item 10 promotes it to `queued`, because its reload path is an
    @ `async` command. That is a one-word change in the FPP plus a queue dispatch
    @ at the top of the `schedIn` handler, and it is recorded here so work item 10
    @ does not have to rediscover it." No decision entry is opened for it.
    @
    @ Both input ports stay `sync`, so the cyclic work still runs in the context of
    @ the rate group that ticks it and a slip is still visible as a slip. Only the
    @ command is asynchronous, and it is dispatched inside the tick -- which is
    @ what F's own component-and-port-selection.md prescribes, and what
    @ `Svc.Health` does.
    queued component Monitor {

        # ----------------------------------------------------------------------
        # Ports
        # ----------------------------------------------------------------------

        @ The rate-group tick. One tick, one step of the detector -- and, first,
        @ one drain of the command queue. A sync Svc.Sched input is what F's own
        @ component-and-port-selection.md prescribes for cyclic work, so that a
        @ rate-group slip is visible as a slip rather than hidden on a thread
        @ (D32); the same document prescribes `queued` with the queue dispatched
        @ inside that handler once the component also accepts event-driven work,
        @ which the reload command is.
        sync input port schedIn: Svc.Sched

        @ The per-tick channel vector from the mission's adapter. Sync, and on
        @ the same rate group at a lower port index than schedIn, so both run on
        @ one thread and no mutex is needed. A mission wiring a producer on
        @ another thread makes this port guarded.
        sync input port channelsIn: ChannelSample

        # ----------------------------------------------------------------------
        # Telemetry
        # ----------------------------------------------------------------------

        @ The maximum across channels this tick, whichever detector produced it
        telemetry Score: F32

        @ The active cut: the model's calibrated threshold, or the baseline's
        telemetry Threshold: F64

        @ MODEL or BASELINE. The active-tier channel (D5, Objective.md 14.10)
        telemetry ActiveMode: Mode

        @ Ticks since the detector became allowed to speak; 0 while warming up
        telemetry TicksSinceWarmup: U32

        @ The loader's verdict on the model file
        telemetry LoadStatus: ModelLoadStatus

        # ----------------------------------------------------------------------
        # Events
        # ----------------------------------------------------------------------

        @ The model file loaded and verified; the forecaster is armed
        event ModelLoaded(
                           channels: U32
                           layers: U32
                           tier: U8
                           provenance: string size 64
                         ) \
            severity activity high \
            id 0 \
            format "Sentinel model loaded: {} channels, {} layers, tier {}, fitted {}"

        @ The loader refused the file. One event over all 12 refusal codes,
        @ because the code is data and twelve near-identical events would be
        @ twelve places for the text to drift.
        event ModelRefused(
                            reason: ModelLoadStatus
                            bytesRead: U32
                          ) \
            severity warning high \
            id 1 \
            format "Sentinel refused the model file: {} after {} bytes"

        @ Level 1. The component is serving the topology on the statistical
        @ baseline instead of the model, and has not failed (D5).
        event DegradedToBaseline(
                                  reason: DegradeReason
                                  status: ModelLoadStatus
                                ) \
            severity warning high \
            id 2 \
            format "Sentinel degraded to the Level 1 baseline: {} ({})"

        @ The warning. Names the channel whose smoothed residual took the
        @ maximum, and the score that crossed. Warn-only: this component has no
        @ commanding port of any kind (Objective.md 11 rule 3).
        event CrossChannelWarning(
                                   channelId: U32
                                   channelName: string size 16
                                   score: F32
                                   threshold: F64
                                 ) \
            severity warning high \
            id 3 \
            format "Sentinel: channel {} ({}) at {} crossed {}" \
            throttle 10

        @ Enough history has been seen for a warning to mean something
        @ (Objective.md 11 rule 2, silent until validated)
        event WarmupComplete(
                              ticks: U32
                              mode: Mode
                            ) \
            severity activity high \
            id 4 \
            format "Sentinel warm: {} ticks in {} mode; warnings are now enabled"

        @ The throttled warning has been reset, so warnings resume
        @ A reload was commanded and the file named was loaded. The model that
        @ arrives is reported by ModelLoaded as usual; this says which path it
        @ came from, because the operator needs to know the command took the file
        @ they approved and not the one already there.
        event ModelReloadAccepted(
                                   modelPath: string size 40
                                 ) \
            severity activity high \
            id 6 \
            format "Model reloaded from {}"

        @ (!) THE COMMANDED RELOAD FAILED AND THE PREVIOUS MODEL WAS RESTORED.
        @
        @ `loadModel` degrades to the Level 1 baseline on any refusal, which is
        @ right at topology setup -- there is nothing to fall back to. It is wrong
        @ for a commanded reload: a candidate that does not load would otherwise
        @ cost the operator the model that was working. So the previous path is
        @ reloaded and this says so. `Objective.md` section 12 asks for exactly
        @ this -- "Previous model kept for rollback".
        @
        @ If the rollback ALSO fails, the component is on the baseline and
        @ DegradedToBaseline has already said so with the code.
        event ModelReloadRolledBack(
                                     attempted: string size 40
                                     restored: string size 40
                                     status: ModelLoadStatus
                                   ) \
            severity warning high \
            id 7 \
            format "Reload of {} refused ({}); restored {}"

        @ (!) THE CANDIDATE LOADED, AND IT WAS STILL WRONG FOR THIS TOPOLOGY.
        @
        @ `loadModel` lets the file's channel count win over the topology's --
        @ "the weights only make sense at their own shape" -- and at startup that
        @ is right, because the file is the only thing that knows. On a COMMANDED
        @ reload it is not: the channel source is already wired and has not
        @ changed, so a candidate declaring a different width is a model for a
        @ different subsystem. Accepting it would leave the detector scoring
        @ whatever the unwired slots of the vector happened to hold.
        @
        @ Found by running it (`docs/MODELS.md` 73): a 16-channel candidate was
        @ accepted into an 8-channel deployment and nothing objected.
        event ModelReloadWidthRefused(
                                       attempted: string size 40
                                       expected: U32
                                       offered: U32
                                     ) \
            severity warning high \
            id 8 \
            format "Reload of {} refused: this topology feeds {} channels, the file declares {}"

        event WarningThrottleCleared() \
            severity activity low \
            id 5 \
            format "Sentinel warning throttle cleared"

        # ----------------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------------
        #
        # D34. The Level 1 baseline's constants live here rather than in
        # model.bin, so they are readable when the model file is not. Work item
        # 10's uplink path reaches them by this same mechanism; nothing here
        # implements that path.

        @ Per-channel scale for the baseline, in the model's channel order
        param BASELINE_SCALE: ChannelScale

        @ The baseline's calibrated cut. F64, and compared as F64, for the same
        @ reason model.bin stores the model's threshold as F64: an F32 would
        @ round the cut and could flip a crossing at the boundary.
        param BASELINE_THRESHOLD: F64 default 1.0e30

        ###############################################################################
        # Standard AC Ports: Required for Channels, Events, Commands, and Parameters  #
        ###############################################################################
        # ----------------------------------------------------------------------
        # Commands
        # ----------------------------------------------------------------------

        @ Reload the model from a named file. Work item 10's path, and the only
        @ command this component has ever declared.
        @
        @ (!) IT DOES NOT WEAKEN Objective.md 11 RULE 3. The rule is "warn-only,
        @ commands nothing", and this component still ISSUES no command and has no
        @ commanding port of any kind. Receiving one is rule 1's other half --
        @ "retraining is explicit and human-approved" -- made real: a model only
        @ ever changes because a human sent this.
        @
        @ The path is a parameter rather than implicit so that a candidate can be
        @ uplinked BESIDE the flying model rather than over it. Uplinking over it
        @ would mean a refused candidate had already destroyed the file it
        @ replaced; with a separate path the rollback below has something to
        @ restore, and "roll back" is a second RELOAD_MODEL naming the old path.
        @ (!) OPCODE 0x10, NOT 0x0. The two parameters above generate four
        @ autocoded commands -- a SET and a SAVE each -- and FPP assigns those
        @ from 0, so 0x0 through 0x3 are already spent. `opcode 0x0` here is
        @ rejected as a duplicate, which is the autocoder catching a real
        @ collision rather than a nuisance. 0x10 leaves room for a mission to add
        @ parameters without moving this.
        @ (!) 40 CHARACTERS, AND IT IS THE PLATFORM'S BOUND RATHER THAN A CHOICE.
        @ `Fw::CmdStringArg` is `StringTemplate<FW_CMD_STRING_MAX_SIZE>`
        @ (`Fw/Cmd/CmdString.hpp:14`) and that constant is **40** in F's default
        @ configuration, so a longer path is TRUNCATED at the command boundary
        @ whatever this declaration says. Declaring 80 here promised something the
        @ transport does not deliver, and the truncation showed up as a reload
        @ that silently rolled back. Operationally: the candidate has to live
        @ somewhere short -- beside the binary, not at a long absolute path.
        async command RELOAD_MODEL(
                                    modelPath: string size 40 @< the file to load
                                  ) \
            opcode 0x10

        @ Command receive. F' requires a command recv port on any component
        @ that declares parameters: the parameter protocol is implemented as the
        @ autocoded PARAM_SET and PARAM_SAVE commands, and since work item 10 for
        @ RELOAD_MODEL above. The component ISSUES no command and has no
        @ commanding port of any kind, which is what Objective.md 11 rule 3 is
        @ about; it receives one, which is rule 1's human approval.
        command recv port cmdIn

        @ Command registration
        command reg port cmdRegOut

        @ Command response
        command resp port cmdResponseOut

        @ Port for requesting the current time
        time get port timeCaller

        @ Enables event handling
        import Fw.Event

        @ Enables telemetry channels handling
        import Fw.Channel

        @ Port to return the value of a parameter
        param get port prmGetOut

        @Port to set the value of a parameter
        param set port prmSetOut

    }
}
