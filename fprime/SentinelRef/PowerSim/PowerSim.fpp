# (!) THIS COMPONENT IS NOT IN THE `SentinelRef` MODULE, AND THE BUILD IS WHY.
# A component declared in the deployment's own namespace has its autocoded
# parameter names placed in the same scope as the deployment's INSTANCE names,
# and F' builds at -Wshadow -Werror. `cmdResponseOut_out(..., U32 cmdSeq, ...)`
# then shadows the `cmdSeq` instance of `Svc.CmdSequencer`, and the topology
# will not compile. `Sentinel.Monitor` never hit this because it lives in its
# own module. Apparatus gets its own namespace for the same reason product does.
module Testbed {

    @ The number of channels this subsystem publishes. Must not exceed
    @ Sentinel.MAX_CHANNELS; PowerSim.cpp asserts it at compile time.
    constant POWERSIM_CHANNELS = 8

    @ Which degradation is seeded, if any. Selected by parameter so a run is
    @ reproducible from its seed and its mode alone (docs/MODELS.md 42).
    enum FaultMode: U8 {
        @ No fault. Used for the false-alarm runs 42's T5 measures.
        HEALTHY = 0
        @ Cell internal resistance rises slowly. Every channel stays inside its
        @ limits while the current-to-temperature relationship breaks, which is
        @ the contextual family 42's T2 requires.
        RESISTANCE_RISE = 1
    }

    @ A coupled power and thermal subsystem, simulated, with dictionary limits.
    @
    @ (!) THIS IS TEST APPARATUS AND IT LIVES IN THE DEPLOYMENT, NOT THE LIBRARY.
    @ A mission adopting Sentinel consumes `fprime/Sentinel/Monitor` and should
    @ not inherit a simulated battery. `fprime/library.cmake` exports the Monitor
    @ and nothing here.
    @
    @ The physics is deterministic and seeded: the same seed and mode produce the
    @ same trace, tick for tick, which is what makes a warning time measurable at
    @ all (Objective.md 11 rule 5 applies to the detector; this holds itself to
    @ the same bar so a measurement can be repeated).
    passive component PowerSim {

        # ----------------------------------------------------------------------
        # The watched channels, with their limits
        # ----------------------------------------------------------------------
        #
        # (!) THE LIMITS ARE DECLARED HERE AND EVALUATED ONBOARD, AND 42.3
        # DEPARTURE 1 IS WHY. F' checks these on the GROUND -- "Limit checking is
        # performed by the ground system based on the dictionary definition"
        # (F' v4.3.0 docs/reference/system-functional/telemetry-chan.md:32) -- so
        # a ground-stamped limit trip and an onboard-stamped warning sit on
        # different clocks, and subtracting one from the other measures the
        # downlink. `PowerSim.cpp` evaluates these same numbers itself and emits
        # `LimitTripped` with the onboard time, so both instants share a base.
        # The ground check still runs; the difference between the two is a
        # measurement of the downlink path and is reported as one.

        @ Solar array input power.
        @
        @ (!) ITS LOW LIMITS SIT BELOW ZERO, AND MEASURING IT IS WHAT FOUND THAT.
        @ A 5 W yellow low fired on tick 0 of every run, healthy ones included:
        @ the array reads 0 W through eclipse, which is 35% of every orbit. Low
        @ solar is not a fault -- it is night. Whether it is anomalous depends on
        @ whether the array SHOULD be lit, which is a question about the
        @ relationship between this channel and the orbit phase and is exactly
        @ the class of thing no limit can hold (Objective.md 2). So the low
        @ limits are set where a sensor bias would be and nowhere near the
        @ operating floor.
        telemetry SolarInput: F32 id 0 \
            low { red -5.0, yellow -2.0 } \
            high { red 320.0, yellow 300.0 }

        @ Battery charge current, positive into the cell
        telemetry ChargeCurrent: F32 id 1 \
            low { red -12.0, yellow -10.0 } \
            high { red 12.0, yellow 10.0 }

        @ Bus load current
        telemetry LoadCurrent: F32 id 2 \
            low { red 0.0, yellow 0.5 } \
            high { red 11.0, yellow 9.5 }

        @ Regulated bus voltage
        telemetry BusVoltage: F32 id 3 \
            low { red 26.0, yellow 27.0 } \
            high { red 33.0, yellow 32.0 }

        @ Cell stack temperature
        telemetry CellTemp: F32 id 4 \
            low { red -10.0, yellow -5.0 } \
            high { red 45.0, yellow 40.0 }

        @ Radiator temperature
        telemetry RadiatorTemp: F32 id 5 \
            low { red -40.0, yellow -35.0 } \
            high { red 35.0, yellow 30.0 }

        @ Survival heater duty cycle
        telemetry HeaterDuty: F32 id 6 \
            low { red -0.01, yellow 0.0 } \
            high { red 1.01, yellow 1.0 }

        @ Battery state of charge
        telemetry StateOfCharge: F32 id 7 \
            low { red 0.20, yellow 0.30 } \
            high { red 1.01, yellow 1.0 }

        @ Ticks elapsed since the run began. The trace's own clock, in timesteps.
        telemetry SimTick: U32 id 8

        # ----------------------------------------------------------------------
        # Events
        # ----------------------------------------------------------------------

        @ A watched channel crossed a RED limit, evaluated onboard against the
        @ values this component's own dictionary declares (42.3 departure 1).
        @ This is the instant a warning time is measured against.
        event LimitTripped(
                            channel: U32 @< Index in the published channel order
                            value: F32   @< The value that crossed
                            tick: U32    @< Ticks since the run began
                          ) \
            severity warning high \
            id 0 \
            format "RED limit tripped: channel {} at {f}, tick {}"

        @ The seeded degradation began. Ground truth by construction: the fault
        @ is injected here and nowhere else, so its onset needs no annotation.
        event FaultInjected(
                             mode: FaultMode @< Which degradation
                             tick: U32       @< Ticks since the run began
                           ) \
            severity activity high \
            id 1 \
            format "fault {} injected at tick {}"

        @ The run reached its configured length with no RED crossing.
        event RunCompletedInLimits(
                                    ticks: U32 @< Length of the run
                                  ) \
            severity activity high \
            id 2 \
            format "run completed in limits after {} ticks"

        # ----------------------------------------------------------------------
        # Parameters -- a run is reproducible from these alone
        # ----------------------------------------------------------------------

        @ Seed for the deterministic noise. Same seed, same trace.
        param SEED: U32 default 1

        @ Which degradation to inject
        param FAULT_MODE: FaultMode default FaultMode.HEALTHY

        @ Tick at which the degradation begins
        param FAULT_START: U32 default 8000

        @ Fractional increase in cell internal resistance per tick, once the
        @ fault is running. Dimensionless (D55): a rate in ohms would be a
        @ property of this particular cell.
        param FAULT_RATE: F32 default 2.0e-5

        # ----------------------------------------------------------------------
        # Ports
        # ----------------------------------------------------------------------

        @ One tick of physics, driven by the rate group
        sync input port schedIn: Svc.Sched

        @ The watched channel values, in the model's channel order, to whatever
        @ is listening. 42.2: this is the port SentinelRef's topology left
        @ unconnected, and connecting it is what 42 is.
        output port channelOut: Sentinel.ChannelSample

        ###############################################################################
        # Standard AC Ports                                                           #
        ###############################################################################
        @ Command receive, required because this component declares parameters
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
        @ Port to set the value of a parameter
        param set port prmSetOut

    }
}
