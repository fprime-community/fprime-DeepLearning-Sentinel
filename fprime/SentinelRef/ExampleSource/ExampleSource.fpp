# (!) THIS COMPONENT IS NOT IN THE `SentinelRef` MODULE, AND THE BUILD IS WHY.
# Same trap `PowerSim.fpp`, `Retrainer.fpp` and `ChannelAdapter.fpp` all record: a
# component declared in the deployment's own namespace has its autocoded names
# placed in the same scope as the deployment's INSTANCE names, F' builds at
# -Wshadow -Werror, and the topology then will not compile.
#
# (!) IT IS A SYNTHETIC FEED FOR THE WORKED EXAMPLE. IT IS NOT A SENSOR, AND IT
# IS NOT A MODEL OF ONE. It emits a fixed deterministic pattern from a seed so
# that a reader can follow the whole adoption chain -- source -> adapter ->
# Monitor -- in one place, without reading the physics testbed. A mission
# replaces THIS component with its own channels; that replacement is the
# adoption step, and `Example.ChannelValue` is the seam it happens at.
#
# (!) AND IT IS NOT INSTANCED IN THIS DEPLOYMENT. `SentinelRef` already feeds
# `sentinelMonitor.channelsIn` from `powerSim`, which is a single `sync` input
# port; a second producer on it would make the testbed's run non-deterministic,
# and instancing would move `connections RateGroups`, which `docs/MODELS.md` 65's
# DP3 recorded byte-identical and 42.9's run depends on. So it is registered,
# built and readable, and nothing wires it. `fprime/README.md` says so too.
module Example {

    @ How many channels this source publishes. Three, because
    @ `flight/test/vectors/p1.bin` -- the loadable example this branch ships --
    @ is a three-channel model, and a reader following the chain should be able
    @ to load that file without the loader refusing it for a width mismatch.
    constant EXAMPLE_SOURCE_CHANNELS = 3

    @ A deterministic synthetic feed, for the worked example only.
    passive component ExampleSource {

        @ The rate-group tick. Wire this at a LOWER port index than the
        @ adapter's schedIn, so every value for this cycle has landed before the
        @ adapter assembles the vector.
        sync input port schedIn: Svc.Sched

        @ One channel's value per call, to `Example.ChannelAdapter.valueIn`.
        @ One call per channel per tick.
        output port valueOut: ChannelValue

        @ How many ticks this source has emitted. The only state it has.
        telemetry Ticks: U32

        @ Port for requesting the current time
        time get port timeCaller

        @ Enables event handling
        import Fw.Event

        @ Enables telemetry channels handling
        import Fw.Channel

    }
}
