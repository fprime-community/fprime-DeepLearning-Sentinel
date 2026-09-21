# (!) THIS COMPONENT IS NOT IN THE `SentinelRef` MODULE, AND THE BUILD IS WHY.
# Same trap `PowerSim.fpp` and `Retrainer.fpp` both record: a component declared
# in the deployment's own namespace has its autocoded names placed in the same
# scope as the deployment's INSTANCE names, F' builds at -Wshadow -Werror, and
# the topology then will not compile.
#
# (!) AND THIS IS EXAMPLE CODE, NOT PRODUCT. `fprime/library.cmake` exports
# `Sentinel/Monitor` and nothing here. A mission does not inherit this file; it
# copies it, renames it, and replaces `valueIn` with its own typed telemetry.
# It exists because `Sentinel/Monitor/docs/sdd.md` tells a reader to supply "a
# small adapter" and, until now, the only implementation in the repository was
# buried in a physics model (`PowerSim.cpp:72-85`).
module Example {

    @ One channel's value, as a mission's own telemetry path would deliver it.
    @ A real mission replaces this port with whatever types its channels are:
    @ the adapter's job is exactly this conversion, and nothing else.
    port ChannelValue(
                       index: U32
                       value: F32
                     )

    @ F's Passive Adapter Pattern, at its smallest useful size.
    @ `docs/user-manual/framework/component-and-port-selection.md` names the
    @ pattern and gives `Drv.ByteStreamBufferAdapter` as its canonical example:
    @ "a passive component as an adapter that is called synchronously as part of
    @ the primary port call".
    @
    @ Channels arrive one at a time and the detector wants one vector per tick,
    @ so this holds the latest value of each and emits on the rate group. That
    @ is the whole of it -- about forty lines of C++, with no state a mission
    @ has to reason about beyond "last value wins".
    passive component ChannelAdapter {

        @ One channel's latest value. Sync, so the conversion happens in the
        @ caller's context, which is what makes this an adapter rather than a
        @ queue.
        sync input port valueIn: ChannelValue

        @ The rate-group tick. Wire this at a HIGHER port index than the
        @ producers feeding valueIn, so every channel for this cycle has landed
        @ before the vector is emitted.
        sync input port schedIn: Svc.Sched

        @ The assembled vector, to `Sentinel.Monitor.channelsIn`.
        output port channelOut: Sentinel.ChannelSample

        @ How many channels this adapter publishes. A mission sets it to its own
        @ count; it must not exceed Sentinel.MAX_CHANNELS, which the
        @ implementation asserts at compile time.
        telemetry PublishedChannels: U32

        @ A value arrived for an index at or beyond the published count and was
        @ dropped. The vector is still emitted, because one bad index is not a
        @ reason to blind the detector.
        event IndexRefused(
                            index: U32
                            published: U32
                          ) \
            severity warning low \
            id 0 \
            format "channel index {} refused; this adapter publishes {}" \
            throttle 5

        @ Port for requesting the current time
        time get port timeCaller

        @ Enables event handling
        import Fw.Event

        @ Enables telemetry channels handling
        import Fw.Channel

    }
}
