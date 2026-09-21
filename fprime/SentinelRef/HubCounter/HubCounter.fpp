# (!) THIS COMPONENT EXISTS BECAUSE `docs/MODELS.md` 70.5 COULD NOT CLOSE HB2b.
#
# 70.5 measured the crossing at the GROUND and found that it cannot be measured
# there: `Svc.TlmChan` packetises several channels into one downlink packet and
# the GDS distributor aborts a packet at its first unknown id, so a channel's
# arrival count reflects packet composition and decoder behaviour rather than
# whether the hub delivered it. 70.6 owed a counter where the hub EMITS.
#
# (!) IT IS A TAP AND NOT A SINK. `Svc.GenericHub` has no counter of its own and
# `GenericHub.cpp:249-252` drops silently, so the count has to be taken on the
# way past. Everything received is forwarded unchanged, so inserting this
# between the hub and CdhCore changes what the ground sees by nothing at all.
#
# (!) NOT IN THE `SentinelRef` MODULE, and the build is why -- the same trap
# `PowerSim.fpp`, `Retrainer.fpp` and `ChannelAdapter.fpp` all record: a
# component declared in the deployment's own namespace has its autocoded names
# placed in the same scope as the deployment's INSTANCE names, F' builds at
# -Wshadow -Werror, and the topology then will not compile.
#
# (!) APPARATUS, NOT PRODUCT. `fprime/library.cmake` exports `Sentinel/Monitor`
# and nothing here.
module HubTap {

    @ A passive tap on a hub's received events and telemetry, counting by id.
    @
    @ Passive and sync on purpose: it runs in the context of whatever calls it,
    @ which is the hub's own dispatch, so a count is taken at the instant the
    @ hub emits and no queue can reorder or lose one before it is counted.
    passive component HubCounter {

        # ----------------------------------------------------------------------
        # The tap
        # ----------------------------------------------------------------------

        @ Events received from the hub, counted and forwarded.
        sync input port eventIn: Fw.Log

        @ Where they go on to -- the deployment's own event manager.
        output port eventFwd: Fw.Log

        @ Telemetry received from the hub, counted and forwarded.
        sync input port tlmIn: Fw.Tlm

        @ Where it goes on to -- the deployment's own telemetry database.
        output port tlmFwd: Fw.Tlm

        @ Reports the tallies. Driven by a rate group.
        sync input port schedIn: Svc.Sched

        # ----------------------------------------------------------------------
        # What it reports
        # ----------------------------------------------------------------------

        @ The tallies, as a TEXT event on purpose.
        @
        @ (!) A TEXT EVENT AND NOT TELEMETRY, AND THAT IS THE WHOLE POINT. Reporting
        @ these as channels would send them through the same `Svc.TlmChan` and the
        @ same downlink whose behaviour made 70.5 unmeasurable. The text path goes
        @ straight to `Svc.PassiveTextLogger` and out of the process, so the count
        @ does not depend on the thing it is counting.
        event Tallies(
                       ticks: U32       @< source ticks, from SampleCount / SAMPLES_PER_TICK
                       events: U32      @< events received from the hub
                       channels: U32    @< telemetry points received from the hub
                       complete: U32    @< ticks that delivered ALL FIVE channels
                     ) \
            severity activity high \
            id 0 \
            format "hub tap: {} source ticks, {} events, {} channels, {} complete ticks"

        @ A telemetry point arrived whose id is not one this tap knows.
        @ Reported once and then counted silently, because a hub carrying an
        @ unexpected id is a finding and a hub carrying it 300 times is the same
        @ finding.
        event UnknownChannel(
                              chanId: U32
                            ) \
            severity warning low \
            id 1 \
            format "hub tap: telemetry id {} is not one of the retrainer's five" \
            throttle 1

        # ----------------------------------------------------------------------
        # Framework ports
        # ----------------------------------------------------------------------

        time get port timeCaller

        @ Provides the event output ports, the text-event port included.
        import Fw.Event
    }
}
