# D85 / docs/MODELS.md 78.11, LC2: the Monitor's own emit, read where it is made.
#
# (!) APPARATUS, NOT PRODUCT, AND THE MONITOR IS NOT TOUCHED. LC2 asks whether the
# retrainer's replica emits on exactly the detector's ticks, and nothing the Monitor
# sends can answer it: `CrossChannelWarning` goes silent after 10 (`throttle 10`,
# Monitor.fpp) and the `Score` channel is the residual maximum, not the fused score the
# cut is applied to. So this component asks the core itself, through the read-only
# `Monitor::detector()` accessor that already exists for tests, on the same rate-group
# thread right after the Monitor's sync `schedIn`. It has no port into the Monitor and no
# command, and nothing it does can change what the Monitor computes.
#
# Instanced only in SentinelRef's LOOP variant (Top/loop/instances_loop.fpp), and silent
# unless SentinelRef is started with -E. OFF, SentinelRef is exactly what it was.
#
# (!) NOT IN THE `SentinelRef` MODULE, for HubCounter.fpp's reason: a component in the
# deployment's namespace collides with its instance names under -Wshadow -Werror.
module EmitTap {

    @ Logs every tick the Monitor's detector emitted on, by the tap's sequence number.
    passive component EmitProbe {

        @ Driven by the 1 Hz rate group, after the Monitor.
        sync input port schedIn: Svc.Sched

    }
}
