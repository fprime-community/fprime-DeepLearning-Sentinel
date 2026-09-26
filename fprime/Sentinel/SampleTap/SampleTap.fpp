# D85 / docs/MODELS.md 78: the telemetry path from the detector's process to the
# retrainer's -- exported OPT-IN with the retrainer, never in the detector's default.
@ A sample on its way to the retrainer: the vector the detector saw, its validity,
@ and a sequence number, so the far side can tell a lost datagram from a quiet tick
@ (the hub runs over UDP, docs/MODELS.md 71.3).
@
@ (!) DECLARED AT THE TOP LEVEL, NOT IN `module Sentinel`, AND THE AUTOCODER IS WHY.
@ The hub carries this port serialized, and fpp v3.3.0's topology code for a
@ serial-to-typed connection names the port's serializer UNQUALIFIED inside
@ `namespace Svc` (SentinelRetrainTopologyAc.cpp: "unknown type name
@ 'TappedSamplePortSerializer'; did you mean 'Sentinel::TappedSamplePortSerializer'").
@ A top-level port has an unqualified name, and the prefix keeps it from colliding
@ with a mission's own.
port SentinelTappedSample(
                           ref values: Sentinel.ChannelVector
                           valid: bool
                           seq: U32
                         )

module Sentinel {

    @ The tap between a mission's adapter and the Monitor.
    @
    @ (!) THE DETECTOR'S INPUT IS FORWARDED FIRST AND UNCHANGED. `sampleOut` carries
    @ the same reference and the same flag to `Monitor.channelsIn` before anything else
    @ happens, so what the Monitor computes cannot depend on the retrainer: a
    @ retrainer that is absent, slow or dead changes nothing here. The copy goes out
    @ second, on `sampleCopy`, to the hub's serial port -- serialized values only, the
    @ D70 rule for crossing a process boundary. A tap with `sampleCopy` unconnected is
    @ a wire.
    passive component SampleTap {

        @ From the adapter
        sync input port sampleIn: ChannelSample

        @ To Monitor.channelsIn, unchanged
        output port sampleOut: ChannelSample

        @ To the hub, for the retrainer's process
        output port sampleCopy: SentinelTappedSample

        @ Samples forwarded
        telemetry Forwarded: U32 update on change

        @ Port for requesting the current time
        time get port timeCaller

        @ Enables telemetry channels handling
        import Fw.Channel
    }
}
