# Sentinel::SampleTap -- the telemetry path to the retrainer

**Exported OPT-IN with the retrainer, OFF by default** (`docs/DECISIONS.md` D85). With
`SENTINEL_WITH_RETRAINER` OFF it is not registered, and a mission's detector input is
exactly what it was.

It sits between a mission's adapter and `Sentinel.Monitor.channelsIn`. Each sample is
forwarded to the Monitor **first and unchanged** -- the same reference and the same
validity flag -- and only then copied, with a sequence number, onto `sampleCopy`, which a
topology wires to its hub's serial input port. The hub carries it to the retrainer's own
process as serialized values (D70 c.2: the retrainer never shares the detector's process).

**What the detector's work does not depend on.** The tap's work per sample is fixed and
connectionless: one forward, one hub call, one telemetry write. Whether a retrainer is
listening, slow or absent changes nothing the Monitor computes.

It declares no command and no parameter.
