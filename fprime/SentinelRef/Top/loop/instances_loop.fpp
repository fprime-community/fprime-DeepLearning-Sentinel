# D85: the one instance the retraining loop adds to the detector's deployment -- used
# only when SENTINEL_WITH_RETRAINER is ON (Top/CMakeLists.txt). OFF, this file is not
# an autocoder input and SentinelRef is exactly what it was.
module SentinelRef {

  @ The SampleTap: PowerSim's vector to the Monitor first and unchanged, then a copy
  @ over the hub to the retrainer's process (docs/DECISIONS.md D85).
  instance sampleTap: Sentinel.SampleTap base id 0x22000000

  @ LC2's detector-side log (docs/MODELS.md 78.11): apparatus, silent unless -E. It asks
  @ the Monitor's core what it emitted, through the read-only accessor, after the Monitor
  @ has stepped; the Monitor is not touched.
  instance emitProbe: EmitTap.EmitProbe base id 0x22001000 \
  {
    phase Fpp.ToCpp.Phases.configComponents """
    SentinelRef::emitProbe.configure(&SentinelRef::sentinelMonitor, &SentinelRef::sampleTap,
                                     state.emitLog);
    """
  }

}
