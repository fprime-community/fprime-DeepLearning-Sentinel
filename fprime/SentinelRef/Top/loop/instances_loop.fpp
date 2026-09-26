# D85: the one instance the retraining loop adds to the detector's deployment -- used
# only when SENTINEL_WITH_RETRAINER is ON (Top/CMakeLists.txt). OFF, this file is not
# an autocoder input and SentinelRef is exactly what it was.
module SentinelRef {

  @ The SampleTap: PowerSim's vector to the Monitor first and unchanged, then a copy
  @ over the hub to the retrainer's process (docs/DECISIONS.md D85).
  instance sampleTap: Sentinel.SampleTap base id 0x22000000

}
