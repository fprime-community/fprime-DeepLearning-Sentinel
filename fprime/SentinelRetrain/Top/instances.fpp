module SentinelRetrain {

  # ----------------------------------------------------------------------
  # Base ID Convention
  # ----------------------------------------------------------------------
  #
  # 0xDSSCCxxx, as SentinelRef/Top/instances.fpp sets out, with D = 3.
  #
  # (!) D = 2 IS NOT AVAILABLE. SentinelRef already places sentinelMonitor at
  # 0x20000000 and powerSim at 0x21000000, and the two deployments' dictionaries
  # are merged for the ground system, where a duplicated id is refused outright
  # by fprime-merge-dictionary. Taking D = 3 is what makes that merge the cheapest
  # possible proof that these ids do not collide.

  module Default {
    constant QUEUE_SIZE = 10
    constant STACK_SIZE = 64 * 1024
  }

  # ----------------------------------------------------------------------
  # Active component instances
  # ----------------------------------------------------------------------

  @ The 1 Hz rate group that ticks the retrainer.
  @ (!) ACTIVE, not Svc.PassiveRateGroup, and the reason is the command path:
  @ PassiveRateGroup declares command recv/resp/reg ports and a CLEAR_STATISTICS
  @ command, which would drag a CommandDispatcher into a deployment that needs
  @ none. ActiveRateGroup has no command ports.
  instance rateGroup_1Hz: Svc.ActiveRateGroup base id 0x30001000 \
    queue size Default.QUEUE_SIZE \
    stack size Default.STACK_SIZE \
    priority 43

  # ----------------------------------------------------------------------
  # Passive component instances
  # ----------------------------------------------------------------------

  instance chronoTime: Svc.ChronoTime base id 0x30010000

  instance rateGroupDriver: Svc.RateGroupDriver base id 0x30011000

  instance timer: Svc.LinuxTimer base id 0x30012000

  @ The local witness. It prints every component's text events to stdout, which
  @ is what separates "the retrainer did not tick" from "the tick did not cross"
  @ once a hub exists. One instance, and it earns it.
  instance textLogger: Svc.PassiveTextLogger base id 0x30013000

  @ E1's pipe and 61's float32 cycle, in its OWN process and its OWN deployment.
  @
  @ (!) THIS IS WHAT 47.15b NAMED AS C5's DISCHARGER: "a minimal retrainer
  @ deployment -- its own Top/, a topology carrying the Retrainer and a rate group
  @ and nothing else, per D70 consequence 2". Stop 32 as drafted at 61.5 forbade
  @ any topology; 61.5a narrows it to topologies carrying the Monitor or sharing
  @ a process with it, because all three of 47.15b's grounds are about the
  @ DETECTOR's process and none reaches this one.
  @
  @ (!) AND READING (b) STAYS REJECTED. SentinelRef does not name Retrainer, and
  @ tests/test_detector_binary_has_no_ocaml_runtime.py asserts by symbol that its
  @ binary carries no OCaml runtime. Nobody may cite this instance as evidence
  @ that instancing into the detector's topology became safe.
  instance retrainer: Retrain.Retrainer base id 0x30000000

}
