module SentinelRetrain {

  deployment topology SentinelRetrain {

  # ----------------------------------------------------------------------
  # Instances used in the topology
  # ----------------------------------------------------------------------
    instance chronoTime
    instance rateGroupDriver
    instance rateGroup_1Hz
    instance timer
    instance textLogger
    instance retrainer

  # ----------------------------------------------------------------------
  # Pattern graph specifiers
  # ----------------------------------------------------------------------
  #
  # (!) THERE IS NO `event connections` AND NO `telemetry connections` HERE, AND
  # THAT IS DELIBERATE RATHER THAN UNFINISHED. This deployment carries no
  # EventManager and no telemetry database: the retrainer's events and channels
  # are destined for the hub, and a hub is a transport rather than an event
  # source (hub-pattern.md:64-66). Until the hub exists, `retrainer.logOut` and
  # `retrainer.tlmOut` stay unconnected, which is safe because every autocoded
  # emission is guarded -- see isConnected_logOut_OutputPort and
  # isConnected_timeCaller_OutputPort in RetrainerComponentAc.cpp.
  #
  # The text-event path IS wired, because it is this deployment's only local
  # witness and step 5 of the stage needs it to tell a payload fault from a
  # crossing fault.

    text event connections instance textLogger
    time connections instance chronoTime

  # ----------------------------------------------------------------------
  # Direct graph specifiers
  # ----------------------------------------------------------------------

    connections RateGroups {
      timer.CycleOut -> rateGroupDriver.CycleIn
      rateGroupDriver.CycleOut[0] -> rateGroup_1Hz.CycleIn
      rateGroup_1Hz.RateGroupMemberOut[0] -> retrainer.schedIn
    }

  }

}
