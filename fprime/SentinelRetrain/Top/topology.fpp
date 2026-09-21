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
    instance hub
    instance hubAdapter
    instance hubComm
    instance hubBufferManager

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

  # (!) THE HUB IS WIRED EXPLICITLY AND `event connections instance hub` IS NOT
  # USED, WHICH IS WHAT Svc/GenericHub/docs/sdd.md:260 RECOMMENDS.
  #
  # Under that pattern EVERY component's events go to hub.eventIn, the adapter
  # included. ByteStreamBufferAdapter.cpp:31 logs DriverNotReady when the driver
  # is not ready, so the event would go hub.eventIn -> send_data ->
  # toBufferDriver -> hubAdapter.bufferIn_handler -> still not ready -> log
  # again. Every component on that path is passive and every port is sync, so it
  # is direct recursion on one stack, and the driver IS not-ready at startup.
  #
  # So the retrainer's two ports are wired by hand and the adapter's and pool's
  # event ports are left unconnected. That is safe because every autocoded
  # emission is isConnected-guarded.

    connections Hub {
      retrainer.logOut -> hub.eventIn
      retrainer.tlmOut -> hub.tlmIn
    }

    connections HubTransport {
      # hub <-> adapter: PassiveBufferDriverClient against PassiveBufferDriver
      hub.toBufferDriver -> hubAdapter.bufferIn
      hubAdapter.bufferInReturn -> hub.toBufferDriverReturn
      hubAdapter.bufferOut -> hub.fromBufferDriver
      hub.fromBufferDriverReturn -> hubAdapter.bufferOutReturn

      # adapter <-> byte stream driver
      hubAdapter.toByteStreamDriver -> hubComm.$send
      hubComm.ready -> hubAdapter.byteStreamDriverReady
      hubComm.$recv -> hubAdapter.fromByteStreamDriver
      hubAdapter.fromByteStreamDriverReturn -> hubComm.recvReturnIn

      # allocation. Both the hub and the driver allocate; one pool serves both.
      hub.allocate -> hubBufferManager.bufferGetCallee
      hub.deallocate -> hubBufferManager.bufferSendIn
      hubComm.allocate -> hubBufferManager.bufferGetCallee
      hubComm.deallocate -> hubBufferManager.bufferSendIn
    }

  }

}
