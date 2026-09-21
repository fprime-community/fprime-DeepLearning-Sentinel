module SentinelRef {

  # ----------------------------------------------------------------------
  # Base ID Convention
  # ----------------------------------------------------------------------
  #
  # All Base IDs follow the 8-digit hex format: 0xDSSCCxxx
  #
  # Where:
  #   D   = Deployment digit (1 for this deployment)
  #   SS  = Subtopology digits (00 for main topology, 01-05 for subtopologies)
  #   CC  = Component digits (00, 01, 02, etc.)
  #   xxx = Reserved for internal component items (events, commands, telemetry)
  #

  # ----------------------------------------------------------------------
  # Defaults
  # ----------------------------------------------------------------------

  module Default {
    constant QUEUE_SIZE = 10
    constant STACK_SIZE = 64 * 1024
  }

  # ----------------------------------------------------------------------
  # Active component instances
  # ----------------------------------------------------------------------

  # 1Hz rate group (divisor 1 of 1Hz base clock)
  instance rateGroup_1Hz: Svc.ActiveRateGroup base id 0x10001000 \
    queue size Default.QUEUE_SIZE \
    stack size Default.STACK_SIZE \
    priority 43

  # 0.5Hz rate group (divisor 2 of 1Hz base clock)
  instance rateGroup_0_5Hz: Svc.ActiveRateGroup base id 0x10002000 \
    queue size Default.QUEUE_SIZE \
    stack size Default.STACK_SIZE \
    priority 42

  # 0.25Hz rate group (divisor 4 of 1Hz base clock)
  instance rateGroup_0_25Hz: Svc.ActiveRateGroup base id 0x10003000 \
    queue size Default.QUEUE_SIZE \
    stack size Default.STACK_SIZE \
    priority 41

  instance cmdSeq: Svc.CmdSequencer base id 0x10004000 \
    queue size Default.QUEUE_SIZE \
    stack size Default.STACK_SIZE \
    priority 40

  # ----------------------------------------------------------------------
  # Queued component instances
  # ----------------------------------------------------------------------


  # ----------------------------------------------------------------------
  # Passive component instances
  # ----------------------------------------------------------------------

  instance chronoTime: Svc.ChronoTime base id 0x10010000

  instance rateGroupDriver: Svc.RateGroupDriver base id 0x10011000

  instance systemResources: Svc.SystemResources base id 0x10012000

  instance timer: Svc.LinuxTimer base id 0x10013000

  instance comDriver: Drv.TcpClient base id 0x10014000

  # ----------------------------------------------------------------------
  # The hub: this deployment is the SERVER end
  # ----------------------------------------------------------------------
  #
  # (!) THE DETECTOR LISTENS AND THE RETRAINER CONNECTS, not the other way
  # round. SentinelRef must come up and keep running whether or not a retrainer
  # exists; a client end would make the detector's startup depend on a process
  # D70 consequence 2 deliberately separated from it.
  #
  # (!) A DEDICATED POOL, NOT ComCcsds.commsBufferManager. A hub that exhausts
  # the pool must not starve the downlink, and the downlink is how the hub is
  # observed. Sharing one pool would make a crossing fault and a telemetry
  # fault indistinguishable.

  instance hub: Svc.GenericHub base id 0x10015000

  instance hubAdapter: Drv.ByteStreamBufferAdapter base id 0x10016000

  @ (!) Drv.Udp, NOT Drv.TcpServer. docs/MODELS.md 65.4 registered this route
  @ before the number that forced it, and stop 43 forbids it for any other
  @ reason. 70.6 measured 0 of 355 ticks delivering all five channels and the
  @ event over TCP, because TCP coalesces the six messages a tick emits and
  @ GenericHub.cpp:129-130 rejects the combined frame on an exact-size check.
  @ Only the driver instance moves: Drv.Udp imports the same ByteStreamDriver
  @ interface, so the hub and adapter wiring is unchanged.
  instance hubServer: Drv.Udp base id 0x10017000 \
  {
    phase Fpp.ToCpp.Phases.configComponents """
    if (state.hubPort != 0) {
        (void) SentinelRef::hubServer.configureRecv("0.0.0.0", state.hubPort);
        (void) SentinelRef::hubServer.configureSend("127.0.0.1",
                   static_cast<U16>(state.hubPort + 1));
    }
    """

    phase Fpp.ToCpp.Phases.startTasks """
    if (state.hubPort != 0) {
        Os::TaskString hubTaskName("HubRecv");
        SentinelRef::hubServer.start(hubTaskName, 40, 64 * 1024);
    }
    """

    phase Fpp.ToCpp.Phases.stopTasks """
    SentinelRef::hubServer.stop();
    """

    phase Fpp.ToCpp.Phases.freeThreads """
    (void) SentinelRef::hubServer.join();
    """
  }

  @ The counter docs/MODELS.md 70.6 owed. Passive and sync, so a count is taken
  @ in the hub's own dispatch and nothing can lose a message before it is
  @ counted. It forwards everything unchanged, so the ground sees exactly what
  @ it saw before this was inserted.
  instance hubCounter: HubTap.HubCounter base id 0x10019000

  instance hubBufferManager: Svc.BufferManager base id 0x10018000 \
  {
    phase Fpp.ToCpp.Phases.configObjects """
    Svc::BufferManager::BufferBins bins;
    """

    phase Fpp.ToCpp.Phases.configComponents """
    memset(&ConfigObjects::SentinelRef_hubBufferManager::bins, 0,
           sizeof(ConfigObjects::SentinelRef_hubBufferManager::bins));
    ConfigObjects::SentinelRef_hubBufferManager::bins.bins[0].bufferSize = 2048;
    ConfigObjects::SentinelRef_hubBufferManager::bins.bins[0].numBuffers = 40;
    SentinelRef::hubBufferManager.setup(
        1, 0, SentinelRef::hubAllocator,
        ConfigObjects::SentinelRef_hubBufferManager::bins);
    """
  }

  @ The Sentinel cross-channel monitor. Queued (D32 consequence 2), and the
  @ queue is drained inside the rate-group tick, so the cyclic work still runs
  @ in the context of the rate group that ticks it and a slip shows up as a
  @ slip. Only the reload command is asynchronous.
  @
  @ The queue size matches Monitor's DISPATCH_DEPTH: a tick drains at most what
  @ the queue can hold, so a burst of commands cannot make one tick unbounded.
  @
  @ Objective.md 10.2 fix 2: a topology may instantiate one per subsystem,
  @ each with its own model.bin and its own channel set, so the base id is
  @ given room above the system services rather than squeezed among them.
  instance sentinelMonitor: Sentinel.Monitor base id 0x20000000 \
    queue size 10

  @ The physics testbed's subsystem simulation (docs/MODELS.md 42). Passive, so
  @ its tick runs in the rate group's thread and the plant advances in lockstep
  @ with the detector watching it -- which is what lets a warning instant and a
  @ limit instant be compared at all.
  instance powerSim: Testbed.PowerSim base id 0x21000000

}
