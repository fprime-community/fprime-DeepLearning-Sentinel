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

  # ----------------------------------------------------------------------
  # The hub, and the three instances it cannot work without
  # ----------------------------------------------------------------------
  #
  # (!) FOUR INSTANCES, AND EACH ONE IS FORCED. hub-pattern.md:67-69 requires a
  # buffer driver at each end of the transport, and ByteStreamBufferAdapter is
  # what pairs a byte-stream driver with the buffer-driver interface GenericHub
  # expects. TcpServer.fpp:7 and TcpClient.fpp:7 declare `allocate` as an
  # Fw.BufferGet OUTPUT port, so it must terminate on a component --
  # Fw::MallocAllocator cannot substitute for Svc.BufferManager, it is the
  # BACKING allocator passed to BufferManager::setup.
  #
  # (!) A DEDICATED POOL, NOT A SHARED ONE. This deployment has no other pool,
  # but the reasoning is recorded here because SentinelRef does: a hub that
  # exhausts the pool must not starve the downlink, and the downlink is how the
  # hub is observed.
  #
  # GenericHubCfg.fpp:9-12 defaults every port array to 10, so no config
  # override is needed.

  instance hub: Svc.GenericHub base id 0x30014000

  instance hubAdapter: Drv.ByteStreamBufferAdapter base id 0x30015000

  @ (!) Drv.Udp, NOT Drv.TcpClient, and docs/MODELS.md 65.4 registered this route
  @ BEFORE the number that forced it. 70.6 measured 0 of 355 ticks delivering all
  @ five channels and the event over TCP: the hub emits six messages back to back
  @ and TCP coalesces them into one segment, which GenericHub.cpp:129-130 rejects
  @ on its exact-size check and :249-252 drops silently. UDP preserves message
  @ boundaries, which is the property TCP lacks. Stop 43 forbids this substitution
  @ for any reason other than HB2b's band, and HB2b's band is the reason.
  @
  @ Drv.Udp imports the same ByteStreamDriver interface as Drv.TcpClient
  @ (Drv/Interfaces/ByteStreamDriver.fpp:6-19), so the hub and adapter wiring is
  @ untouched and ONLY the driver instance moves.
  instance hubComm: Drv.Udp base id 0x30016000 \
  {
    phase Fpp.ToCpp.Phases.configComponents """
    if (state.hubPort != 0) {
        (void) SentinelRetrain::hubComm.configureSend(state.hubHostname, state.hubPort);
        (void) SentinelRetrain::hubComm.configureRecv("0.0.0.0",
                   static_cast<U16>(state.hubPort + 1));
    }
    """

    phase Fpp.ToCpp.Phases.startTasks """
    if (state.hubPort != 0) {
        Os::TaskString hubTaskName("HubRecv");
        SentinelRetrain::hubComm.start(hubTaskName, 40, 64 * 1024);
    }
    """

    phase Fpp.ToCpp.Phases.stopTasks """
    SentinelRetrain::hubComm.stop();
    """

    phase Fpp.ToCpp.Phases.freeThreads """
    (void) SentinelRetrain::hubComm.join();
    """
  }

  instance hubBufferManager: Svc.BufferManager base id 0x30017000 \
  {
    phase Fpp.ToCpp.Phases.configObjects """
    Svc::BufferManager::BufferBins bins;
    """

    phase Fpp.ToCpp.Phases.configComponents """
    memset(&ConfigObjects::SentinelRetrain_hubBufferManager::bins, 0,
           sizeof(ConfigObjects::SentinelRetrain_hubBufferManager::bins));
    ConfigObjects::SentinelRetrain_hubBufferManager::bins.bins[0].bufferSize = 2048;
    ConfigObjects::SentinelRetrain_hubBufferManager::bins.bins[0].numBuffers = 40;
    SentinelRetrain::hubBufferManager.setup(
        0, 0,
        SentinelRetrain::hubAllocator,
        ConfigObjects::SentinelRetrain_hubBufferManager::bins);
    """
  }

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
  instance retrainer: Retrain.Retrainer base id 0x30000000 \
  {
    phase Fpp.ToCpp.Phases.configComponents """
    // A crossing measurement needs more than the six ticks RETRAINER_CAPACITY
    // allows, and `init` refuses a second call so the accumulator cannot be
    // reset per tick. This raises the bound before boot; it allocates nothing.
    SentinelRetrain::retrainer.configure(4096);

    // 72 / E5-e, HO1: where the flying model is read from and where the
    // candidate is written. Both sit beside the binary, which is the shared
    // filesystem path SentinelRef's FileDownlink reads the candidate from --
    // both processes are on one host, which is what a real spacecraft looks
    // like, and it is why this deployment needs no Com stack of its own.
    //
    // (!) RetrainModel.bin IS NOT SentinelRef's SentinelModel.bin. The cycle is
    // maxima-shaped -- 16 inputs, 75,360 parameters (deep_f32.ml:28-34) -- and
    // SentinelRef flies an 8-channel model with 66,960 weights, so no candidate
    // this process builds can replace that one. docs/MODELS.md 72 reports that
    // rather than hiding it, and 72.9 carries it as owed. The file is built by
    // scripts/s72_flying_file.py and is a run artifact, cited by path.
    //
    // Absent, the component still ticks and still cycles; it produces no
    // candidate and says so once. Degrade rather than die.
    SentinelRetrain::retrainer.configureShadow("RetrainModel.bin",
                                               "RetrainCandidate.bin");
    """
  }

}
