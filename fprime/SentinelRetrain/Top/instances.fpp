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
  instance retrainer: Sentinel.Retrainer base id 0x30000000 \
  {
    phase Fpp.ToCpp.Phases.configComponents """
    // D85: the loop's constants, once, before boot (docs/MODELS.md 78.4).
    //   yellow bands   PowerSim's engineering dictionary (PowerSim.fpp), which
    //                  56's G-limit reads -- a mission gives its own
    //   guard          260 ticks either side of the training span
    //   budget         1 step per admitted tick      EXPERIMENTAL (D73's N is owed)
    //   schedule       6,550 admitted steps          EXPERIMENTAL
    //   nominal        the flying model's own held-out rate, from its pre-launch
    //                  report: 0.1046% for D85's balanced-plant model
    {
        Sentinel::LoopConfig cfg = {};
        const F32 low[8] = {-2.0F, -10.0F, 0.5F, 27.0F, -5.0F, -35.0F, 0.0F, 0.30F};
        const F32 high[8] = {300.0F, 10.0F, 9.5F, 32.0F, 40.0F, 30.0F, 1.0F, 1.0F};
        for (U32 c = 0U; c < 8U; ++c) {
            cfg.yellowLow[c] = low[c];
            cfg.yellowHigh[c] = high[c];
        }
        cfg.guard = 260U;
        cfg.budget = 1;
        cfg.schedule = 6550U;
        cfg.nominalPpm = 1046;
        (void) SentinelRetrain::retrainer.configureLoop(cfg);
    }

    // 72 / E5-e, HO1: where the flying model is read from and where the
    // candidate is written. Both sit beside the binary, which is the shared
    // filesystem path SentinelRef's FileDownlink reads the candidate from --
    // both processes are on one host, which is what a real spacecraft looks
    // like, and it is why this deployment needs no Com stack of its own.
    //
    // (!) RetrainModel.bin IS THE FLYING FILE AT THE SHAPE THIS BUILD WAS
    // GENERATED FOR, and not SentinelRef's own SentinelModel.bin. Until D84 the
    // cycle was maxima-shaped -- 16 inputs, 75,360 parameters -- and SentinelRef
    // flies 8 channels and 66,960 weights, so no candidate fitted it
    // (docs/MODELS.md 72.4). D84 generates the cycle at SENTINEL_RETRAINER_CHANNELS
    // and _PREDICTIONS; at 8 and 10 the candidate is SentinelRef's shape. The
    // file is built by `scripts/s72_flying_file.py --channels C --predictions P`,
    // seeded and NOT trained, and is a run artifact, cited by path. A mission
    // points this at its own flown model instead.
    //
    // Absent or refused, the component reports it once and stays inert. Degrade
    // rather than die.
    SentinelRetrain::retrainer.configureShadow("RetrainModel.bin",
                                               "RetrainCandidate.bin");
    """
  }

}
