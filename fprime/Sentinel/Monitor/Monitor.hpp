// ======================================================================
// \title  Monitor.hpp
// \brief  Cross-channel telemetry monitor: the work item 8 core, wrapped
//
// Warn-only. This component has no commanding port of any kind and issues no
// command (Objective.md 11 rule 3). The command ports it does carry are the ones
// F' requires of any component declaring parameters -- the parameter protocol is
// implemented as the autocoded PARAM_SET and PARAM_SAVE commands.
//
// Level 1, the safe failure mode (D5, Objective.md 14.10): on any loader refusal
// or with model.bin's baseline_only set, the component runs the statistical
// baseline INSTEAD of the model, emits the degradation event with the refusal
// code, and goes on serving the topology. It never asserts on file contents
// (CPP-4: model.bin is uplinked), never throws (CPP-25), and never fails the
// topology.
// ======================================================================

#ifndef Sentinel_Monitor_HPP
#define Sentinel_Monitor_HPP

#include "Fw/Types/FileNameString.hpp"
#include "Sentinel/Monitor/MonitorComponentAc.hpp"
#include "sentinel/Baseline.hpp"
#include "sentinel/Detector.hpp"
#include "sentinel/ModelFile.hpp"

namespace Sentinel {

class Monitor final : public MonitorComponentBase {
  public:
    //! The largest model.bin the compile-time maxima admit, so the read is
    //! bounded before the header is trusted. 302,048 bytes at MAX_CHANNELS = 16
    //! and MAX_PARAMETERS = 75,360; the flown shape is 285,136.
    //!
    //! D36: the file is read whole rather than fed in chunks. The chunked reader
    //! docs/MODEL_FILE.md 8 anticipates is deferred, and the loader's check
    //! order is untouched so it stays droppable-in.
    static const U32 MODEL_FILE_MAX_BYTES =
        Format::HEADER_BYTES + (Format::CHANNEL_RECORD_BYTES * Config::MAX_CHANNELS) +
        (4U * Format::MAX_PARAMETERS) + Format::PARAM_FIXED_BYTES +
        (8U * Config::MAX_CHANNELS);

    explicit Monitor(const char* const compName);
    ~Monitor();

    //! Topology setup supplies the model file and the instance's channel width.
    //! The width is needed even when no model file exists, because Level 1 must
    //! run without one. Matches the shape v4.3.0 gives PrmDb's own configure().
    void configure(const char* modelPath, U32 nChannels);

    //! Read and verify the model file, then arm whichever detector the result
    //! allows. Safe to call before the topology is running; emits nothing until
    //! the event port is connected, so the topology calls it during setup.
    //! Returns the loader's verdict; never throws, never asserts.
    ModelLoadStatus loadModel();

    //! Which detector is running, for the topology and the tests.
    Mode activeMode() const { return m_mode; }

  private:
    // ----------------------------------------------------------------------
    // Handler implementations for typed input ports
    // ----------------------------------------------------------------------

    //! Latch the tick's channel vector. Does no work beyond copying.
    void channelsIn_handler(FwIndexType portNum, Sentinel::ChannelVector& values,
                            bool valid) override;

    //! One rate-group tick: step a detector, emit telemetry, warn if warranted.
    void schedIn_handler(FwIndexType portNum, U32 context) override;

    // ----------------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------------

    //! Enter Level 1, with an event that says why. Idempotent: re-entering the
    //! same mode for the same reason emits nothing, so a repeated refusal does
    //! not flood the event log.
    void degrade(DegradeReason reason, ModelLoadStatus status);

    //! Push the parameters into the baseline. Called after a parameter update
    //! and at load, never inside the tick's hot path.
    void refreshBaselineParameters();

    //! Name and id of a channel, from model.bin's CHANNELS block when a model is
    //! loaded and synthesised when it is not.
    void describeChannel(U32 index, U32& id, Fw::String& name) const;

    void emitTelemetry(F32 score, F64 threshold);

    // -- the two detectors, one of which is running ------------------------
    Detector m_detector;
    Baseline m_baseline;

    // -- the model file ----------------------------------------------------
    U8 m_fileBuffer[MODEL_FILE_MAX_BYTES];
    Fw::FileNameString m_modelPath;

    // -- the latched sample ------------------------------------------------
    F32 m_sample[Config::MAX_CHANNELS];
    bool m_sampleValid;
    bool m_sampleFresh;

    // -- state -------------------------------------------------------------
    Mode m_mode;
    ModelLoadStatus m_status;
    DegradeReason m_degradeReason;
    U32 m_channels;
    U32 m_ticks;
    bool m_configured;
    bool m_degraded;
    bool m_announcedWarm;
};

}  // namespace Sentinel

#endif
