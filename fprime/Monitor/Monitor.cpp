// ======================================================================
// \title  Monitor.cpp
// \brief  Cross-channel telemetry monitor: the work item 8 core, wrapped
// ======================================================================

#include "Monitor/Monitor.hpp"

#include "Monitor/FppConstantsAc.hpp"

#include <cstdio>

#include "Os/File.hpp"

namespace Sentinel {

namespace {

//! The FPP model and the core must agree about how wide an instance can be.
//! FPP owns the ground-facing side (CPP-23), Config.hpp owns the buffers.
static_assert(static_cast<U32>(Sentinel::MAX_CHANNELS) == Config::MAX_CHANNELS,
              "Monitor.fpp's MAX_CHANNELS must equal Config::MAX_CHANNELS");

//! The FPP enum is the ground's view of the core's LoadStatus, so the two must
//! not drift. tests/test_flight_build.py already pins the C++ enum against the
//! Python one; this pins the FPP enum against the C++ one, closing the ring.
static_assert(static_cast<U8>(ModelLoadStatus::T::OK) ==
                  static_cast<U8>(LoadStatus::OK), "OK");
static_assert(static_cast<U8>(ModelLoadStatus::T::BAD_MAGIC) ==
                  static_cast<U8>(LoadStatus::BAD_MAGIC), "BAD_MAGIC");
static_assert(static_cast<U8>(ModelLoadStatus::T::BAD_VERSION) ==
                  static_cast<U8>(LoadStatus::BAD_VERSION), "BAD_VERSION");
static_assert(static_cast<U8>(ModelLoadStatus::T::BAD_HEADER_CRC) ==
                  static_cast<U8>(LoadStatus::BAD_HEADER_CRC), "BAD_HEADER_CRC");
static_assert(static_cast<U8>(ModelLoadStatus::T::BAD_STATIC_CRC) ==
                  static_cast<U8>(LoadStatus::BAD_STATIC_CRC), "BAD_STATIC_CRC");
static_assert(static_cast<U8>(ModelLoadStatus::T::BAD_PARAM_CRC) ==
                  static_cast<U8>(LoadStatus::BAD_PARAM_CRC), "BAD_PARAM_CRC");
static_assert(static_cast<U8>(ModelLoadStatus::T::BAD_ARCH) ==
                  static_cast<U8>(LoadStatus::BAD_ARCH), "BAD_ARCH");
static_assert(static_cast<U8>(ModelLoadStatus::T::BAD_GATE_ORDER) ==
                  static_cast<U8>(LoadStatus::BAD_GATE_ORDER), "BAD_GATE_ORDER");
static_assert(static_cast<U8>(ModelLoadStatus::T::BAD_SHAPE) ==
                  static_cast<U8>(LoadStatus::BAD_SHAPE), "BAD_SHAPE");
static_assert(static_cast<U8>(ModelLoadStatus::T::TOO_LARGE) ==
                  static_cast<U8>(LoadStatus::TOO_LARGE), "TOO_LARGE");
static_assert(static_cast<U8>(ModelLoadStatus::T::TRUNCATED) ==
                  static_cast<U8>(LoadStatus::TRUNCATED), "TRUNCATED");
static_assert(static_cast<U8>(ModelLoadStatus::T::BAD_NORM_POLICY) ==
                  static_cast<U8>(LoadStatus::BAD_NORM_POLICY), "BAD_NORM_POLICY");

ModelLoadStatus::T toFpp(LoadStatus status) {
    // A plain cast would be a silent bet on the two enums agreeing; the
    // static_asserts above make it a checked one.
    return static_cast<ModelLoadStatus::T>(static_cast<U8>(status));
}

}  // namespace

// ----------------------------------------------------------------------
// Component construction and destruction
// ----------------------------------------------------------------------

Monitor ::Monitor(const char* const compName)
    : MonitorComponentBase(compName),
      m_detector(),
      m_baseline(),
      m_fileBuffer(),
      m_modelPath(""),
      m_sample(),
      m_sampleValid(false),
      m_sampleFresh(false),
      m_mode(Mode::BASELINE),
      m_status(ModelLoadStatus::NOT_LOADED),
      m_degradeReason(DegradeReason::NO_MODEL_FILE),
      m_channels(0U),
      m_ticks(0U),
      m_configured(false),
      m_degraded(false),
      m_announcedWarm(false) {
    // Level 1 from the first instruction: before any file is read, the baseline
    // is what runs. A component that started in MODEL mode and fell back would
    // have a window in which it was neither.
    for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
        m_sample[c] = 0.0F;
    }
    for (U32 i = 0U; i < MODEL_FILE_MAX_BYTES; ++i) {
        m_fileBuffer[i] = 0U;
    }
}

Monitor ::~Monitor() {}

// ----------------------------------------------------------------------
// Configuration and loading
// ----------------------------------------------------------------------

void Monitor ::configure(const char* modelPath, U32 nChannels) {
    m_modelPath = (modelPath != nullptr) ? modelPath : "";
    m_channels = (nChannels <= Config::MAX_CHANNELS) ? nChannels : 0U;
    m_baseline.configure(m_channels);
    m_configured = true;
}

ModelLoadStatus Monitor ::loadModel() {
    m_status = ModelLoadStatus::NOT_LOADED;

    Os::File file;
    const Os::File::Status opened = file.open(m_modelPath.toChar(), Os::File::OPEN_READ);
    if (opened != Os::File::OP_OK) {
        // No file at all is not a refusal code, it is an absence. Level 1 is
        // already what is running, and it says so rather than inventing a code.
        this->degrade(DegradeReason::NO_MODEL_FILE, ModelLoadStatus::NOT_LOADED);
        return m_status;
    }

    FwSizeType wanted = static_cast<FwSizeType>(MODEL_FILE_MAX_BYTES);
    const Os::File::Status readStatus = file.read(m_fileBuffer, wanted);
    file.close();

    if (readStatus != Os::File::OP_OK) {
        this->degrade(DegradeReason::NO_MODEL_FILE, ModelLoadStatus::NOT_LOADED);
        return m_status;
    }

    const U32 bytesRead = static_cast<U32>(wanted);
    const LoadStatus verdict = m_detector.load(m_fileBuffer, bytesRead);
    m_status = ModelLoadStatus(toFpp(verdict));

    if (verdict != LoadStatus::OK) {
        // The core leaves the model unusable and step() does nothing, so the
        // forecaster cannot warn. The topology still gets a detector.
        this->log_WARNING_HI_ModelRefused(m_status, bytesRead);
        this->degrade(DegradeReason::MODEL_REFUSED, m_status);
        return m_status;
    }

    const Model& model = m_detector.model();
    Fw::String provenance;
    // The block is NUL-terminated by the loader's own BAD_SHAPE check, so this
    // cannot run off the end.
    provenance = model.provenance;
    this->log_ACTIVITY_HI_ModelLoaded(model.nChannels, model.nLayers, model.tier,
                                      provenance);

    // The width the file declares wins over the width the topology guessed: the
    // weights only make sense at their own shape.
    m_channels = model.nChannels;
    m_baseline.configure(m_channels);
    this->refreshBaselineParameters();

    if (model.baselineOnly != 0U) {
        // Level 1 by request rather than by failure (Objective.md 14.10). The
        // file is good; it says do not run the network.
        this->degrade(DegradeReason::BASELINE_ONLY_SET, m_status);
        return m_status;
    }

    m_mode = Mode::MODEL;
    m_degraded = false;
    m_announcedWarm = false;
    m_ticks = 0U;
    return m_status;
}

void Monitor ::degrade(DegradeReason reason, ModelLoadStatus status) {
    const bool changed = (!m_degraded) || (m_degradeReason != reason.e);
    m_mode = Mode::BASELINE;
    m_degradeReason = reason.e;
    m_degraded = true;
    m_announcedWarm = false;
    m_ticks = 0U;
    m_baseline.reset();
    this->refreshBaselineParameters();
    if (changed) {
        this->log_WARNING_HI_DegradedToBaseline(reason, status);
    }
}

void Monitor ::refreshBaselineParameters() {
    Fw::ParamValid scaleValid = Fw::ParamValid::INVALID;
    Fw::ParamValid cutValid = Fw::ParamValid::INVALID;
    const ChannelScale scale = this->paramGet_BASELINE_SCALE(scaleValid);
    const F64 cut = this->paramGet_BASELINE_THRESHOLD(cutValid);

    // A parameter that has never been set reads as its FPP default and reports
    // INVALID; the default is deliberately usable, because Level 1 must run with
    // no parameter database at all (D34). So the value is taken either way and
    // the validity is not a gate.
    F64 values[Config::MAX_CHANNELS];
    for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
        values[c] = scale[static_cast<FwSizeType>(c)];
    }
    m_baseline.setScale(values, Config::MAX_CHANNELS);
    m_baseline.setThreshold(cut);
}

void Monitor ::describeChannel(U32 index, U32& id, Fw::String& name) const {
    if (m_detector.model().loaded() && (index < m_detector.model().nChannels)) {
        id = m_detector.model().channelId[index];
        name = m_detector.model().channelName[index];
        return;
    }
    // No model file, so no channel map. The index is the only honest name.
    char synthesised[Config::CHANNEL_NAME_BYTES];
    (void)snprintf(synthesised, sizeof(synthesised), "ch%u", index);
    id = index;
    name = synthesised;
}

// ----------------------------------------------------------------------
// Handler implementations for typed input ports
// ----------------------------------------------------------------------

void Monitor ::channelsIn_handler(FwIndexType portNum, Sentinel::ChannelVector& values,
                                  bool valid) {
    (void)portNum;
    for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
        m_sample[c] = values[static_cast<FwSizeType>(c)];
    }
    m_sampleValid = valid;
    m_sampleFresh = true;
}

void Monitor ::schedIn_handler(FwIndexType portNum, U32 context) {
    (void)portNum;
    (void)context;

    if (!m_configured || (m_channels == 0U)) {
        // Unconfigured is silent, not fatal. The topology still ticks.
        return;
    }

    // A tick with no sample since the last one runs anyway, with valid false,
    // which scores negative infinity and cannot alarm. Upstream falling silent
    // must not stop the detector's clock, or a later warning would be measured
    // from the wrong history.
    const bool valid = m_sampleValid && m_sampleFresh;
    m_sampleFresh = false;

    F32 score = 0.0F;
    F64 threshold = 0.0;
    bool emitted = false;
    U32 peak = 0U;
    bool warmed = false;

    if (m_mode == Mode::MODEL) {
        m_detector.step(m_sample, valid);
        score = m_detector.score();
        threshold = m_detector.model().threshold;
        emitted = m_detector.emitted();
        warmed = m_detector.steps() > static_cast<U64>(m_detector.model().warmupSteps);
        // The core reduces into a local and discards the index
        // (Detector.cpp:100-106), so the argmax is taken here, with the same
        // strict `>` so a tie keeps the lowest index exactly as the core does.
        const F32* smoothed = m_detector.smoothed();
        F32 largest = smoothed[0];
        for (U32 c = 1U; c < m_channels; ++c) {
            if (smoothed[c] > largest) {
                largest = smoothed[c];
                peak = c;
            }
        }
    } else {
        m_baseline.step(m_sample, valid);
        score = static_cast<F32>(m_baseline.score());
        threshold = m_baseline.threshold();
        emitted = m_baseline.emitted();
        warmed = m_baseline.warmed();
        peak = m_baseline.peakChannel();
    }

    if (warmed) {
        m_ticks += 1U;
        if (!m_announcedWarm) {
            m_announcedWarm = true;
            this->log_ACTIVITY_HI_WarmupComplete(m_ticks, m_mode);
        }
    }

    this->emitTelemetry(score, threshold);

    if (emitted) {
        U32 id = 0U;
        Fw::String name;
        this->describeChannel(peak, id, name);
        this->log_WARNING_HI_CrossChannelWarning(id, name, score, threshold);
    }
}

void Monitor ::emitTelemetry(F32 score, F64 threshold) {
    this->tlmWrite_Score(score);
    this->tlmWrite_Threshold(threshold);
    this->tlmWrite_ActiveMode(m_mode);
    this->tlmWrite_TicksSinceWarmup(m_ticks);
    this->tlmWrite_LoadStatus(m_status);
}

}  // namespace Sentinel
