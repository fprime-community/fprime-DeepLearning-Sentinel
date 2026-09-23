// ======================================================================
// \title  Monitor.cpp
// \brief  Cross-channel telemetry monitor: the work item 8 core, wrapped
// ======================================================================

#include "Sentinel/Monitor/Monitor.hpp"

#include "Sentinel/Monitor/FppConstantsAc.hpp"

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
static_assert(static_cast<U8>(ModelLoadStatus::T::BAD_PARAM_VERSION) ==
                  static_cast<U8>(LoadStatus::BAD_PARAM_VERSION), "BAD_PARAM_VERSION");

//! (!) The one the per-code asserts above cannot make: NOT_LOADED is this enum's
//! own sentinel and is NOT a core code, so nothing pairs it with a LoadStatus and
//! nothing noticed when D68 gave the core a twelfth code at the value NOT_LOADED
//! already held. `toFpp`'s cast is only checked where a pair exists. This asserts
//! the sentinel stays clear of the core's range, so the thirteenth code fails the
//! build instead of reporting a refusal to the ground as "never read".
static_assert(static_cast<U8>(ModelLoadStatus::T::NOT_LOADED) >
                  static_cast<U8>(LoadStatus::BAD_PARAM_VERSION),
              "NOT_LOADED collides with a core LoadStatus code");

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

void Monitor ::RELOAD_MODEL_cmdHandler(FwOpcodeType opCode, U32 cmdSeq,
                                       const Fw::CmdStringArg& modelPath) {
    // (!) THE PREVIOUS PATH IS SAVED BEFORE ANYTHING IS TRIED, and that is the
    // difference between this and topology setup. `loadModel` degrades to the
    // Level 1 baseline on any refusal, which is right at startup -- there is
    // nothing to fall back to. On a commanded reload it would mean a candidate
    // that does not load costs the operator the model that WAS working, and the
    // operator would learn that from a DegradedToBaseline event after the fact.
    //
    // `Objective.md` section 12 asks for "Previous model kept for rollback", and
    // this is the cheap form of it: no second Detector -- that would be another
    // 603,032 B in a component every mission links -- just the path, reloaded
    // from the file that is still on disk because the candidate was uplinked
    // beside it rather than over it.
    const Fw::String previous = m_modelPath;
    // (!) AND THE WIDTH THE TOPOLOGY IS ACTUALLY FEEDING, captured before the
    // load, because `loadModel` overwrites m_channels with the file's own.
    const U32 wired = m_channels;

    this->configure(modelPath.toChar(), m_channels);
    const ModelLoadStatus attempted = this->loadModel();

    if (attempted == ModelLoadStatus::OK) {
        const U32 offered = m_detector.model().nChannels;
        if (offered != wired) {
            // (!) IT LOADED AND IT IS STILL WRONG. The channel source was wired
            // by the topology and has not changed; a candidate declaring a
            // different width is a model for a different subsystem, and running
            // it would score whatever the unwired slots of the vector held.
            // Found by running the deployment, not by reading it.
            this->configure(previous.toChar(), wired);
            static_cast<void>(this->loadModel());
            this->log_WARNING_HI_ModelReloadWidthRefused(
                Fw::String(modelPath.toChar()), wired, offered);
            this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
            return;
        }
        this->log_ACTIVITY_HI_ModelReloadAccepted(Fw::String(modelPath.toChar()));
        this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
        return;
    }

    // Refused. ModelRefused and DegradedToBaseline have already named the code.
    // Put back what was running; if that fails too the component is on the
    // baseline and has already said so, and there is nothing further to try.
    this->configure(previous.toChar(), wired);
    static_cast<void>(this->loadModel());
    this->log_WARNING_HI_ModelReloadRolledBack(Fw::String(modelPath.toChar()),
                                               previous, attempted);

    // (!) THE COMMAND RESPONDS EXECUTION_ERROR, NOT OK. 65.4: "a candidate
    // refused by name is the handoff working" -- but it is not a reload that
    // happened, and a command sequencer that treats it as one would carry on to
    // the next step as though the model had changed.
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::EXECUTION_ERROR);
}

void Monitor ::degrade(const DegradeReason& reason, const ModelLoadStatus& status) {
    m_mode = Mode::BASELINE;
    m_degradeReason = reason.e;
    m_degraded = true;
    m_announcedWarm = false;
    m_ticks = 0U;
    m_baseline.reset();
    this->refreshBaselineParameters();

    // Emitted on every degrade, not only when the reason changes. This is
    // reached only from loadModel(), which is an explicit act -- topology setup
    // now, a commanded reload at work item 10 -- and never from the tick path,
    // so it cannot flood the event log. Suppressing a repeat would mean an
    // operator who commands a reload that fails again the same way is answered
    // with silence.
    this->log_WARNING_HI_DegradedToBaseline(reason, status);
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

    // (!) THE QUEUE IS DRAINED HERE, INSIDE THE TICK, AND THAT IS THE WHOLE
    // REASON THIS COMPONENT IS `queued` RATHER THAN `active`. D32 consequence 2
    // authorised it; F's own component-and-port-selection.md prescribes it for a
    // cyclic component that also accepts event-driven work, and `Svc::Health`
    // does the same thing in its own Run handler.
    //
    // Bounded by DISPATCH_DEPTH, so a burst of commands cannot turn one tick
    // into an unbounded amount of work -- Objective.md 11 rule 5 is fixed
    // compute per cycle, and a command path is not an exemption from it.
    //
    // It runs BEFORE the unconfigured check below on purpose: a component that
    // was never given a model path must still be able to accept the command that
    // gives it one.
    for (FwSizeType i = 0U; i < DISPATCH_DEPTH; ++i) {
        const MsgDispatchStatus status = this->doDispatch();
        if (status == MSG_DISPATCH_EMPTY) {
            break;
        }
        if (status != MSG_DISPATCH_OK) {
            break;
        }
    }

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
        // (!) THE WARNING NAMES THE CHANNEL THE RULE THAT FIRED PICKED, AND UNTIL
        // 2026-09-22 IT DID NOT. This scanned `smoothed()` -- the residual stream --
        // and that was CORRECT when it was written: `docs/MODELS.md` 20's row 8
        // records that the core "reduces into a local and discards" the index, so the
        // component took its own argmax at "zero core change". D68 then adopted the
        // fused rule, `max(z_residual, z_derivative)` (`flight/src/Detector.cpp`
        // `Detector::fused`), and the core gained `m_fusedChannel` with it -- an
        // argmax over the fused statistic, seeded from channel 0 for the reason
        // `Detector.cpp:168-176` gives. This scan was never updated, so whenever the
        // DERIVATIVE term was the one that crossed, the event named whichever channel
        // had the largest residual: a real channel, a plausible number, and the wrong
        // answer. It never moved a flag -- `m_emitted` is the core's -- which is why
        // nothing caught it.
        //
        // So the source of the index now follows the statistic the cut was applied
        // to, which `param_version` already names and the loader already refuses to
        // mismatch (`ModelFile.cpp:232-234`). This is a REPORTING fix: the detection
        // rule, `param_version` and the model format are untouched, and which ticks
        // warn is unchanged.
        if (m_detector.model().paramVersion == Format::PARAM_VERSION_FUSED) {
            peak = m_detector.fusedChannel();
        } else {
            // `param_version` 1 cuts the smoothed residual itself, so the residual
            // argmax IS the rule that fired. Same strict `>` as the core, so a tie
            // keeps the lowest index exactly as it does.
            const F32* smoothed = m_detector.smoothed();
            F32 largest = smoothed[0];
            for (U32 c = 1U; c < m_channels; ++c) {
                if (smoothed[c] > largest) {
                    largest = smoothed[c];
                    peak = c;
                }
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
