// ======================================================================
// \title  MonitorTester.cpp
// \brief  Test harness for Sentinel::Monitor
// ======================================================================

#include "MonitorTester.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstring>

#include "Sentinel/Monitor/FppConstantsAc.hpp"
#include "sentinel/Crc32.hpp"
#include "sentinel/ModelFile.hpp"

namespace Sentinel {

namespace {

//! Where the work item 8 golden vectors live. Supplied by CMake, because a unit
//! test's working directory is the build cache and not the source tree.
#ifndef SENTINEL_VECTORS_DIR
#define SENTINEL_VECTORS_DIR "."
#endif

const char* const G1_PATH = SENTINEL_VECTORS_DIR "/g1.bin";

//! A second REAL model, at a different width: g1 is 3 channels and g2 is 7.
//! Used only to offer the reload a valid file this topology is not wired for.
const char* const G2_PATH = SENTINEL_VECTORS_DIR "/g2.bin";
const char* const WIDE_PATH = "MonitorTester_wide.bin";

//! D68's flight configuration: 3 channels at `param_version` 2, the FUSED
//! statistic. g1 and g2 are version 1, where the cut is the smoothed residual
//! and the residual argmax IS the rule -- so only this tier can show the
//! difference between the two.
const char* const P1_PATH = SENTINEL_VECTORS_DIR "/p1.bin";

//! Put the process in the build cache, so the short relative paths above resolve
//! there rather than into the source tree. Called by every reload test before it
//! writes anything. Idempotent, and a failure is not fatal -- the test then
//! writes where it was launched, which is what the assertion below catches.
void moveToBuildCache() {
    (void)::chdir(SENTINEL_UT_TMP);
}

//! Written into the build cache, never into the source tree: a test that
//! dirties the tree is what tests/test_no_local_persistence.py forbids on the
//! Python side, and the rule is the same here.
#ifndef SENTINEL_UT_TMP
#define SENTINEL_UT_TMP "."
#endif
const char* const WORK_PATH = SENTINEL_UT_TMP "/MonitorTester_model.bin";

//! HO3's candidate lands BESIDE the flying model, never over it, which is what
//! makes the rollback in `RELOAD_MODEL_cmdHandler` possible at all.
//!
//! (!) RELATIVE, AND SHORT, AND THAT IS NOT TIDINESS. `Fw::CmdStringArg` holds
//! FW_CMD_STRING_MAX_SIZE characters and that is 40 in F's default
//! configuration, so the absolute build-cache path the other fixtures use --
//! 121 characters here -- is truncated at the command boundary and the reload
//! fails on a file that does not exist. The bound is the platform's, it is real
//! in flight too, and a mission's candidate has to live somewhere short.
//!
//! (!) AND A RELATIVE PATH LANDS WHEREVER THE TEST WAS LAUNCHED FROM, WHICH THE
//! FIRST VERSION OF THIS DID -- four .bin files in the source tree, swept into a
//! commit by `git add -A`. `moveToBuildCache()` below puts the process in the
//! build cache before any of them is written, so the path stays short AND the
//! tree stays clean. The other fixtures avoid this by using SENTINEL_UT_TMP
//! outright; they can, because they never go through a command.
const char* const CANDIDATE_PATH = "MonitorTester_candidate.bin";

}  // namespace

// C++14 odr-use: a static const member passed by reference to a gtest macro
// needs a definition, not just an in-class initialiser.
const FwSizeType MonitorTester::MAX_HISTORY_SIZE;
const FwEnumStoreType MonitorTester::TEST_INSTANCE_ID;
const U32 MonitorTester::G1_BYTES;
const U32 MonitorTester::G1_CHANNELS;

// ----------------------------------------------------------------------
// Construction and destruction
// ----------------------------------------------------------------------

MonitorTester ::MonitorTester()
    : MonitorGTestBase("MonitorTester", MonitorTester::MAX_HISTORY_SIZE),
      component("Monitor"),
      m_original(),
      m_working(),
      m_length(0U),
      m_haveVectors(false) {
    this->initComponents();
    this->connectPorts();
    this->loadPristine();
}

MonitorTester ::~MonitorTester() {
    this->component.deinit();
}

// ----------------------------------------------------------------------
// Model-file helpers
// ----------------------------------------------------------------------

void MonitorTester ::loadPristine() {
    std::FILE* handle = std::fopen(G1_PATH, "rb");
    if (handle == nullptr) {
        return;
    }
    const size_t read = std::fread(m_original, 1U, static_cast<size_t>(G1_BYTES), handle);
    (void)std::fclose(handle);
    m_length = static_cast<U32>(read);
    m_haveVectors = (m_length == G1_BYTES);
    this->restore();
}

void MonitorTester ::restore() {
    (void)std::memcpy(m_working, m_original, static_cast<size_t>(G1_BYTES));
    m_length = G1_BYTES;
}

void MonitorTester ::writeU16At(U32 offset, U32 value) {
    m_working[offset] = static_cast<U8>(value & 0xFFU);
    m_working[offset + 1U] = static_cast<U8>((value >> 8U) & 0xFFU);
}

void MonitorTester ::writeU32At(U32 offset, U32 value) {
    m_working[offset] = static_cast<U8>(value & 0xFFU);
    m_working[offset + 1U] = static_cast<U8>((value >> 8U) & 0xFFU);
    m_working[offset + 2U] = static_cast<U8>((value >> 16U) & 0xFFU);
    m_working[offset + 3U] = static_cast<U8>((value >> 24U) & 0xFFU);
}

U32 MonitorTester ::paramBlockOffset() const {
    return m_length - (Format::PARAM_FIXED_BYTES + (8U * G1_CHANNELS));
}

void MonitorTester ::resign() {
    this->writeU32At(Format::HEADER_CRC_OFFSET,
                     crc32(m_working, Format::HEADER_CRC_OFFSET));
}

void MonitorTester ::resignParams() {
    const U32 lo = this->paramBlockOffset();
    this->writeU32At(48U, crc32(m_working + lo, m_length - lo));
    this->resign();
}

Mode MonitorTester ::loadWorkingAs(const char* path) {
    std::FILE* handle = std::fopen(path, "wb");
    if (handle != nullptr) {
        (void)std::fwrite(m_working, 1U, static_cast<size_t>(m_length), handle);
        (void)std::fclose(handle);
    }
    this->component.configure(path, G1_CHANNELS);
    (void)this->component.loadModel();
    return this->component.activeMode();
}

void MonitorTester ::tickChannels(const F32* values, U32 n, bool valid) {
    ChannelVector v;
    for (U32 c = 0U; c < static_cast<U32>(MAX_CHANNELS); ++c) {
        v[static_cast<FwSizeType>(c)] = (c < n) ? values[c] : 0.0F;
    }
    this->invoke_to_channelsIn(0, v, valid);
    this->invoke_to_schedIn(0, 0U);
}

void MonitorTester ::tick(U32 count, F32 value, bool valid) {
    for (U32 i = 0U; i < count; ++i) {
        ChannelVector values;
        for (U32 c = 0U; c < static_cast<U32>(MAX_CHANNELS); ++c) {
            values[static_cast<FwSizeType>(c)] = value;
        }
        this->invoke_to_channelsIn(0, values, valid);
        this->invoke_to_schedIn(0, 0U);
    }
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

void MonitorTester ::testAGoodFileArmsTheModel() {
    ASSERT_TRUE(m_haveVectors) << "g1.bin not readable from " << G1_PATH;
    this->restore();
    ASSERT_EQ(Mode::MODEL, this->loadWorkingAs(WORK_PATH));
    ASSERT_EVENTS_ModelLoaded_SIZE(1);
    ASSERT_EVENTS_ModelLoaded(0, G1_CHANNELS, 2U, 3U,
                              "seeded G1, rng(1), U(-k,k) k=1/sqrt(H)");
    ASSERT_EVENTS_DegradedToBaseline_SIZE(0);
    ASSERT_EVENTS_ModelRefused_SIZE(0);
}

void MonitorTester ::testEveryRefusalCodeDegradesToTheBaseline() {
    ASSERT_TRUE(m_haveVectors);

    // The twelve refusal codes, each reached by the mutation
    // flight/test/RefusalTests.cpp uses for it. BAD_PARAM_VERSION was added to
    // Status.hpp by D68 and this loop was not extended with it, so the component
    // covered 11 of 12 from D68 until 2026-09-17. The gap was never registered as
    // owed on this branch; docs/MODELS.md 20.13 records it and this is its closure.
    struct Case {
        const char* what;
        ModelLoadStatus::T expected;
    };

    for (U32 index = 0U; index < 12U; ++index) {
        this->clearHistory();
        this->restore();
        Case c = {"", ModelLoadStatus::OK};

        switch (index) {
            case 0U:
                m_working[0] = 'X';
                c = {"bad magic", ModelLoadStatus::BAD_MAGIC};
                break;
            case 1U:
                this->writeU16At(4U, Format::FORMAT_VERSION + 1U);
                c = {"bad version", ModelLoadStatus::BAD_VERSION};
                break;
            case 2U:
                m_working[20] = static_cast<U8>(m_working[20] ^ 0xFFU);
                c = {"bad header CRC", ModelLoadStatus::BAD_HEADER_CRC};
                break;
            case 3U:
                m_working[Format::HEADER_BYTES + 40U] =
                    static_cast<U8>(m_working[Format::HEADER_BYTES + 40U] ^ 0xFFU);
                c = {"a weight bit-flip", ModelLoadStatus::BAD_STATIC_CRC};
                break;
            case 4U:
                m_working[m_length - 4U] =
                    static_cast<U8>(m_working[m_length - 4U] ^ 0xFFU);
                c = {"a threshold bit-flip", ModelLoadStatus::BAD_PARAM_CRC};
                break;
            case 5U:
                this->writeU16At(8U, 2U);
                this->resign();
                c = {"a non-GRU architecture", ModelLoadStatus::BAD_ARCH};
                break;
            case 6U:
                this->writeU16At(10U, 2U);
                this->resign();
                c = {"a wrong gate order", ModelLoadStatus::BAD_GATE_ORDER};
                break;
            case 7U:
                this->writeU32At(52U, 1U);
                this->resign();
                c = {"a non-zero reserved field", ModelLoadStatus::BAD_SHAPE};
                break;
            case 8U:
                this->writeU16At(14U, 20U);
                this->writeU16At(16U, 20U);
                this->resign();
                c = {"a model past the maxima", ModelLoadStatus::TOO_LARGE};
                break;
            case 9U:
                m_length = G1_BYTES - 1U;
                c = {"a truncated file", ModelLoadStatus::TRUNCATED};
                break;
            case 10U:
                this->writeU16At(this->paramBlockOffset() + 2U, 1U);
                this->resignParams();
                c = {"a non-identity normalisation policy",
                     ModelLoadStatus::BAD_NORM_POLICY};
                break;
            default:
                // param_version sits at PARAMS offset 0 (docs/MODEL_FILE.md 6.1).
                // A generation no reader knows, not a corrupt one: the block is
                // re-signed, so this reaches the version check rather than the CRC.
                this->writeU16At(this->paramBlockOffset(), 3U);
                this->resignParams();
                c = {"a param_version this reader does not know",
                     ModelLoadStatus::BAD_PARAM_VERSION};
                break;
        }

        const Mode mode = this->loadWorkingAs(WORK_PATH);

        // Level 1, all three halves of it: the baseline is running, the refusal
        // named its code, and the degradation said so.
        ASSERT_EQ(Mode::BASELINE, mode) << c.what;
        ASSERT_EVENTS_ModelRefused_SIZE(1);
        ASSERT_EVENTS_ModelRefused(0, ModelLoadStatus(c.expected), m_length);
        ASSERT_EVENTS_DegradedToBaseline_SIZE(1);
        ASSERT_EVENTS_DegradedToBaseline(0, DegradeReason::MODEL_REFUSED,
                                         ModelLoadStatus(c.expected));
    }
}

void MonitorTester ::testBaselineOnlyDegradesWithoutARefusal() {
    ASSERT_TRUE(m_haveVectors);
    this->restore();
    // baseline_only sits at offset 16 of the parameter block.
    m_working[this->paramBlockOffset() + 16U] = 1U;
    this->resignParams();

    ASSERT_EQ(Mode::BASELINE, this->loadWorkingAs(WORK_PATH));
    // The file is good, so it loaded; it simply says do not run the network.
    ASSERT_EVENTS_ModelLoaded_SIZE(1);
    ASSERT_EVENTS_ModelRefused_SIZE(0);
    ASSERT_EVENTS_DegradedToBaseline_SIZE(1);
    ASSERT_EVENTS_DegradedToBaseline(0, DegradeReason::BASELINE_ONLY_SET,
                                     ModelLoadStatus::OK);
}

void MonitorTester ::testAMissingFileDegrades() {
    this->component.configure("no-such-model-file.bin", G1_CHANNELS);
    (void)this->component.loadModel();
    ASSERT_EQ(Mode::BASELINE, this->component.activeMode());
    ASSERT_EVENTS_ModelRefused_SIZE(0);
    ASSERT_EVENTS_DegradedToBaseline_SIZE(1);
    ASSERT_EVENTS_DegradedToBaseline(0, DegradeReason::NO_MODEL_FILE,
                                     ModelLoadStatus::NOT_LOADED);
}

void MonitorTester ::testTheTopologyKeepsTickingAfterEveryRefusal() {
    ASSERT_TRUE(m_haveVectors);
    this->restore();
    m_working[0] = 'X';
    ASSERT_EQ(Mode::BASELINE, this->loadWorkingAs(WORK_PATH));

    this->clearHistory();
    this->tick(200U, 1.0F);
    // The component served 200 ticks after a refusal, which is what "never fail
    // the topology" means when it is measured rather than asserted.
    ASSERT_TLM_ActiveMode_SIZE(200);
    ASSERT_TLM_ActiveMode(199, Mode::BASELINE);
}

void MonitorTester ::testTelemetryUpdatesOnEveryTick() {
    ASSERT_TRUE(m_haveVectors);
    this->restore();
    ASSERT_EQ(Mode::MODEL, this->loadWorkingAs(WORK_PATH));
    this->clearHistory();
    this->tick(5U, 0.25F);
    ASSERT_TLM_Score_SIZE(5);
    ASSERT_TLM_Threshold_SIZE(5);
    ASSERT_TLM_ActiveMode_SIZE(5);
    ASSERT_TLM_TicksSinceWarmup_SIZE(5);
    ASSERT_TLM_LoadStatus_SIZE(5);
    ASSERT_TLM_ActiveMode(0, Mode::MODEL);
    ASSERT_TLM_LoadStatus(0, ModelLoadStatus::OK);
}

void MonitorTester ::testSilentUntilWarm() {
    ASSERT_TRUE(m_haveVectors);
    this->restore();
    // Drive the model's threshold below every possible score, so the only thing
    // that can keep the component quiet is the warm-up (Objective.md 11 rule 2).
    const U32 lo = this->paramBlockOffset();
    for (U32 i = 0U; i < 8U; ++i) {
        m_working[lo + 24U + i] = 0U;
    }
    m_working[lo + 24U + 7U] = 0xC0U;   // -2.0 as a little-endian F64
    this->resignParams();
    ASSERT_EQ(Mode::MODEL, this->loadWorkingAs(WORK_PATH));

    this->clearHistory();
    this->tick(8U, 0.5F);               // g1's warm-up is 8 ticks
    ASSERT_EVENTS_CrossChannelWarning_SIZE(0);
    this->tick(1U, 0.5F);
    ASSERT_EVENTS_WarmupComplete_SIZE(1);
    ASSERT_EVENTS_CrossChannelWarning_SIZE(1);
}

void MonitorTester ::testATickWithNoSampleCannotAlarm() {
    ASSERT_TRUE(m_haveVectors);
    this->restore();
    const U32 lo = this->paramBlockOffset();
    for (U32 i = 0U; i < 8U; ++i) {
        m_working[lo + 24U + i] = 0U;
    }
    m_working[lo + 24U + 7U] = 0xC0U;
    this->resignParams();
    ASSERT_EQ(Mode::MODEL, this->loadWorkingAs(WORK_PATH));
    this->tick(20U, 0.5F);

    this->clearHistory();
    // schedIn with no channelsIn since the last tick: valid is false, the score
    // is negative infinity, and nothing can cross.
    this->invoke_to_schedIn(0, 0U);
    ASSERT_EVENTS_CrossChannelWarning_SIZE(0);
    ASSERT_TLM_Score_SIZE(1);
}

void MonitorTester ::testTheWarningNamesTheChannelFromTheModelFile() {
    ASSERT_TRUE(m_haveVectors);
    this->restore();
    const U32 lo = this->paramBlockOffset();
    for (U32 i = 0U; i < 8U; ++i) {
        m_working[lo + 24U + i] = 0U;
    }
    m_working[lo + 24U + 7U] = 0xC0U;
    this->resignParams();
    ASSERT_EQ(Mode::MODEL, this->loadWorkingAs(WORK_PATH));

    this->tick(12U, 0.75F);
    ASSERT_GT(this->eventHistory_CrossChannelWarning->size(), 0U);
    // g1's CHANNELS block carries ids 1000..1002 and names channel_0..channel_2,
    // so the event must name one of them and not a synthesised index.
    const U32 id = this->eventHistory_CrossChannelWarning->at(0).channelId;
    ASSERT_GE(id, 1000U);
    ASSERT_LE(id, 1002U);
}

void MonitorTester ::testTheBaselineWarningNamesASynthesisedChannel() {
    // No model file at all, so no channel map: the index is the only honest name.
    this->component.configure("no-such-model-file.bin", G1_CHANNELS);
    (void)this->component.loadModel();
    ASSERT_EQ(Mode::BASELINE, this->component.activeMode());
    this->paramSet_BASELINE_THRESHOLD(-1.0, Fw::ParamValid::VALID);
    this->paramSend_BASELINE_THRESHOLD(0, 0);

    this->clearHistory();
    this->tick(Config::BASELINE_WINDOW + 2U, 1.0F);
    ASSERT_GT(this->eventHistory_CrossChannelWarning->size(), 0U);
    ASSERT_LT(this->eventHistory_CrossChannelWarning->at(0).channelId, G1_CHANNELS);
}


// ----------------------------------------------------------------------
// (!) The warning names the channel the rule that fired picked.
//
// This scanned `smoothed()` -- the residual stream -- which was CORRECT when it
// was written: 20's row 8 records that the core "reduces into a local and
// discards" the index, so the component took its own argmax at "zero core
// change". D68 then adopted `max(z_residual, z_derivative)` and the core gained
// `fusedChannel()` with it. The scan was never updated, so whenever the
// DERIVATIVE term was what crossed, the event named whichever channel had the
// largest residual: a real channel, a plausible number, the wrong answer, and no
// flag moved -- which is exactly why nothing caught it.
// ----------------------------------------------------------------------

void MonitorTester ::testTheWarningNamesTheFusedRulesChannel() {
    // p1 is `param_version` 2. g1 and g2 are version 1, where the cut IS the
    // smoothed residual and the two answers agree by construction, so only this
    // tier can tell the fix from the defect.
    this->component.configure(P1_PATH, 3U);
    if (this->component.loadModel() != ModelLoadStatus::OK) {
        GTEST_SKIP() << "p1.bin not readable from " << P1_PATH;
    }
    ASSERT_EQ(Mode::MODEL, this->component.activeMode());
    ASSERT_EQ(Format::PARAM_VERSION_FUSED,
              this->component.detector().model().paramVersion);

    const U32 warmup =
        static_cast<U32>(this->component.detector().model().warmupSteps) + 8U;
    F32 quiet[3] = {0.0F, 0.0F, 0.0F};
    for (U32 i = 0U; i < warmup; ++i) {
        this->tickChannels(quiet, 3U);
        // The warm-up is ~2,350 ticks and every one writes five telemetry
        // channels, so the tester's 512-entry history has to be drained as we go
        // or it asserts before the component can be judged.
        if ((i % 64U) == 0U) {
            this->clearHistory();
        }
    }
    this->clearHistory();

    // Channel 1 is held at a high steady level: a large RESIDUAL, and a
    // derivative that decays towards nothing because the level stops changing.
    // Channel 2 is stepped hard every other tick: a large DERIVATIVE, on a level
    // the forecaster tracks better. That is the shape that separates the two
    // argmaxes, and the assertion below is that at least one tick did separate
    // them -- a run where they never diverged would prove nothing.
    U32 warnings = 0U;
    U32 diverged = 0U;
    for (U32 i = 0U; i < 400U; ++i) {
        F32 v[3];
        v[0] = 0.0F;
        v[1] = 12.0F;
        v[2] = ((i % 2U) == 0U) ? 30.0F : -30.0F;
        const FwSizeType before = this->eventHistory_CrossChannelWarning->size();
        this->tickChannels(v, 3U);
        const FwSizeType after = this->eventHistory_CrossChannelWarning->size();
        if (after == before) {
            continue;
        }
        ++warnings;

        const U32 fused = this->component.detector().fusedChannel();
        const U32 named = this->eventHistory_CrossChannelWarning->at(
            static_cast<U32>(after - 1U)).channelId;
        ASSERT_EQ(this->component.detector().model().channelId[fused], named)
            << "the event named channel id " << named << " while the fused rule "
            << "peaked on index " << fused << " (id "
            << this->component.detector().model().channelId[fused] << ")";

        // What the OLD code would have said, computed here only to prove the two
        // can differ. If they never differ the test is vacuous and says so.
        const F32* smoothed = this->component.detector().smoothed();
        U32 residualPeak = 0U;
        F32 largest = smoothed[0];
        for (U32 ch = 1U; ch < 3U; ++ch) {
            if (smoothed[ch] > largest) {
                largest = smoothed[ch];
                residualPeak = ch;
            }
        }
        if (residualPeak != fused) {
            ++diverged;
        }
    }

    ASSERT_GT(warnings, 0U) << "no warning was emitted, so nothing was checked";
    ASSERT_GT(diverged, 0U)
        << "the residual argmax and the fused argmax never disagreed across "
        << warnings << " warnings, so this test would have passed before the fix "
        << "too. It is vacuous as written and the drive pattern needs changing.";
}


// ----------------------------------------------------------------------
// Work item 10 / docs/MODELS.md 73, HO3: the commanded reload
// ----------------------------------------------------------------------

void MonitorTester ::testACommandedReloadLoadsTheNamedFile() {
    moveToBuildCache();
    ASSERT_TRUE(m_haveVectors) << "g1.bin not readable from " << G1_PATH;
    this->restore();
    ASSERT_EQ(Mode::MODEL, this->loadWorkingAs(WORK_PATH));
    this->clearHistory();

    // A second good file, at a second path. Same bytes: what is under test is
    // that the command loads the file it NAMES, not that a different model
    // arrives -- ModelReloadAccepted carries the path, and that is the evidence.
    std::FILE* handle = std::fopen(CANDIDATE_PATH, "wb");
    ASSERT_NE(nullptr, handle);
    (void)std::fwrite(m_working, 1U, static_cast<size_t>(m_length), handle);
    (void)std::fclose(handle);

    this->sendCmd_RELOAD_MODEL(0, 10U, Fw::CmdStringArg(CANDIDATE_PATH));
    this->invoke_to_schedIn(0, 0U);

    ASSERT_EVENTS_ModelReloadAccepted_SIZE(1);
    ASSERT_EVENTS_ModelReloadAccepted(0, CANDIDATE_PATH);
    ASSERT_EVENTS_ModelLoaded_SIZE(1);
    ASSERT_EVENTS_ModelReloadRolledBack_SIZE(0);
    ASSERT_EVENTS_DegradedToBaseline_SIZE(0);
    ASSERT_EQ(Mode::MODEL, this->component.activeMode());
    ASSERT_CMD_RESPONSE_SIZE(1);
    ASSERT_CMD_RESPONSE(0, Monitor::OPCODE_RELOAD_MODEL, 10U,
                        Fw::CmdResponse::OK);
}

void MonitorTester ::testARefusedReloadRestoresThePreviousModel() {
    moveToBuildCache();
    ASSERT_TRUE(m_haveVectors);
    this->restore();
    ASSERT_EQ(Mode::MODEL, this->loadWorkingAs(WORK_PATH));
    this->clearHistory();

    // A candidate that will be refused: one flipped byte inside the weights,
    // which the static CRC covers. BAD_STATIC_CRC is a real refusal code and
    // not a synthetic one -- it is the code a corrupted uplink would produce.
    this->restore();
    m_working[Format::HEADER_BYTES + (Format::CHANNEL_RECORD_BYTES * G1_CHANNELS)] ^= 0xFFU;
    std::FILE* handle = std::fopen(CANDIDATE_PATH, "wb");
    ASSERT_NE(nullptr, handle);
    (void)std::fwrite(m_working, 1U, static_cast<size_t>(m_length), handle);
    (void)std::fclose(handle);

    this->sendCmd_RELOAD_MODEL(0, 11U, Fw::CmdStringArg(CANDIDATE_PATH));
    this->invoke_to_schedIn(0, 0U);

    // The refusal is named, and it is named before the rollback.
    ASSERT_EVENTS_ModelRefused_SIZE(1);
    ASSERT_EVENTS_ModelReloadRolledBack_SIZE(1);
    ASSERT_EVENTS_ModelReloadAccepted_SIZE(0);

    // (!) AND THE MODEL THAT WAS WORKING IS WORKING AGAIN. Without the rollback
    // this assertion fails with Mode::BASELINE -- a refused candidate would have
    // cost the operator the model it was offered to replace.
    ASSERT_EQ(Mode::MODEL, this->component.activeMode());

    ASSERT_CMD_RESPONSE_SIZE(1);
    ASSERT_CMD_RESPONSE(0, Monitor::OPCODE_RELOAD_MODEL, 11U,
                        Fw::CmdResponse::EXECUTION_ERROR);
}

void MonitorTester ::testTheReloadCommandIsDispatchedInsideTheTick() {
    moveToBuildCache();
    ASSERT_TRUE(m_haveVectors);
    this->restore();
    ASSERT_EQ(Mode::MODEL, this->loadWorkingAs(WORK_PATH));
    this->clearHistory();

    std::FILE* handle = std::fopen(CANDIDATE_PATH, "wb");
    ASSERT_NE(nullptr, handle);
    (void)std::fwrite(m_working, 1U, static_cast<size_t>(m_length), handle);
    (void)std::fclose(handle);

    // (!) THE OTHER DIRECTION, AND IT IS THE ONE THAT PROVES THE COMPONENT IS
    // QUEUED RATHER THAN PASSIVE. The command is sent and NOT ticked: nothing
    // must have happened yet, because an async command sits on the queue until
    // schedIn drains it. A passive component would have run it on arrival.
    this->sendCmd_RELOAD_MODEL(0, 12U, Fw::CmdStringArg(CANDIDATE_PATH));
    ASSERT_EVENTS_ModelReloadAccepted_SIZE(0);
    ASSERT_CMD_RESPONSE_SIZE(0);

    this->invoke_to_schedIn(0, 0U);
    ASSERT_EVENTS_ModelReloadAccepted_SIZE(1);
    ASSERT_CMD_RESPONSE_SIZE(1);
}

void MonitorTester ::testAReloadThatChangesTheChannelWidthIsRefused() {
    moveToBuildCache();
    ASSERT_TRUE(m_haveVectors);
    this->restore();
    ASSERT_EQ(Mode::MODEL, this->loadWorkingAs(WORK_PATH));
    this->clearHistory();

    // g2 is a REAL, VALID model file -- it loads, its CRCs are right, nothing
    // about it is corrupt. It is simply for a seven-channel subsystem, and this
    // component is wired to a three-channel one.
    std::FILE* in = std::fopen(G2_PATH, "rb");
    if (in == nullptr) {
        GTEST_SKIP() << "g2.bin not readable from " << G2_PATH;
    }
    std::FILE* out = std::fopen(WIDE_PATH, "wb");
    ASSERT_NE(nullptr, out);
    U8 chunk[4096];
    size_t got = 0U;
    while ((got = std::fread(chunk, 1U, sizeof(chunk), in)) > 0U) {
        (void)std::fwrite(chunk, 1U, got, out);
    }
    (void)std::fclose(in);
    (void)std::fclose(out);

    this->sendCmd_RELOAD_MODEL(0, 13U, Fw::CmdStringArg(WIDE_PATH));
    this->invoke_to_schedIn(0, 0U);

    // (!) THE FILE LOADED. That is the point: no refusal code fires, because
    // nothing is wrong with the file. What is wrong is that it is not this
    // topology's model, and only the commanded path can know that.
    ASSERT_EVENTS_ModelReloadWidthRefused_SIZE(1);
    ASSERT_EVENTS_ModelReloadWidthRefused(0, WIDE_PATH, G1_CHANNELS, 7U);
    ASSERT_EVENTS_ModelReloadAccepted_SIZE(0);

    // And the model this topology IS wired for is running again.
    ASSERT_EQ(Mode::MODEL, this->component.activeMode());
    ASSERT_CMD_RESPONSE_SIZE(1);
    ASSERT_CMD_RESPONSE(0, Monitor::OPCODE_RELOAD_MODEL, 13U,
                        Fw::CmdResponse::VALIDATION_ERROR);
}

}  // namespace Sentinel
