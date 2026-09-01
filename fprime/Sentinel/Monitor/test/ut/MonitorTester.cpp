// ======================================================================
// \title  MonitorTester.cpp
// \brief  Test harness for Sentinel::Monitor
// ======================================================================

#include "MonitorTester.hpp"

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

//! Written into the build cache, never into the source tree: a test that
//! dirties the tree is what tests/test_no_local_persistence.py forbids on the
//! Python side, and the rule is the same here.
#ifndef SENTINEL_UT_TMP
#define SENTINEL_UT_TMP "."
#endif
const char* const WORK_PATH = SENTINEL_UT_TMP "/MonitorTester_model.bin";

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

    // The eleven refusal codes, each reached by the mutation
    // flight/test/RefusalTests.cpp uses for it.
    struct Case {
        const char* what;
        ModelLoadStatus::T expected;
    };

    for (U32 index = 0U; index < 11U; ++index) {
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
            default:
                this->writeU16At(this->paramBlockOffset() + 2U, 1U);
                this->resignParams();
                c = {"a non-identity normalisation policy",
                     ModelLoadStatus::BAD_NORM_POLICY};
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

}  // namespace Sentinel
