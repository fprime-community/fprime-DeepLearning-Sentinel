// ======================================================================
// \title  RetrainerTester.cpp
// \brief  D85: the retrainer component, and E1's pipe kept as a fixture
// ======================================================================
#include "RetrainerTester.hpp"

#include <cstdio>
#include <cstdlib>

#include "sentinel_retrainer.h"
#include "sentinel_window.h"

namespace Sentinel {

RetrainerTester::RetrainerTester()
    : RetrainerGTestBase("RetrainerTester", RetrainerTester::MAX_HISTORY_SIZE),
      component("Retrainer")
{
    this->initComponents();
    this->connectPorts();
}

RetrainerTester::~RetrainerTester() {}

void RetrainerTester::sample(U32 seq, F32 value)
{
    ChannelVector v;
    for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
        v[static_cast<FwSizeType>(c)] = value + static_cast<F32>(c);
    }
    this->invoke_to_sampleIn(0, v, true, seq);
}

void RetrainerTester::testTheWholeLifecycle()
{
    // -- E1's pipe, kept as a FIXTURE (D85 c.2): the hand-checked numbers, called
    // directly, since the component no longer feeds the accumulator. X1 to X3.
    ASSERT_EQ(SENTINEL_RT_OK, sentinel_retrainer_boot());
    ASSERT_EQ(SENTINEL_RT_OK, sentinel_retrainer_init(64U));
    F64 samples[10];
    for (U32 i = 0U; i < 10U; ++i) { samples[i] = static_cast<F64>(i + 1U); }
    F64 out[3] = {0.0, 0.0, 0.0};
    ASSERT_EQ(SENTINEL_RT_OK, sentinel_retrainer_feed(samples, 10U));
    ASSERT_EQ(SENTINEL_RT_OK, sentinel_retrainer_step());
    ASSERT_EQ(SENTINEL_RT_OK, sentinel_retrainer_export(out, 3U));
    ASSERT_EQ(10.0, out[0]);
    ASSERT_EQ(55.0, out[1]);
    ASSERT_EQ(5.5, out[2]);
    for (U32 i = 0U; i < 5U; ++i) {
        ASSERT_EQ(SENTINEL_RT_OK, sentinel_retrainer_feed(samples, 10U));
    }
    // X3: the overflow is a STATUS, not an exception across -fno-exceptions.
    ASSERT_EQ(SENTINEL_RT_ERR_OVERFLOW, sentinel_retrainer_feed(samples, 10U));

    // -- D85: the loop's configuration is checked before anything runs -----------
    LoopConfig bad = {};
    ASSERT_FALSE(this->component.configureLoop(bad));
    LoopConfig cfg = {};
    for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
        cfg.yellowLow[c] = -1.0e6F;
        cfg.yellowHigh[c] = 1.0e6F;
    }
    cfg.guard = 10U;
    cfg.budget = 1;
    cfg.schedule = 1U;
    cfg.nominalPpm = 1046;
    ASSERT_TRUE(this->component.configureLoop(cfg));

    // -- the flying model: supplied by the test runner, or absent ----------------
    const char* const flying = std::getenv("SENTINEL_RETRAINER_UT_FLYING");
    if (flying == nullptr) {
        // No flying model: boot refuses once, the component stays inert, and a tick
        // does nothing -- degrade rather than die.
        this->component.configureShadow("no-such-flying-model.bin", "ut_candidate.bin");
        this->invoke_to_schedIn(0, 0);
        ASSERT_FALSE(this->component.armed());
        ASSERT_EVENTS_CandidateRefused_SIZE(1);
        ASSERT_EVENTS_CandidateRefused(0, CandidateStage::FLYING_READ, 0);
        this->invoke_to_schedIn(0, 0);
        ASSERT_EVENTS_CandidateRefused_SIZE(1);
        return;
    }

    // With one: boot on the first tick, on this (the rate-group) thread.
    this->component.configureShadow(flying, "ut_candidate.bin");
    this->invoke_to_schedIn(0, 0);
    ASSERT_TRUE(this->component.armed());
    ASSERT_EVENTS_RetrainerReady_SIZE(1);
    ASSERT_EVENTS_RetrainerReady(0, RetrainLoop::CHANNELS, 1U);

    // Samples queue on the hub thread (here, this one) and drain on the next tick.
    this->sample(0U, 0.1F);
    this->sample(1U, 0.2F);
    this->invoke_to_schedIn(0, 0);
    ASSERT_EQ(2U, this->component.samplesReceived());
    ASSERT_EQ(0U, this->component.samplesMissed());

    // A gap in the sequence is a loss, counted and fed to the loop as missed ticks.
    this->sample(5U, 0.3F);
    this->invoke_to_schedIn(0, 0);
    ASSERT_EQ(3U, this->component.samplesReceived());
    ASSERT_EQ(3U, this->component.samplesMissed());
    ASSERT_EVENTS_SamplesLost_SIZE(1);
    ASSERT_EVENTS_SamplesLost(0, 3U);
    // Missed and real samples alike reached the loop.
    ASSERT_EQ(6U, static_cast<U32>(this->component.loop().ticks()));

    // An overfull queue is bounded: at most QUEUE samples per tick, the rest counted.
    for (U32 i = 0U; i < Retrainer::QUEUE + 3U; ++i) {
        this->sample(6U + i, 0.4F);
    }
    this->invoke_to_schedIn(0, 0);
    ASSERT_EQ(3U + Retrainer::QUEUE, this->component.samplesReceived());
    ASSERT_EVENTS_CallRefused_SIZE(0);
}

// ----------------------------------------------------------------------------------------
// docs/MODELS.md 78.11: the live-window test. Every constant below is fixed there.
// ----------------------------------------------------------------------------------------

namespace {

//! 78.11: SentinelRetrain's own yellow bands (Top/instances.fpp).
const F32 LIVE_LOW[8] = {-2.0F, -10.0F, 0.5F, 27.0F, -5.0F, -35.0F, 0.0F, 0.30F};
const F32 LIVE_HIGH[8] = {300.0F, 10.0F, 9.5F, 32.0F, 40.0F, 30.0F, 1.0F, 1.0F};
const U32 LIVE_SAMPLES = 8250U;      //!< sequences 0 .. 8,249
const U32 LIVE_SPIKE_SEQ = 7000U;    //!< f
const F32 LIVE_SPIKE_WIDTHS = 40.0F; //!< the spike, in units of 1% of channel 0's band

//! Knuth's MMIX LCG, seed 1; u uniform on [-1, 1) from the top 24 bits.
struct Lcg {
    U64 state;
    F32 next() {
        state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
        const U32 top = static_cast<U32>(state >> 40U);
        return (static_cast<F32>(top) / 16777216.0F) * 2.0F - 1.0F;
    }
};

}  // namespace

void RetrainerTester::testLiveWindow(const char* flying, const char* candidate,
                                     const char* tracePath, bool spike)
{
    LoopConfig cfg = {};
    for (U32 c = 0U; c < 8U; ++c) {
        cfg.yellowLow[c] = LIVE_LOW[c];
        cfg.yellowHigh[c] = LIVE_HIGH[c];
    }
    cfg.guard = 260U;
    cfg.budget = 1;
    cfg.schedule = 1000000U;         // no candidate is written
    cfg.nominalPpm = 1046;
    ASSERT_TRUE(this->component.configureLoop(cfg));
    this->component.configureShadow(flying, candidate);

    std::FILE* trace = std::fopen(tracePath, "w");
    ASSERT_NE(nullptr, trace);
    std::fprintf(trace, "seq,admitted,crossing,emitted,window_admits\n");

    Lcg lcg = {1ULL};
    const Detector& replica = this->component.loop().replica();
    for (U32 seq = 0U; seq < LIVE_SAMPLES; ++seq) {
        ChannelVector v;
        for (U32 c = 0U; c < Config::MAX_CHANNELS; ++c) {
            v[static_cast<FwSizeType>(c)] = 0.0F;
        }
        for (U32 c = 0U; c < 8U; ++c) {
            const F32 mid = 0.5F * (LIVE_LOW[c] + LIVE_HIGH[c]);
            const F32 onePct = 0.01F * (LIVE_HIGH[c] - LIVE_LOW[c]);
            F32 value = mid + (lcg.next() * onePct);
            if (spike && (seq == LIVE_SPIKE_SEQ) && (c == 0U)) {
                value += LIVE_SPIKE_WIDTHS * onePct;
            }
            v[static_cast<FwSizeType>(c)] = value;
        }
        // Through the real ports: the hub thread's sampleIn, then the rate group's tick.
        this->invoke_to_sampleIn(0, v, true, seq);
        this->invoke_to_schedIn(0, 0);
        if (seq == 0U) {
            ASSERT_TRUE(this->component.armed());
            ASSERT_EVENTS_RetrainerReady_SIZE(1);
        }
        ASSERT_EVENTS_CallRefused_SIZE(0);
        ASSERT_EVENTS_SamplesLost_SIZE(0);
        ASSERT_EVENTS_CandidateWritten_SIZE(0);

        // The Admitted CHANNEL, as the ground would read it, checked against the loop.
        ASSERT_GT(this->tlmHistory_Admitted->size(), 0U);
        const U32 admitted =
            this->tlmHistory_Admitted->at(this->tlmHistory_Admitted->size() - 1U).arg;
        ASSERT_EQ(static_cast<U32>(this->component.loop().admittedTotal()), admitted);

        // The crossing exactly as RetrainLoop::tick gates it (warmed after the step).
        const bool warmed = (replica.steps() >= static_cast<U64>(replica.model().warmupSteps));
        const bool crossing = warmed && replica.crossing();
        const I32 admits = sentinel_window_admits();
        ASSERT_GE(admits, 0);
        std::fprintf(trace, "%u,%u,%d,%d,%d\n", seq, admitted, crossing ? 1 : 0,
                     replica.emitted() ? 1 : 0, admits);

        // MAX_HISTORY_SIZE is 512 and every tick adds one entry per channel.
        if ((seq % 500U) == 499U) {
            this->clearHistory();
        }
    }
    std::fclose(trace);
    ASSERT_EQ(LIVE_SAMPLES, this->component.samplesReceived());
    ASSERT_EQ(0U, this->component.samplesMissed());
}

}  // namespace Sentinel
