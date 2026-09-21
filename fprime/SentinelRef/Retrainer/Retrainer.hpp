// ======================================================================
// \title  Retrainer.hpp
// \brief  E1, the pipe: an F' component that calls into an OxCaml static library
//
// Test apparatus, not product. `fprime/library.cmake` exports `Sentinel/Monitor`
// and nothing here, so a mission adopting Sentinel does not inherit an OCaml
// runtime. See `docs/MODELS.md` 47 and `docs/DECISIONS.md` D70.
//
// It carries NO machine learning. E1 exists to prove a chain -- compile, link,
// OCaml runtime startup, the C boundary, F' integration -- and the payload is
// arithmetic whose answer can be checked by hand for exactly that reason
// (`docs/MODELS.md` 47.2).
//
// (!) NOTHING HERE MAY THROW, AND NOTHING BELOW IT CAN. CPP-25 forbids
// exceptions and this deployment builds -fno-exceptions. `retrainer.ml` catches
// every OCaml exception and returns a status code, so no exception can reach this
// translation unit -- which is what `docs/MODELS.md` 47.6 means by structurally
// impossible rather than merely avoided.
//
// (!) AND NOTHING HERE ALLOCATES AFTER INIT. CPP-1. The accumulator is sized once,
// inside OCaml, at RETRAINER_CAPACITY; the sample and export buffers are members
// of fixed size. The OCaml runtime's own heap is a separate matter and is exactly
// what E2 and E3 exist to measure -- it is NOT claimed here to be allocation-free.
// ======================================================================

#ifndef Retrain_Retrainer_HPP
#define Retrain_Retrainer_HPP

#include "SentinelRef/Retrainer/RetrainerComponentAc.hpp"

namespace Retrain {

class Retrainer final : public RetrainerComponentBase {
  public:
    //! How many samples one tick feeds. Fixed, so the per-cycle work is fixed
    //! (Objective.md 11 rule 5).
    static const U32 SAMPLES_PER_TICK = 10U;

    //! The export buffer's width: count, sum, mean.
    static const U32 EXPORT_WIDTH = 3U;

    //! Raise the E1 accumulator's bound above RETRAINER_CAPACITY, before boot.
    //!
    //! (!) THIS ALLOCATES NOTHING AND THE REASON MATTERS. `capacity` is a scalar
    //! field of a record on the OCaml side (`oxcaml/retrainer/retrainer.ml:32`),
    //! compared against in `feed` and sizing no array at all; the sample and
    //! export buffers are members of this class, sized by SAMPLES_PER_TICK and
    //! EXPORT_WIDTH. So a larger bound costs zero bytes and CPP-1 is untouched.
    //!
    //! It exists because a crossing measurement needs more than the six ticks
    //! RETRAINER_CAPACITY allows, and because `init` REFUSES a second call --
    //! `retrainer.ml:50` returns `err_already_init` -- so the accumulator cannot
    //! be reset per tick. Configuring before boot is the only route that changes
    //! no OCaml source and keeps the unit test's refusal at tick seven intact.
    //!
    //! Has no effect once the runtime has booted.
    void configure(U32 capacity);

    // 61 / E5-c. The retraining cycle's fixed extents, claimed once (CPP-1).
    //
    // (!) CYCLE_BUDGET IS NOT THE FLIGHT BUDGET. docs/MODELS.md 55.4 stop 23
    // reserves that derivation; this is a unit test's value, chosen so a tick is
    // cheap. What is under test is that the count is FIXED, not that it is right.
    static const U32 CYCLE_T = 8U;
    static const U32 CYCLE_INS = 16U;
    static const U32 CYCLE_WINDOW = 250U * 16U;   // the OCaml side's full extent
    static const I32 CYCLE_BUDGET = 1;

    // 72 / E5-e, HO1. The candidate model file's fixed extents.
    //
    // (!) THE SHAPES ARE THE CYCLE'S, NOT THE DEPLOYMENT'S, and that is a finding
    // rather than a convenience. `Deep_f32` is a fixed, maxima-shaped network --
    // 16 inputs, two layers of 80, 160 outputs, 75,360 parameters
    // (`deep_f32.ml:28-34`, "at Config.hpp's maxima"). `SentinelRef` flies an
    // 8-channel model with 66,960 weights, and `shadow59.ml:79` rightly refuses a
    // weights block whose size disagrees with the header it is written into. So
    // this component can only produce a candidate for a model at ITS OWN shapes,
    // and 59.2's premise -- "the retrainer trains the same architecture at the
    // same shapes" -- does not hold for this deployment. `docs/MODELS.md` 72
    // records it; 72.6 carries it as owed.
    static const U32 CYCLE_N_PARAMS = 75360U;   // deep_f32.ml:34

    // The largest model file this component budgets for. Same derivation as
    // `Monitor.hpp:38-43`, restated rather than shared: this module deliberately
    // does not link `flight/` (`library.cmake:31` exports the Monitor, and the
    // Retrainer is not on that path), so it may not include `ModelFile.hpp`.
    static const U32 MODEL_HEADER_BYTES = 64U;          // ModelFile.hpp:13
    static const U32 MODEL_CHANNEL_RECORD_BYTES = 20U;  // ModelFile.hpp:28
    static const U32 MODEL_PARAM_FIXED_BYTES = 96U;     // ModelFile.hpp:29
    static const U32 MODEL_MAX_CHANNELS = 16U;          // Config.hpp:23
    static const U32 MODEL_STATIC_CRC_OFFSET = 44U;     // shadow59.ml:36
    static const U32 MODEL_FILE_MAX_BYTES =
        MODEL_HEADER_BYTES
        + (MODEL_CHANNEL_RECORD_BYTES * MODEL_MAX_CHANNELS)
        + (4U * CYCLE_N_PARAMS)
        + MODEL_PARAM_FIXED_BYTES
        + (8U * MODEL_MAX_CHANNELS);

    explicit Retrainer(const char* const compName);
    ~Retrainer();

    //! Name the flying model file this process retrains from, and the path the
    //! candidate is written to. Both are read at boot; neither is reopened per
    //! tick. Matches the shape `Sentinel::Monitor::configure` takes, and like it
    //! this is a topology-setup call, never a command.
    //!
    //! Absent or unreadable, the component still ticks and still runs its cycle;
    //! it simply produces no candidate, and says so once. Degrade rather than
    //! die, which is `Monitor`'s posture on a bad model file.
    void configureShadow(const char* flyingPath, const char* candidatePath);

    //! Whether a flying file was loaded and a candidate can be produced.
    bool shadowArmed() const { return m_shadowArmed; }

    //! The last candidate's length and its own static_crc32, for the tests and
    //! for the ground. Zero until one has been written.
    U32 candidateBytes() const { return m_candidateBytes; }
    U32 candidateCrc32() const { return m_candidateCrc32; }

    //! Boot the OCaml runtime and size the accumulator. Safe to call before the
    //! topology is running. Returns true if the component is armed; a false
    //! return leaves it inert rather than failing the topology, which is the same
    //! degrade-rather-than-die posture `Sentinel::Monitor` takes on a bad model
    //! file. Never throws, never asserts on the runtime's behaviour.
    bool boot();

    //! Whether boot() succeeded, for the topology and the tests.
    bool armed() const { return m_armed; }

  private:
    //! One tick: feed a fixed block, step, read the accumulator back.
    void schedIn_handler(FwIndexType portNum, U32 context) override;

    //! Emit CallRefused for a non-OK status and record it. Returns true if the
    //! status was OK.
    bool accept(I32 status);

    // 72 / E5-e, HO1.
    //! Read the flying model file into m_modelFile. Returns its length, or 0.
    //! Calls no OCaml entry point -- boot() crosses the boundary, so the domain
    //! lock guard's permitted set stays at two names.
    U32 readFlying();

    //! Report a refusal on the candidate path, naming the stage.
    void refuseCandidate(CandidateStage::T stage, I32 code);

    //! Write the exported candidate out and report it. Calls no OCaml entry
    //! point; the bytes are already in this component's own buffer.
    void emitCandidate(U32 n, I32 steps);

    bool m_armed;

    //! (!) OCaml 5 grants the domain lock to the thread that calls caml_startup,
    //! and an ActiveRateGroup runs schedIn on its OWN task. Starting the runtime
    //! from the topology's main thread and then calling in from the rate group
    //! aborts the process with "Fatal error: no domain lock held". So the boot is
    //! deferred to the first tick, on the thread that will own the domain for the
    //! life of the process. An explicit boot() sets this too, so a unit test that
    //! boots on its own thread is unaffected. docs/MODELS.md 65.
    bool m_bootAttempted = false;    U32  m_refusals;

    //! 61 / E5-c. The window the cycle reads. THIS COMPONENT OWNS IT; the OCaml
    //! side sees a CAML_BA_EXTERNAL view that does not outlive the call.
    F32  m_window[CYCLE_WINDOW];

    //! Whether the cycle's runtime came up. Separate from m_armed: E1's pipe and
    //! 61's cycle boot independently and either may be absent.
    bool m_cycleArmed;
    //! Fixed-size, member-owned. The OCaml side wraps these in a Bigarray with
    //! CAML_BA_EXTERNAL, so their storage is never owned or moved by the GC.
    //! The accumulator bound passed to the OCaml side at boot.
    U32  m_capacity;

    F64  m_samples[SAMPLES_PER_TICK];
    F64  m_export[EXPORT_WIDTH];

    // 72 / E5-e, HO1. All fixed-size and member-owned (CPP-1). The OCaml side
    // wraps each in a Bigarray with CAML_BA_EXTERNAL, so the GC neither owns nor
    // moves the storage, and no wrapper outlives the call that made it.
    //
    // (!) ONE BUFFER SERVES BOTH DIRECTIONS. The flying bytes are read into it
    // at boot and copied straight into the OCaml side; the candidate is later
    // exported back into the same buffer. They are never both live, so a second
    // 302,048 B member would be 302,048 B of nothing.
    U8   m_modelFile[MODEL_FILE_MAX_BYTES];
    F32  m_weights[CYCLE_N_PARAMS];
    F32  m_loss[1];

    Fw::String m_flyingPath;
    Fw::String m_candidatePath;
    bool m_shadowArmed;
    U32  m_candidateBytes;
    U32  m_candidateCrc32;
};

}  // namespace Retrain

#endif
