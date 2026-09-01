// The `model.bin` reader. `docs/MODEL_FILE.md` is normative.
#ifndef SENTINEL_MODELFILE_HPP
#define SENTINEL_MODELFILE_HPP

#include "sentinel/Config.hpp"
#include "sentinel/Status.hpp"
#include "sentinel/Types.hpp"

namespace Sentinel {

namespace Format {

constexpr U32 HEADER_BYTES = 64U;
constexpr U32 HEADER_CRC_OFFSET = 60U;
constexpr U32 FORMAT_VERSION = 1U;
constexpr U32 ARCH_GRU = 1U;
constexpr U32 GATE_ORDER_RESET_UPDATE_NEW = 1U;
constexpr U32 NORM_IDENTITY = 0U;
constexpr U32 HEADER_LAYER_SLOTS = 4U;
constexpr U32 CHANNEL_RECORD_BYTES = 20U;
constexpr U32 PARAM_FIXED_BYTES = 96U;
constexpr U32 PROVENANCE_BYTES = 64U;

//! The largest weight payload this build budgets for, in float32 elements.
//! Derived, not typed: layer 0 reads MAX_INPUTS columns and every later layer
//! reads MAX_HIDDEN, and the head reads the last layer's width. 75,360 at the
//! maxima in Config.hpp; 71,160 is what the flown model uses.
constexpr U32 maxParameters() {
    U32 total = 0U;
    U32 previous = Config::MAX_INPUTS;
    for (U32 layer = 0U; layer < Config::MAX_LAYERS; ++layer) {
        total += (Config::MAX_GATE_WIDTH * previous)      // w_ih
               + (Config::MAX_GATE_WIDTH * Config::MAX_HIDDEN)  // w_hh
               + Config::MAX_GATE_WIDTH                   // b_ih
               + Config::MAX_GATE_WIDTH;                  // b_hh
        previous = Config::MAX_HIDDEN;
    }
    total += (Config::MAX_OUTPUTS * previous) + Config::MAX_OUTPUTS;  // head
    return total;
}

constexpr U32 MAX_PARAMETERS = maxParameters();

}  // namespace Format

//! Where one layer's arrays sit inside the weight arena, and how wide they are.
struct LayerView {
    U32 hidden;
    U32 inputs;
    U32 wIh;      // offset into Model::weights
    U32 wHh;
    U32 bIh;
    U32 bHh;
};

//! Everything the forward pass and the decision layer need, as plain arrays.
//!
//! One contiguous weight arena, in `reference.Weights.arrays()` order, which is
//! the order the file stores them -- so loading is a single copy and every layer
//! is an offset rather than a separate buffer. No allocation anywhere (F' CPP-1).
struct Model {
    // -- shape -------------------------------------------------------------
    U32 nLayers;
    U32 nChannels;
    U32 nInputs;
    U32 nExogenous;
    U32 window;
    U32 nPredictions;
    U32 hidden[Config::MAX_LAYERS];
    LayerView layers[Config::MAX_LAYERS];
    U32 headW;                                   // offset into weights
    U32 headB;
    U32 nOutputs;                                // nPredictions * nChannels

    // -- channel map -------------------------------------------------------
    U32 channelId[Config::MAX_CHANNELS];
    char channelName[Config::MAX_CHANNELS][Config::CHANNEL_NAME_BYTES];

    // -- weights -----------------------------------------------------------
    F32 weights[Format::MAX_PARAMETERS];

    // -- the separately replaceable parameter block ------------------------
    U32 paramVersion;
    U32 ewmaSpan;
    U32 agreement;
    U32 persistence;
    U32 warmupSteps;
    U8 baselineOnly;
    U8 tier;
    F64 threshold;
    F32 normOffset[Config::MAX_CHANNELS];
    F32 normScale[Config::MAX_CHANNELS];
    char provenance[Format::PROVENANCE_BYTES];

    Model();

    //! Parse and verify. Returns a status; never throws (F' CPP-25).
    //!
    //! Order of checks is fixed by `docs/MODEL_FILE.md` 8 and it matters: magic,
    //! version, then the header CRC -- **before any length field is used** -- so a
    //! corrupted `weight_bytes` can never drive an over-read. The model is left
    //! unusable on any non-OK status, so nothing is ever used before it is checked.
    LoadStatus load(const U8* data, U32 length);

    bool loaded() const { return m_loaded; }

  private:
    bool m_loaded;
};

}  // namespace Sentinel

#endif  // SENTINEL_MODELFILE_HPP
