#include "sentinel/ModelFile.hpp"

#include <cstring>

#include "sentinel/Crc32.hpp"

namespace Sentinel {
namespace {

// Little-endian readers. Assembled from bytes rather than memcpy'd over the
// stream, so the file reads the same on a big-endian target; F' CPP-10
// discourages reinterpret_cast and none is needed.
U16 readU16(const U8* p) {
    return static_cast<U16>(static_cast<U32>(p[0]) | (static_cast<U32>(p[1]) << 8U));
}

U32 readU32(const U8* p) {
    return static_cast<U32>(p[0]) | (static_cast<U32>(p[1]) << 8U)
         | (static_cast<U32>(p[2]) << 16U) | (static_cast<U32>(p[3]) << 24U);
}

U64 readU64(const U8* p) {
    return static_cast<U64>(readU32(p))
         | (static_cast<U64>(readU32(p + 4)) << 32U);
}

// The bit pattern is little-endian in the file; the host's float layout is IEEE
// 754 either way, so assemble the integer first and copy the bits across.
F32 readF32(const U8* p) {
    const U32 bits = readU32(p);
    F32 value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

F64 readF64(const U8* p) {
    const U64 bits = readU64(p);
    F64 value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

Model::Model()
    : nLayers(0U), nChannels(0U), nInputs(0U), nExogenous(0U), window(0U),
      nPredictions(0U), hidden(), layers(), headW(0U), headB(0U), nOutputs(0U),
      channelId(), channelName(), weights(), paramVersion(0U), ewmaSpan(0U),
      agreement(0U), persistence(0U), warmupSteps(0U), baselineOnly(0U), tier(0U),
      threshold(0.0), normOffset(), normScale(), provenance(), m_loaded(false) {}

LoadStatus Model::load(const U8* data, U32 length) {
    m_loaded = false;
    if (data == nullptr || length < Format::HEADER_BYTES) {
        return LoadStatus::TRUNCATED;
    }
    if (data[0] != 'S' || data[1] != 'N' || data[2] != 'T' || data[3] != 'L') {
        return LoadStatus::BAD_MAGIC;
    }
    if (static_cast<U32>(readU16(data + 4)) != Format::FORMAT_VERSION) {
        return LoadStatus::BAD_VERSION;
    }

    // Nothing below this line trusts a length field until the header CRC holds.
    if (crc32(data, Format::HEADER_CRC_OFFSET)
            != readU32(data + Format::HEADER_CRC_OFFSET)) {
        return LoadStatus::BAD_HEADER_CRC;
    }

    const U32 headerBytes = static_cast<U32>(readU16(data + 6));
    const U32 archId = static_cast<U32>(readU16(data + 8));
    const U32 gateOrderId = static_cast<U32>(readU16(data + 10));
    const U32 fileLayers = static_cast<U32>(readU16(data + 12));
    const U32 fileChannels = static_cast<U32>(readU16(data + 14));
    const U32 fileInputs = static_cast<U32>(readU16(data + 16));
    const U32 fileExogenous = static_cast<U32>(readU16(data + 18));
    const U32 fileWindow = static_cast<U32>(readU16(data + 20));
    const U32 filePredictions = static_cast<U32>(readU16(data + 22));

    if ((headerBytes != Format::HEADER_BYTES) || (readU32(data + 52) != 0U)
            || (readU32(data + 56) != 0U)) {
        return LoadStatus::BAD_SHAPE;
    }
    if (archId != Format::ARCH_GRU) {
        return LoadStatus::BAD_ARCH;
    }
    if (gateOrderId != Format::GATE_ORDER_RESET_UPDATE_NEW) {
        return LoadStatus::BAD_GATE_ORDER;
    }

    U32 slot[Format::HEADER_LAYER_SLOTS];
    for (U32 i = 0U; i < Format::HEADER_LAYER_SLOTS; ++i) {
        slot[i] = static_cast<U32>(readU16(data + 24U + (2U * i)));
    }
    if ((fileLayers == 0U) || (fileLayers > Format::HEADER_LAYER_SLOTS)) {
        return LoadStatus::BAD_SHAPE;
    }
    for (U32 i = 0U; i < Format::HEADER_LAYER_SLOTS; ++i) {
        const bool used = (i < fileLayers);
        if (used == (slot[i] == 0U)) {          // used-and-zero, or unused-and-set
            return LoadStatus::BAD_SHAPE;
        }
    }
    if ((fileInputs != (fileChannels + fileExogenous)) || (fileChannels == 0U)) {
        return LoadStatus::BAD_SHAPE;
    }

    // The header is coherent; now it must fit what this build budgeted for.
    if ((fileChannels > Config::MAX_CHANNELS) || (fileLayers > Config::MAX_LAYERS)
            || (filePredictions > Config::MAX_PREDICTIONS)
            || (fileInputs > Config::MAX_INPUTS) || (filePredictions == 0U)) {
        return LoadStatus::TOO_LARGE;
    }
    for (U32 i = 0U; i < fileLayers; ++i) {
        if (slot[i] > Config::MAX_HIDDEN) {
            return LoadStatus::TOO_LARGE;
        }
    }

    // Every declared size must account for the payload exactly (D16).
    const U32 channelBytes = readU32(data + 32);
    const U32 weightBytes = readU32(data + 36);
    const U32 paramBytes = readU32(data + 40);
    const U32 staticCrc = readU32(data + 44);
    const U32 paramCrc = readU32(data + 48);

    U32 expectedParameters = 0U;
    U32 previous = fileInputs;
    for (U32 i = 0U; i < fileLayers; ++i) {
        const U32 gates = Config::N_GATES * slot[i];
        expectedParameters += (gates * previous) + (gates * slot[i]) + gates + gates;
        previous = slot[i];
    }
    const U32 outputs = filePredictions * fileChannels;
    expectedParameters += (outputs * previous) + outputs;

    if ((channelBytes != (Format::CHANNEL_RECORD_BYTES * fileChannels))
            || (weightBytes != (4U * expectedParameters))
            || (paramBytes != (Format::PARAM_FIXED_BYTES + (8U * fileChannels)))) {
        return LoadStatus::BAD_SHAPE;
    }
    if (expectedParameters > Format::MAX_PARAMETERS) {
        return LoadStatus::TOO_LARGE;
    }
    if (length != (Format::HEADER_BYTES + channelBytes + weightBytes + paramBytes)) {
        return LoadStatus::TRUNCATED;
    }

    const U32 channelLo = Format::HEADER_BYTES;
    const U32 weightLo = channelLo + channelBytes;
    const U32 paramLo = weightLo + weightBytes;

    if (crc32(data + channelLo, channelBytes + weightBytes) != staticCrc) {
        return LoadStatus::BAD_STATIC_CRC;
    }
    if (crc32(data + paramLo, paramBytes) != paramCrc) {
        return LoadStatus::BAD_PARAM_CRC;
    }

    // -- the parameter block, checked before anything is committed ----------
    const U8* params = data + paramLo;
    const U32 filePolicy = static_cast<U32>(readU16(params + 2));
    const U32 fileBaselineOnly = static_cast<U32>(params[16]);
    const U32 fileTier = static_cast<U32>(params[17]);
    if ((readU16(params + 10) != 0U) || (readU16(params + 18) != 0U)
            || (readU32(params + 20) != 0U)) {
        return LoadStatus::BAD_SHAPE;
    }
    if (filePolicy != Format::NORM_IDENTITY) {
        return LoadStatus::BAD_NORM_POLICY;
    }
    if ((fileBaselineOnly > 1U) || (fileTier < 1U) || (fileTier > 3U)) {
        return LoadStatus::BAD_SHAPE;
    }
    if (params[Format::PARAM_FIXED_BYTES - 1U] != 0U) {
        return LoadStatus::BAD_SHAPE;          // provenance not NUL-terminated
    }
    for (U32 i = 0U; i < fileChannels; ++i) {
        const U8* record = data + channelLo + (Format::CHANNEL_RECORD_BYTES * i);
        if (record[Format::CHANNEL_RECORD_BYTES - 1U] != 0U) {
            return LoadStatus::BAD_SHAPE;      // channel name not NUL-terminated
        }
    }

    // Everything has been checked. Only now is anything written.
    nLayers = fileLayers;
    nChannels = fileChannels;
    nInputs = fileInputs;
    nExogenous = fileExogenous;
    window = fileWindow;
    nPredictions = filePredictions;
    nOutputs = outputs;

    U32 offset = 0U;
    previous = fileInputs;
    for (U32 i = 0U; i < fileLayers; ++i) {
        const U32 gates = Config::N_GATES * slot[i];
        hidden[i] = slot[i];
        layers[i].hidden = slot[i];
        layers[i].inputs = previous;
        layers[i].wIh = offset;
        offset += gates * previous;
        layers[i].wHh = offset;
        offset += gates * slot[i];
        layers[i].bIh = offset;
        offset += gates;
        layers[i].bHh = offset;
        offset += gates;
        previous = slot[i];
    }
    headW = offset;
    offset += outputs * previous;
    headB = offset;

    for (U32 i = 0U; i < expectedParameters; ++i) {
        weights[i] = readF32(data + weightLo + (4U * i));
    }

    for (U32 i = 0U; i < fileChannels; ++i) {
        const U8* record = data + channelLo + (Format::CHANNEL_RECORD_BYTES * i);
        channelId[i] = readU32(record);
        std::memcpy(channelName[i], record + 4, Config::CHANNEL_NAME_BYTES);
    }

    paramVersion = static_cast<U32>(readU16(params + 0));
    ewmaSpan = static_cast<U32>(readU16(params + 4));
    agreement = static_cast<U32>(readU16(params + 6));
    persistence = static_cast<U32>(readU16(params + 8));
    warmupSteps = readU32(params + 12);
    baselineOnly = static_cast<U8>(fileBaselineOnly);
    tier = static_cast<U8>(fileTier);
    threshold = readF64(params + 24);
    std::memcpy(provenance, params + 32, Format::PROVENANCE_BYTES);

    const U8* constants = params + Format::PARAM_FIXED_BYTES;
    for (U32 i = 0U; i < fileChannels; ++i) {
        normOffset[i] = readF32(constants + (4U * i));
        normScale[i] = readF32(constants + (4U * (fileChannels + i)));
    }

    if (ewmaSpan == 0U) {
        return LoadStatus::BAD_SHAPE;
    }

    m_loaded = true;
    return LoadStatus::OK;
}

}  // namespace Sentinel
