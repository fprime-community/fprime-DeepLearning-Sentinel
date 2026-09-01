// Re-emit the file from what was parsed, so Python can compare byte for byte.
//
// `RoundTripDump in.bin out.bin` loads a model and writes the whole file back out
// from the `Model` struct alone -- header, channel map, weights and parameter
// block, with all three CRCs recomputed. If the output is byte-identical to the
// input then the C++ read every field correctly, which is what "round-trips
// Python -> C++ -> Python byte-identical" means (`docs/MODELS.md` 19, F6).
//
// This is a test tool. Flight code never writes a model file.
#include <cstdio>
#include <cstring>

#include "sentinel/Crc32.hpp"
#include "sentinel/ModelFile.hpp"
#include "TestSupport.hpp"

namespace SentinelTest {
int failures = 0;
}

using namespace Sentinel;

namespace {

U8 g_in[SentinelTest::MAX_FILE_BYTES];
U8 g_out[SentinelTest::MAX_FILE_BYTES];

void writeU16(U8* p, U32 value) {
    p[0] = static_cast<U8>(value & 0xFFU);
    p[1] = static_cast<U8>((value >> 8U) & 0xFFU);
}

void writeU32(U8* p, U32 value) {
    p[0] = static_cast<U8>(value & 0xFFU);
    p[1] = static_cast<U8>((value >> 8U) & 0xFFU);
    p[2] = static_cast<U8>((value >> 16U) & 0xFFU);
    p[3] = static_cast<U8>((value >> 24U) & 0xFFU);
}

void writeF32(U8* p, F32 value) {
    U32 bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    writeU32(p, bits);
}

void writeF64(U8* p, F64 value) {
    U64 bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    writeU32(p, static_cast<U32>(bits & 0xFFFFFFFFU));
    writeU32(p + 4, static_cast<U32>((bits >> 32U) & 0xFFFFFFFFU));
}

U32 emit(const Model& model, U8* out) {
    const U32 channels = model.nChannels;
    const U32 channelBytes = Format::CHANNEL_RECORD_BYTES * channels;
    const U32 paramBytes = Format::PARAM_FIXED_BYTES + (8U * channels);

    U32 parameters = 0U;
    U32 previous = model.nInputs;
    for (U32 i = 0U; i < model.nLayers; ++i) {
        const U32 gates = Config::N_GATES * model.hidden[i];
        parameters += (gates * previous) + (gates * model.hidden[i]) + gates + gates;
        previous = model.hidden[i];
    }
    parameters += (model.nOutputs * previous) + model.nOutputs;
    const U32 weightBytes = 4U * parameters;

    const U32 channelLo = Format::HEADER_BYTES;
    const U32 weightLo = channelLo + channelBytes;
    const U32 paramLo = weightLo + weightBytes;
    const U32 total = paramLo + paramBytes;

    std::memset(out, 0, static_cast<size_t>(total));
    out[0] = 'S'; out[1] = 'N'; out[2] = 'T'; out[3] = 'L';
    writeU16(out + 4, Format::FORMAT_VERSION);
    writeU16(out + 6, Format::HEADER_BYTES);
    writeU16(out + 8, Format::ARCH_GRU);
    writeU16(out + 10, Format::GATE_ORDER_RESET_UPDATE_NEW);
    writeU16(out + 12, model.nLayers);
    writeU16(out + 14, channels);
    writeU16(out + 16, model.nInputs);
    writeU16(out + 18, model.nExogenous);
    writeU16(out + 20, model.window);
    writeU16(out + 22, model.nPredictions);
    for (U32 i = 0U; i < model.nLayers; ++i) {
        writeU16(out + 24U + (2U * i), model.hidden[i]);
    }
    writeU32(out + 32, channelBytes);
    writeU32(out + 36, weightBytes);
    writeU32(out + 40, paramBytes);

    for (U32 i = 0U; i < channels; ++i) {
        U8* record = out + channelLo + (Format::CHANNEL_RECORD_BYTES * i);
        writeU32(record, model.channelId[i]);
        std::memcpy(record + 4, model.channelName[i], Config::CHANNEL_NAME_BYTES);
    }
    for (U32 i = 0U; i < parameters; ++i) {
        writeF32(out + weightLo + (4U * i), model.weights[i]);
    }

    U8* params = out + paramLo;
    writeU16(params + 0, model.paramVersion);
    writeU16(params + 2, Format::NORM_IDENTITY);
    writeU16(params + 4, model.ewmaSpan);
    writeU16(params + 6, model.agreement);
    writeU16(params + 8, model.persistence);
    writeU32(params + 12, model.warmupSteps);
    params[16] = model.baselineOnly;
    params[17] = model.tier;
    writeF64(params + 24, model.threshold);
    std::memcpy(params + 32, model.provenance, Format::PROVENANCE_BYTES);
    for (U32 i = 0U; i < channels; ++i) {
        writeF32(params + Format::PARAM_FIXED_BYTES + (4U * i), model.normOffset[i]);
        writeF32(params + Format::PARAM_FIXED_BYTES + (4U * (channels + i)),
                 model.normScale[i]);
    }

    writeU32(out + 44, crc32(out + channelLo, channelBytes + weightBytes));
    writeU32(out + 48, crc32(out + paramLo, paramBytes));
    writeU32(out + Format::HEADER_CRC_OFFSET, crc32(out, Format::HEADER_CRC_OFFSET));
    return total;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::printf("usage: RoundTripDump <in.bin> <out.bin>\n");
        return 2;
    }
    const U32 length = SentinelTest::readFile(argv[1], g_in,
                                              SentinelTest::MAX_FILE_BYTES);
    if (length == 0U) {
        std::printf("cannot read %s\n", argv[1]);
        return 2;
    }

    Model model;
    const LoadStatus status = model.load(g_in, length);
    if (status != LoadStatus::OK) {
        std::printf("load failed: %s\n", statusName(status));
        return 2;
    }

    const U32 written = emit(model, g_out);
    std::FILE* out = std::fopen(argv[2], "wb");
    if (out == nullptr) {
        std::printf("cannot write %s\n", argv[2]);
        return 2;
    }
    (void)std::fwrite(g_out, 1U, static_cast<size_t>(written), out);
    (void)std::fclose(out);
    return 0;
}
