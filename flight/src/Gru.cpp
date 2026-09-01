#include "sentinel/Gru.hpp"

#include <cmath>

namespace Sentinel {
namespace Gru {
namespace {

//! `y = M x + b`, with the bias added **after** the dot product.
//!
//! NumPy computes `(x @ w.T) + b`, so the bias is one addition onto a finished
//! sum. Seeding the accumulator with the bias instead would be the same value in
//! exact arithmetic and a different one in float32, and this core is held to the
//! reference at 1e-5.
void affine(const F32* matrix, const F32* bias, const F32* vector, U32 rows,
            U32 columns, F32* out) {
    for (U32 r = 0U; r < rows; ++r) {
        const F32* row = matrix + (r * columns);
        F32 accumulator = 0.0F;
        for (U32 c = 0U; c < columns; ++c) {
            accumulator += row[c] * vector[c];
        }
        out[r] = accumulator + bias[r];
    }
}

}  // namespace

F32 sigmoid(F32 x) {
    if (x >= 0.0F) {
        return 1.0F / (1.0F + std::exp(-x));
    }
    const F32 exponent = std::exp(x);
    return exponent / (1.0F + exponent);
}

void step(const Model& model, U32 layerIndex, const F32* input, F32* hidden,
          F32* projected, F32* recurrent) {
    const LayerView& layer = model.layers[layerIndex];
    const U32 width = layer.hidden;
    const U32 gates = Config::N_GATES * width;

    affine(&model.weights[layer.wIh], &model.weights[layer.bIh], input, gates,
           layer.inputs, projected);
    // One product for all three gates, carrying the whole of b_hh -- which is why
    // b_hn ends up inside the reset product below and cannot be folded away.
    affine(&model.weights[layer.wHh], &model.weights[layer.bHh], hidden, gates,
           width, recurrent);

    for (U32 i = 0U; i < width; ++i) {
        const F32 reset = sigmoid(projected[i] + recurrent[i]);
        const F32 update = sigmoid(projected[width + i] + recurrent[width + i]);
        const F32 candidate = std::tanh(projected[(2U * width) + i]
                                        + (reset * recurrent[(2U * width) + i]));
        hidden[i] = ((hidden[i] - candidate) * update) + candidate;
    }
}

void head(const Model& model, const F32* hidden, F32* out) {
    const U32 width = model.layers[model.nLayers - 1U].hidden;
    affine(&model.weights[model.headW], &model.weights[model.headB], hidden,
           model.nOutputs, width, out);
}

}  // namespace Gru
}  // namespace Sentinel
