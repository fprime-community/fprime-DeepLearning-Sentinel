// The GRU forward pass, transcribed from `src/sentinel_models/reference.py`.
//
// Every line here has a named counterpart in `reference.gru_cell` and
// `reference.gru_layer`, because the C++ is a transcription and is verified
// against them at 1e-5 (D15, `docs/MODELS.md` 19).
#ifndef SENTINEL_GRU_HPP
#define SENTINEL_GRU_HPP

#include "sentinel/ModelFile.hpp"
#include "sentinel/Types.hpp"

namespace Sentinel {
namespace Gru {

//! Logistic, in the branch form that does not overflow for large |x|.
//!
//! `reference.py:394-401` uses this form and not `1/(1+exp(-x))`, and the
//! transcription follows the form that was measured rather than the one that is
//! shorter.
F32 sigmoid(F32 x);

//! One tick of one layer. `hidden` is read and overwritten in place.
//!
//!     r  = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
//!     z  = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
//!     n  = tanh   (W_in x + b_in + r * (W_hn h + b_hn))
//!     h' = (h - n) * z + n
//!
//! **`b_hn` sits inside the reset product.** The recurrent product is formed once
//! for all three gates with the *full* `b_hh` added, and the third block of it is
//! then multiplied by the reset gate. A loader that folded `b_hh` into `b_ih`
//! would be wrong on exactly that gate and nowhere else (`docs/MODELS.md` 3, D26).
//!
//! The state update is ATen's algebraic form `(h - n) * z + n` rather than the
//! textbook `(1 - z) n + z h`. They are equal in exact arithmetic and not in
//! float32, and `reference.py:475` uses ATen's -- which is what every measurement
//! in this project was taken against.
//!
//! `projected` and `recurrent` are caller-supplied scratch of at least
//! `3 * hidden` elements. Nothing is allocated (F' CPP-1).
void step(const Model& model, U32 layerIndex, const F32* input, F32* hidden,
          F32* projected, F32* recurrent);

//! The output head: `h_last @ head_w.T + head_b`.
//!
//! Row `step * n_channels + channel` is the forecast for that channel that many
//! steps ahead, which is `flat.reshape(steps, n_predictions, n_channels)` in C
//! order (`reference.py:565-566`).
void head(const Model& model, const F32* hidden, F32* out);

}  // namespace Gru
}  // namespace Sentinel

#endif  // SENTINEL_GRU_HPP
