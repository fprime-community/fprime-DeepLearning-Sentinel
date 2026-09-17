// Section 54 FT3: is the OCaml cell the FLOWN cell?
//
// (!) IT LINKS THE FLIGHT CORE, NOT A RESTATEMENT OF IT. Sentinel::Gru::step is the same
// function the F' component runs and the golden vectors pin. A comparison against a second
// transcription would compare two of my own transcriptions to each other.
//
// The Model is filled directly rather than loaded from a model.bin: G6 is about the cell's
// algebra, and routing through the loader would drag in the file format, the CRC and the
// param block, none of which this prediction is about.
//
// Weights come from the SAME deterministic recurrence gru_check.ml uses, so both sides see
// bit-identical inputs -- in their own precision, which is the point of 48.3's departure.
#include "sentinel/Gru.hpp"
#include "sentinel/ModelFile.hpp"
#include "sentinel/Config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

namespace {

const uint32_t HS = 80U;
const uint32_t INS = 16U;
const uint32_t GW = 3U * HS;

// gru_check.ml's `fill`, transcribed. Integer arithmetic, so both sides agree exactly.
void fill(Sentinel::F32* dst, uint32_t count, double scale, int off)
{
    long s = 12345L + off;
    for (uint32_t i = 0U; i < count; ++i) {
        s = ((s * 1103515245L) + 12345L) & 0x3FFFFFFFL;
        dst[i] = static_cast<Sentinel::F32>(
            scale * ((static_cast<double>(s % 2000L) / 1000.0) - 1.0));
    }
}

}  // namespace

int main()
{
    static Sentinel::Model model;
    model.nLayers = 1U;
    model.nInputs = INS;
    model.hidden[0] = HS;

    Sentinel::LayerView& L = model.layers[0];
    L.hidden = HS;
    L.inputs = INS;
    L.wIh = 0U;
    L.wHh = L.wIh + (GW * INS);
    L.bIh = L.wHh + (GW * HS);
    L.bHh = L.bIh + GW;

    fill(&model.weights[L.wIh], GW * INS, 0.3, 1);
    fill(&model.weights[L.wHh], GW * HS, 0.3, 2);
    fill(&model.weights[L.bIh], GW, 0.2, 3);
    fill(&model.weights[L.bHh], GW, 0.2, 4);

    Sentinel::F32 x[INS];
    Sentinel::F32 hidden[HS];
    fill(x, INS, 1.0, 5);
    fill(hidden, HS, 1.0, 6);

    Sentinel::F32 projected[GW];
    Sentinel::F32 recurrent[GW];
    Sentinel::Gru::step(model, 0U, x, hidden, projected, recurrent);

    // the OCaml side's h', one per line
    std::FILE* f = std::fopen("h_new_f32.txt", "r");
    if (f == nullptr) { std::fprintf(stderr, "run s48_check first\n"); return 2; }
    double worst = 0.0;
    uint32_t worstIdx = 0U;
    uint32_t identical = 0U;
    for (uint32_t i = 0U; i < HS; ++i) {
        double ml = 0.0;
        if (std::fscanf(f, "%lf", &ml) != 1) { std::fprintf(stderr, "short file\n"); return 2; }
        // both sides are F32; compare the F32 values exactly, then by magnitude
        if (static_cast<Sentinel::F32>(ml) == hidden[i]) { ++identical; }
        const double d = std::fabs(ml - static_cast<double>(hidden[i]));
        if (d > worst) { worst = d; worstIdx = i; }
    }
    std::fclose(f);

    std::printf("== Section 54, FT3: the F32 OCaml cell against flight/'s Gru::step ==\n");
    std::printf("   hidden %u, inputs %u, one layer, one timestep, BOTH AT F32\n", HS, INS);
    std::printf("   bit-identical outputs: %u of %u\n", identical, HS);
    std::printf("   worst |OCaml(F32) - flight(F32)| = %.6e  at index %u\n", worst, worstIdx);
    const char* v = (identical == HS) ? "HOLD"
                    : ((worst <= 1.2e-7) ? "NO VERDICT" : "FAIL");
    std::printf("   band: all %u bit-identical HOLD, <=1.2e-07 (one F32 ulp) NO VERDICT, "
                "else FAIL -> %s\n", HS, v);
    return (identical == HS) ? 0 : ((worst <= 1.2e-7) ? 1 : 2);
}
