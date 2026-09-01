# `model.bin` - the normative format specification

**Frozen 2026-09-01 by `docs/DECISIONS.md` D30. Version 1.**

This document defines the file. It is the contract between the Python training
toolkit (`src/sentinel_export/`) and the C++ flight loader (`flight/`), and it is
normative: where this document and any implementation disagree, **this document is
right and the implementation is a defect**. `src/sentinel_export/` owns it, as that
package's docstring has said since it was created.

The C++ loader is written against this specification and imports nothing from the
Python side. No Python, no ML framework and no interpreter goes to space; a flight
review board approves a C++ program reading a data file, which is how F's own
parameter database already works (Objective.md 4.3 step 2).

**What this format is not.** Objective.md 14.10 proposed "a quantized,
self-describing FlatBuffer, TFLite-Micro compatible". That proposal is superseded by
D30 and the reasons are recorded there. Nothing here is quantized, nothing is
generated, and there is no third-party parser.

---

## 1. Conventions

- **Little-endian**, every multi-byte field, without exception.
- **Four-byte alignment** of every block start. Fields are nevertheless read and
  written with `memcpy`, never by casting a pointer into the byte stream, so the
  format imposes no alignment requirement on the reader's buffer (F' CPP-10 discourages
  `reinterpret_cast`; `memcpy` needs no justification and optimises to the same code).
- `U8`, `U16`, `U32`, `F32`, `F64` are F's names for `uint8_t`, `uint16_t`,
  `uint32_t`, `float`, `double` (`nasa/fprime` v4.3.0 `docs/reference/numerical-types.md`).
- **Reserved fields are zero and the reader refuses a non-zero one.** A reserved
  field that is tolerated is a field that drifts silently; refusing it is the same
  discipline `detectors._load_weights` already applies to `cell` (D16).
- Every count is a `k/n` fact about the file, and every field is **verified against
  the payload, never merely read** (section 6).

---

## 2. File layout

Four blocks, in this order, with no padding between them.

```
  +== model.bin =====================================================+
  |  HEADER      64 bytes, fixed, self-protecting                    |
  +------------------------------------------------------------------+
  |  CHANNELS    20 * n_channels bytes         )                     |
  +---------------------------------------------)  one static_crc32  |
  |  WEIGHTS     4 * n_parameters bytes        )                     |
  +------------------------------------------------------------------+
  |  PARAMS      96 + 8 * n_channels bytes, own param_crc32          |
  |              -- separately replaceable in orbit                  |
  +==================================================================+
```

**Why PARAMS carries its own CRC, and it is the design's load-bearing idea.** A
threshold is a fitted quantity belonging to the model it was measured against and to
the spacecraft, and it must be recalibrable in orbit without retraining
(Objective.md 10.2 fix 4, 14.10; D29 consequence 3). A single CRC over the whole file
would force a 278 KiB rewrite and a full re-verification to change one number. With a
separate `param_crc32`, an in-orbit recalibration is a fixed-length overwrite of the
PARAMS block plus an eight-byte header patch, and the weights are never touched.

Sizes for the flown configuration - 12 channels, two GRU layers of 80, `l_p = 10`,
71,160 parameters (`docs/MODELS.md` 3):

```
  HEADER         64
  CHANNELS      240
  WEIGHTS   284,640
  PARAMS        192
  ----------------
  TOTAL     285,136 bytes  =  278.45 KiB
```

---

## 3. HEADER - 64 bytes

```
  off  size  field            type      value / meaning
  ---  ----  ---------------  --------  -------------------------------------
    0     4  magic            U8[4]     'S','N','T','L'  (53 4E 54 4C)
    4     2  format_version   U16       1
    6     2  header_bytes     U16       64
    8     2  arch_id          U16       1 = GRU   (2, 3 reserved: LSTM, TCN)
   10     2  gate_order_id    U16       1 = (reset, update, new)
   12     2  n_layers         U16       2
   14     2  n_channels       U16       12
   16     2  n_inputs         U16       n_channels + n_exogenous
   18     2  n_exogenous      U16       0 until D6's command inputs are measured
   20     2  window           U16       250   (recorded; see below)
   22     2  n_predictions    U16       10    (telemanom's l_p)
   24     2  hidden[0]        U16       80
   26     2  hidden[1]        U16       80
   28     2  hidden[2]        U16       0     (MAX_LAYERS = 4)
   30     2  hidden[3]        U16       0
   32     4  channel_bytes    U32       20 * n_channels
   36     4  weight_bytes     U32       4 * n_parameters
   40     4  param_bytes      U32       96 + 8 * n_channels
   44     4  static_crc32     U32       over CHANNELS ++ WEIGHTS
   48     4  param_crc32      U32       over PARAMS
   52     4  reserved0        U32       0
   56     4  reserved1        U32       0
   60     4  header_crc32     U32       over bytes [0, 60)
```

**`arch_id` and `gate_order_id` are named, not inferred.** `docs/MODELS.md` 3: "Gate
order must be in the format, not in the reader's memory ... A loader that guesses
wrong produces a model that runs, converges to nothing useful, and fails silently."
`gate_order_id = 1` means the stacking `reference.GRU_GATES == ("reset", "update",
"new")`, pinned by `tests/test_reference_equivalence.py:166`.

**`window` is recorded and unused by the flight loop.** The GRU carries its state
across ticks, so there is no 250-step history buffer; that was the TCN's ring, and
the TCN was declined at D28. The field travels because the ground side needs it to
reproduce the fit and because a reader that cannot see it cannot check it.

**`header_crc32` makes the header self-protecting.** It is verified *before* any
length field is used, so a corrupted `weight_bytes` can never drive an over-read.
This is the first hook WI9's Level 1 safe failure mode (D5, Objective.md 14.10)
builds on.

---

## 4. CHANNELS - 20 bytes per channel

```
  off  size  field   type      meaning
  ---  ----  ------  --------  --------------------------------------------
    0     4  id      U32       the F' telemetry channel id (FwChanIdType)
    4    16  name    char[16]  NUL-padded ASCII; name[15] must be 0
```

Repeated `n_channels` times, **in the model's channel order** - the order the
forecaster was fitted in, and the order every weight row assumes. Objective.md 4.1
lists the channel map as part of what `model.bin` carries; this is it.

---

## 5. WEIGHTS - float32, `reference.Weights.arrays()` order

Plain float32 arrays, C order (row-major), concatenated with no headers, in exactly
the order `reference.Weights.arrays()` emits them - layers in index order, then the
head:

```
  l0_w_ih  (3*H0, n_inputs)     l0_w_hh  (3*H0, H0)   l0_b_ih  (3*H0,)   l0_b_hh  (3*H0,)
  l1_w_ih  (3*H1, H0)           l1_w_hh  (3*H1, H1)   l1_b_ih  (3*H1,)   l1_b_hh  (3*H1,)
  head_w   (n_predictions * n_channels, H_last)       head_b   (n_predictions * n_channels,)
```

**Both bias vectors are stored unsummed, and for the GRU that is not optional.**
`docs/MODELS.md` 3, amended 2026-08-28: the third gate is
`n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`, so `b_hn` sits **inside the reset
product** and cannot be folded into `b_in`. An LSTM's two bias vectors may be summed;
a GRU's may not. This is the mistake a C++ author with the LSTM in their hands is most
likely to make, and it is pinned two-sided by
`tests/test_reference_equivalence.py:202` - folding diverges in general and agrees
exactly once the reset gate is saturated open.

Within `w_ih` and `w_hh` the three gate blocks are stacked along axis 0:
rows `[0:H]` reset, `[H:2H]` update, `[2H:3H]` new. Same for `b_ih` and `b_hh`.

**No dropout line.** It is identity at inference, exports no parameters, and a flight
component dropping a random 30% of its hidden units per cycle would violate
Objective.md 11 rule 5 outright. **No weight normalisation, no batch normalisation,
no state shape.**

---

## 6. PARAMS - the separately replaceable block

```
  off  size  field           type       value / meaning
  ---  ----  --------------  ---------  ------------------------------------
    0     2  param_version   U16        1
    2     2  norm_policy     U16        0 = identity (D2). No other value is
                                        accepted by a version-1 reader
    4     2  ewma_span       U16        105
    6     2  agreement       U16        1  (k-of-n; 1 is the maximum)
    8     2  persistence     U16        1  (N=1; no persistence filter)
   10     2  reserved0       U16        0
   12     4  warmup_steps    U32        2350 = window 250 + error_window 2100
   16     1  baseline_only   U8         0 or 1  (Objective.md 14.10, D5)
   17     1  tier            U8         1, 2 or 3  (Objective.md 14.10)
   18     2  reserved1       U16        0
   20     4  reserved2       U32        0
   24     8  threshold       F64        the calibrated global cut
   32    64  provenance      char[64]   NUL-padded ASCII; how it was fitted
   96   4*C  norm_offset     F32[C]     all 0.0 under D2
  ...   4*C  norm_scale      F32[C]     all 1.0 under D2
```

`param_bytes = 96 + 8 * n_channels`.

**`threshold` is F64, and that is deliberate.** The harness compares
`combined.astype(np.float64) >= threshold` (`harness.py:169-172`) against a threshold
produced by `np.quantile` in float64 (`detector.py:127-132`). Storing it as F32 would
round the cut and could flip a crossing at the boundary. The comparison is `>=`, not
`>`.

**`norm_policy`, `norm_offset` and `norm_scale` exist and are trivial.** Phase 1
normalisation is identity (D2, Objective.md 14.8), so the constants are 0.0 and 1.0.
The *slot* still has to exist, because a mission whose data is not ESA-preprocessed
will need it (`docs/MODELS.md` 3), and because Objective.md 14.10 requires
normalisation constants stored outside the weights as PrmDb-style parameters.

**`baseline_only` is mandatory** (Objective.md 14.10). It is the Level 1 switch: set,
the component runs the statistical baseline and never the network, whatever else the
file contains. WI9 wires it to the active-tier telemetry channel.

**`provenance` is not decoration.** D29: a threshold is a parameter with a
provenance, not a constant. Two calibrations of one rule on one set of residuals
differed nineteenfold (Objective.md 14.10, `docs/RESULTS.md` 6b), and a threshold
that suited a one-epoch model produced 3,548 alarm ranges on a trained one (D17). The
field records which fitting window and which procedure produced the number beside it.

### 6.1 Replacing PARAMS in orbit

`param_bytes` is fixed for a given model, so recalibration is:

```
  1. overwrite PARAMS in place            (192 bytes at the flown shape)
  2. patch header param_crc32             (4 bytes at offset 48)
  3. patch header header_crc32            (4 bytes at offset 60)
  4. reload; static_crc32 is unchanged and the weights were never read
```

WI10 owns the uplink path and the human approval around it (Objective.md 4.3 step 3).
This section owns only the guarantee that the format permits it.

---

## 7. CRC

**CRC-32/IEEE 802.3.** Reversed polynomial `0xEDB88320`, initial value
`0xFFFFFFFF`, final complement. Check value: `crc32("123456789") == 0xCBF43926`.

This is exactly `zlib.crc32` on the Python side, and exactly F's `Utils::Hash` on the
flight side - `nasa/fprime` v4.3.0 `Utils/Hash/Crc32/Crc32.hpp` describes
`crc32_ieee802_3_update` as "an implementation of the IEEE 802.3 CRC32 polynomial,
which is 0x82608EDB when expressed as a reverse reciprocal form, or 0xEDB88320 when
expressed in a reverse form", and notes that the one's complement is applied by the
`Hash` class rather than by the update function. `flight/` carries its own table so
WI8 needs no F' checkout; WI9 may substitute `Utils::Hash` and the value will not
change.

---

## 8. Reading, and refusing

The reader returns a status. **It never throws** - flight code is compiled
`-fno-exceptions` (F' CPP-25), so throwing is impossible by construction rather than
by discipline.

```
  OK                 the file is loaded and the detector is armed
  BAD_MAGIC          the first four bytes are not 'SNTL'
  BAD_VERSION        format_version is not 1
  BAD_HEADER_CRC     header_crc32 does not match bytes [0, 60)
  BAD_STATIC_CRC     static_crc32 does not match CHANNELS ++ WEIGHTS
  BAD_PARAM_CRC      param_crc32 does not match PARAMS
  BAD_ARCH           arch_id is not 1
  BAD_GATE_ORDER     gate_order_id is not 1
  BAD_SHAPE          the declared shape does not account for weight_bytes,
                     or n_inputs != n_channels + n_exogenous, or a reserved
                     field is non-zero, or a channel name is not terminated
  TOO_LARGE          the file exceeds a compile-time maximum in Config.hpp
  TRUNCATED          the file is shorter than its own header says
  BAD_NORM_POLICY    norm_policy is not 0
```

**Order of checks, and it matters.** Magic, then `format_version`, then
`header_crc32` - *before any length field is used*. Only then are `channel_bytes`,
`weight_bytes` and `param_bytes` trusted enough to bound a read. Then `TOO_LARGE`
against the compile-time maxima, then `BAD_SHAPE`, then the payload is streamed and
the two payload CRCs verified.

**One pass, and the loader adds no buffer of its own.** The reader is handed the
file's bytes and copies them once, into the arrays it will use; it allocates
nothing and holds no second copy (F' CPP-1). Work item 8's reader takes the whole
buffer from its caller because work item 8 has no filesystem -- `Os::File` and the
chunked feed are work item 9's, and the check order below is written so that a
chunked reader can be dropped in without changing which failure is reported first.

**Nothing is used before it is checked.** The detector is **not armed** until every
CRC verifies; on any non-`OK` status the model is left unusable, `step()` does
nothing and no warning can be emitted. That refusal is the hook work item 9's
Level 1 degrade-with-an-event is built on (D5, Objective.md 14.10): never fail the
topology.

**Every field is verified against the payload.** `weight_bytes` must equal exactly
the parameter count the declared shape implies; `n_inputs` must equal
`n_channels + n_exogenous`; `channel_bytes` must equal `20 * n_channels`;
`param_bytes` must equal `96 + 8 * n_channels`. A field written and not checked is
D16.

---

## 9. Determinism

Objective.md 11 rule 5 requires fixed memory, fixed compute per cycle, and the same
outputs from the same inputs. The build flags are part of the specification, not a
build-system detail:

```
  -std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
  never -ffast-math, never -Ofast
```

**`-ffp-contract=off` is the bit-identity guarantor.** By default a compiler may
contract `a*b + c` into a fused multiply-add, which rounds once instead of twice.
That is a different number, it differs per target and per optimisation level, and it
would silently break equivalence with the NumPy reference every result in this
project was measured against. It is switched off explicitly, at every optimisation
level.

**`-ffast-math` and `-Ofast` are forbidden outright.** They license reassociation,
finite-math assumptions and reciprocal substitution - each of which changes results,
and the last of which would break the `x >= 0` branch in the sigmoid.

**`-Wconversion -Werror` enforces the dtype map below.** The reference deliberately
mixes precisions; with implicit narrowing an error, every cast must be written where
the reference casts, so the compiler holds the map rather than the author's memory.

**The dtype map, which is not uniform and must not be made uniform.**

```
  GRU layers, head, residual, EWMA output      F32
  prediction aggregation (windows.py:209)      F64 accumulate -> F32
  EWMA numerator and denominator (telemanom)   F64          -> F32
  threshold comparison (harness.py:169)        F64
```

`telemanom.ewma` opens `np.asarray(values, dtype=np.float64)` and closes
`.astype(np.float32)`; `windows.aggregate_predictions` accumulates into a float64
`total` and returns float32. A C++ core that ran the decision layer in F32 throughout
would be a different detector.

**This makes F64 a hard requirement of the flight target.** Sentinel requires it, and
Phase 4's board selection inherits that constraint.

**(!) AMENDED 2026-09-01, by work item 9.** This paragraph previously continued: "F'
treats `F64` as a configurable platform type that a platform may switch off
(`FW_HAS_F64` in `PlatformTypes.h`, `nasa/fprime` v4.3.0
`docs/reference/numerical-types.md`)." **That is not true of the code.** At v4.3.0 `F64`
is defined unconditionally at `Fw/Types/BasicTypes.h:86` -- "64-bit floating point
(double). Required for compiler-supplied double promotion" -- and it was unconditional at
v4.2.2 as well. `FW_HAS_F64` appears exactly once in the entire framework, and it is the
documentation table this project read: `docs/reference/numerical-types.md:35`. F'
documents a macro its code does not define. The original sentence is kept above because
it records what was believed and where it came from. See `docs/MODELS.md` 20.2 correction
13 and D35.

**The constraint that replaces it, from what F' actually provides.** A requirement built
on a symbol that does not exist is not a requirement, so it is restated here against the
two switches F' really has. Both live in
`default/config/FPrimeNumericalConfig.h`, which is a **project-configurable** header -- a
mission overrides it through `config_directory` in `settings.ini` -- so both are things a
Phase 4 board selection can genuinely get wrong.

```
  FW_HAS_64_BIT                   must be 1     default (1)
  SKIP_FLOAT_IEEE_754_COMPLIANCE  must be 0     default (0)
```

**`FW_HAS_64_BIT` is the switch that actually bites**, and it is not about `F64` at all:
it guards `U64` and `I64` at `Fw/Types/BasicTypes.h:77`. This core needs `U64` in sixteen
places outside the types shim -- `Detector::m_steps` is the tick counter that drives the
warm-up gate and the prediction ring's modulo, and `ModelFile`'s `readU64` assembles the
threshold's F64 bit pattern before `memcpy`. On a platform with `FW_HAS_64_BIT (0)` this
core does not compile, which is the correct outcome and is now an `#error` with a sentence
rather than a cascade of unknown-type diagnostics.

**`SKIP_FLOAT_IEEE_754_COMPLIANCE` is F's own name for the property section 9 depends on.**
Set to 1 it tells F' not to check that the platform's floating point is IEEE 754. Every
equivalence result in this project -- 1.788e-07 against the NumPy reference, 0.000e+00 on
the Level 1 baseline -- assumes it is. The shim asserts the property directly with
`numeric_limits<F32>::is_iec559` and `numeric_limits<F64>::is_iec559`, which is stronger
than reading a configuration value, and refuses the configuration value too so that a
mission which sets it learns why here rather than from a drifting number in orbit.

**F64 itself constrains nothing**, because F' always provides it. That half of the
original paragraph was the half that was wrong.

---

## 10. Seeded weights, for the golden vectors

Tiers G1, G2 and G3 (`docs/MODELS.md` 19) use weights generated rather than trained,
so a committed vector can be regenerated on a fresh clone and cannot drift silently.
The recipe is fixed here so it is reproducible:

```
  rng = numpy.random.default_rng(seed)
  k   = 1.0 / sqrt(hidden)                      # PyTorch's own GRU initialiser
  array = rng.uniform(-k, k, shape).astype(numpy.float32)
```

Arrays are drawn in `reference.Weights.arrays()` order from one generator, so the
sequence is fixed by the seed and the shape list alone. `k` uses the hidden size of
the layer the array belongs to; the head uses `H_last`.

---

## 11. Versioning

`format_version` is bumped for any change to the byte layout, and a version-1 reader
refuses anything else. `param_version` is independent: a recalibration that changes
only the PARAMS block bumps it and leaves `format_version` alone, which is what lets a
threshold have a provenance separate from the model it belongs to (D29).

Adding a field is a `format_version` bump. Reserved fields are not a growth mechanism
- a version-1 reader refuses a non-zero reserved field - they are alignment padding
that is checked.
