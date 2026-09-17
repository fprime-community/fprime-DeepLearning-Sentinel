/* Section 60 AS2's reference only. It is NOT used by deep_f32.ml, which reaches
   no C external for the square root -- that is the whole point of 60.2 route 2.
   This exists so the claim is checked against a correctly-rounded sqrtf rather
   than asserted from the precision argument. */
#include <math.h>
float caml_sqrtf_ref(float x) { return sqrtf(x); }
