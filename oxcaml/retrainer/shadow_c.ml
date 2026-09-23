(* Section 72 / E5-e: the C-callable surface over 59's shadow model file.
 *
 * (!) THIS MODULE CARRIES NO ARITHMETIC OF ITS OWN, and 59's module is NOT edited.
 * Shadow59 was measured under [@zero_alloc strict] at Stage 59 and its figures
 * stand against the file they were measured on; adding a C surface to it would
 * invalidate that. So the surface is here, and it copies bytes and calls
 * Shadow59.write_shadow. Same discipline cycle_c.ml applied to deep_f32.ml.
 *
 * (!) THE BUFFERS ARE THE CALLER'S. Every Bigarray arriving here is wrapped by the
 * C++ side over memory it owns, with CAML_BA_EXTERNAL, and NONE of them outlives
 * the call that brought it. Nothing is stored. 47.6's C boundary: fixed-size
 * scalars, Bigarray and status codes.
 *
 * (!) EVERY EXCEPTION IS CAUGHT HERE, and this module takes retrainer.ml:41-44's
 * pattern rather than cycle_c.ml's. cycle_c.ml has no guard at all, although
 * sentinel_cycle.h:6 claims CPP-25 compliance for its surface -- found by reading
 * at Section 72 and recorded at 72.2 rather than fixed here, because cycle_c.ml is
 * a measured module too. The new surface is written the way 47.6 requires:
 * structurally impossible, not merely avoided.
 *
 * (!) THE GUARD COSTS A CLOSURE, so the registered wrappers are NOT annotated
 * [@zero_alloc strict] -- exactly as retrainer.ml's init/feed/step are not. The
 * copy loops, which are the part 54.2a's list governs, ARE annotated. That split
 * is retrainer.ml's and is not a new position. *)


(* D82: every element access below is bounds-checked. `acc.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and all 56 `[@zero_alloc strict]` sites still
   hold. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc

type u8 = Shadow59.u8
type f32 = Shadow59.f32

(* Status codes. Mirror sentinel_shadow.h; the C side never sees anything else.
   0, -1 and -2 are Shadow59.write_shadow's own returns, passed through unchanged
   (shadow59.ml:79-80) rather than remapped, so one code means one thing on both
   sides of the boundary. *)
let ok            = 0
let err_shape     = -1
let err_capacity  = -2
let err_internal  = -6

(* (!) The only place an exception may be turned into a status. Nothing below
   raises past this. retrainer.ml:41-44 is the precedent and it is not re-argued. *)
let guard f = try f () with _ -> err_internal

(* (!) THE CANDIDATE'S LENGTH IS THE FLYING FILE'S LENGTH, and it is remembered
   here rather than re-derived from the header. 59's premise is that the candidate
   IS the flying file with new weights -- same architecture, same shapes, every
   header field but two already correct (shadow59.ml:3-11) -- so the length is
   known exactly at load time.

   Deriving it instead was tried and was wrong: header_bytes + channels + weights
   is 1,276 B for p1.bin, whose real length is 1,396 B, because it omits the
   trailing PARAMS block (PARAM_FIXED_BYTES 96 plus 8 per channel,
   Monitor.hpp:38-43). Detector::load returned TRUNCATED. Remembering the extent
   needs no knowledge of the layout at all, and a layout this module does not know
   is a layout it cannot get wrong. *)
let loaded = ref 0

(* The flying file's bytes, copied in. Fixed extent, checked before the copy. *)
let[@zero_alloc strict] copy_in (src : u8) n =
  for i = 0 to n - 1 do
    U8.set Shadow59.file i (U8.get src i)
  done

let load (src : u8) =
  guard (fun () ->
    let n = Bigarray.Array1.dim src in
    if n <= Shadow59.header_bytes then err_shape
    else if n > Shadow59.max_file then err_capacity
    else begin copy_in src n; loaded := n; ok end)

(* 59's write, unchanged. The weight count is the caller's buffer extent, so a
   short or long weights block is refused by shadow59.ml:79 rather than here. *)
let write (w : f32) =
  guard (fun () -> Shadow59.write_shadow w (Bigarray.Array1.dim w))

(* The weights region the header declares must lie inside what was loaded. This
   is not the length -- it is the cross-check that the header and the file agree,
   so a corrupt header is refused here rather than by Detector::load later. *)
let[@zero_alloc strict] weights_end () =
  Shadow59.header_bytes
  + Shadow59.get_u32 Shadow59.off_channels_bytes
  + Shadow59.get_u32 Shadow59.off_weights_bytes

let[@zero_alloc strict] copy_out (dst : u8) n =
  for i = 0 to n - 1 do
    U8.set dst i (U8.get Shadow59.file i)
  done

(* (!) RETURNS THE BYTE COUNT, POSITIVE, not a status of 0. A candidate of zero
   bytes is not a thing, so positive-is-length and negative-is-refusal is
   unambiguous, and it keeps the surface at three entry points. *)
let export (dst : u8) =
  guard (fun () ->
    let total = !loaded in
    if total <= 0 then err_shape                    (* nothing loaded yet *)
    else if weights_end () > total then err_shape   (* header disagrees with the file *)
    else if Bigarray.Array1.dim dst < total then err_shape
    else begin copy_out dst total; total end)

(* (!) THE CYCLE'S OWN LOSS, EXPOSED HERE AND NOT IN cycle_c.ml. HO2's sanity
   report needs it (the owner's ruling of 2026-09-21: candidate_bytes,
   candidate_crc32, steps_run, final_loss, samples_seen). 61's C surface predates
   that report and has five entry points, none of them a loss; cycle_c.ml is a
   module measured at Stage 61 and this section does not edit it. So the accessor
   sits on the new surface, reading Deep_f32.loss_out (deep_f32.ml:90) directly.

   It is an accessor and nothing else -- no arithmetic, no accumulation. If a
   later section gives 61's surface a metrics entry of its own, this one should
   move there and be deleted here rather than duplicated. *)
let[@zero_alloc strict] copy_loss (out : f32) =
  F32.set out 0 (Array.unsafe_get Deep_f32.loss_out 0)

let loss (out : f32) =
  guard (fun () ->
    if Bigarray.Array1.dim out < 1 then err_shape
    else begin copy_loss out; ok end)

(* (!) THE SAME NARROW OPT-OUT E1 AND 61 TOOK, AND FOR THE SAME REASON --
   retrainer.ml:102-114 is the precedent and cycle_c.ml:68-78 restated it; neither
   is re-argued here. OxCaml's stdlib flags Callback.register as multidomain-unsafe
   and points at Callback.Safe.register, whose signature demands a portable
   closure. The thing it warns about cannot happen by construction: this process is
   SINGLE-DOMAIN, which is D70 consequence 2 and consequence 9 -- a requirement,
   not a preference. Domain.spawn is never called. If that ever stops being true
   this opt-out is wrong and the alert should be allowed to fire.

   Everything else builds at -warn-error +a -alert @all, so this is the one
   exception in this module and any other alert is a build failure. *)
let[@alert "-unsafe_multidomain"] () =
  Callback.register "sentinel_shd_load" load;
  Callback.register "sentinel_shd_write" write;
  Callback.register "sentinel_shd_export" export;
  Callback.register "sentinel_shd_loss" loss
