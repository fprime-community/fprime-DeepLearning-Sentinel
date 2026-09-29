(* Section 59 / E5-a: the shadow model file, written by the retrainer.
 *
 * (!) THE SHADOW IS THE FLYING FILE WITH NEW WEIGHTS (59.2). The retrainer trains
 * the same architecture at the same shapes, so every header field except two is
 * already correct. The write is src/sentinel_export/writer.py:185-208's
 * replace_params shape, with the WEIGHTS block substituted for the PARAMS block:
 *
 *     copy the flying file's bytes
 *     overwrite the weights payload
 *     patch static_crc32 (44) and header_crc32 (60)
 *     leave the shapes, the channel records and PARAMS untouched
 *
 * D30's freeze is not touched: no field is added, moved or resized, and
 * format_version stays 1. ModelFile.hpp:17-23 is the precedent -- PARAM_VERSION
 * went 1 to 2 the same way.
 *
 * (!) STOP 21 IS NOT ENGAGED. Int32.bits_of_float and its shifts are integer and
 * bit operations, not float32 arithmetic, and 54.2a's list governs the latter.
 * Measured: they hold [@zero_alloc strict] in this switch, flambda2 unboxing the
 * local int32. No C external is called.
 *
 * (!) STOP 29: the shape fields are never written. The offsets below are read-only
 * except for 44 and 60. *)


(* D82: every element access below is bounds-checked. `acc_int.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and every `[@zero_alloc strict]` site still
   holds. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc_int

let max_file = 1 lsl 20                 (* 1 MiB; the flown file is 268,224 B *)

type u8 = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
type f32 = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

let file : u8 =
  Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout max_file

(* Header offsets, derived from src/sentinel_export/format.py:64, "<4s10H4H8I". *)
let off_channels_bytes = 32
let off_weights_bytes = 36
let off_static_crc = 44
let off_header_crc = 60
let header_bytes = 64

let mask = 0xFFFFFFFF

(* zlib's CRC-32, the one writer.py uses. Table built once at init. *)
let crc_table = Array.make 256 0
let () =
  for n = 0 to 255 do
    let c = ref n in
    for _ = 0 to 7 do
      c := if !c land 1 <> 0 then 0xEDB88320 lxor (!c lsr 1) else !c lsr 1
    done;
    crc_table.(n) <- !c land mask
  done

let[@inline] get_u8 off = U8.get file off

let[@zero_alloc strict] get_u32 off =
  (get_u8 off) lor ((get_u8 (off + 1)) lsl 8)
  lor ((get_u8 (off + 2)) lsl 16) lor ((get_u8 (off + 3)) lsl 24)

let[@zero_alloc strict] put_u32 off v =
  U8.set file off (v land 0xFF);
  U8.set file (off + 1) ((v lsr 8) land 0xFF);
  U8.set file (off + 2) ((v lsr 16) land 0xFF);
  U8.set file (off + 3) ((v lsr 24) land 0xFF)

let[@zero_alloc strict] crc_range lo n =
  let c = ref mask in
  for i = lo to lo + n - 1 do
    let idx = (!c lxor (get_u8 i)) land 0xFF in
    c := (Array.unsafe_get crc_table idx) lxor (!c lsr 8)
  done;
  (!c lxor mask) land mask

(* The whole operation. Returns 0, or a negative code -- every entry point is an
   int32-shaped status, which is the C boundary's rule (sentinel_retrainer.h:44-51)
   even though this one is not yet crossed by C. *)
let[@zero_alloc strict] write_shadow (w : f32) n_weights =
  let cb = get_u32 off_channels_bytes in
  let wb = get_u32 off_weights_bytes in
  if wb <> 4 * n_weights then (-1)
  else if header_bytes + cb + wb > max_file then (-2)
  else begin
    let w_off = header_bytes + cb in
    for i = 0 to n_weights - 1 do
      let bits = Int32.bits_of_float (F32.get w i) in
      let base = w_off + (4 * i) in
      for s = 0 to 3 do
        U8.set file (base + s)
          ((Int32.to_int (Int32.shift_right_logical bits (8 * s))) land 0xFF)
      done
    done;
    (* static_crc32 is over the channel records and the weights together, which
       are contiguous. writer.py:152. *)
    put_u32 off_static_crc (crc_range header_bytes (cb + wb));
    (* header_crc32 is over the first 60 bytes only. writer.py:177-178. *)
    put_u32 off_header_crc (crc_range 0 off_header_crc);
    0
  end
