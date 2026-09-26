(* Section 61 / E5-c: the C-callable surface over 60's float32 cycle.
 *
 * (!) THIS MODULE CARRIES NO ARITHMETIC OF ITS OWN. It copies across the boundary
 * and calls Deep_f32. The cycle is 60's and is not restated here, which is the
 * same discipline 52 applied to 48 and 60 applied to 52.
 *
 * (!) THE BUFFERS ARE THE CALLER'S. Every Bigarray arriving here is wrapped by the
 * C++ side over memory it owns, with CAML_BA_EXTERNAL, and NONE of them outlives
 * the call that brought it. Nothing is stored. 47.6's C boundary: fixed-size
 * scalars, Bigarray and status codes.
 *
 * (!) EVERY ENTRY POINT RETURNS AN INT. The stubs convert to int32_t, which is
 * sentinel_retrainer.h:44-51's rule and CPP-3's. *)


(* D82: element access below is bounds-checked via `acc.ml`. *)
open Acc

type f32 = Deep_f32.f32

let ok = 0
let err_shape = -1
let err_budget = -2
let err_internal = -6

(* (!) THE ONLY PLACE AN EXCEPTION MAY BE TURNED INTO A STATUS, and until 2026-09-22
   this module had none -- `sentinel_cycle.h:6` claimed CPP-25 compliance for a
   surface with five unguarded entry points, recorded at 72.2 and carried at 72.9 as
   owed. `retrainer.ml:41-44` is the precedent and `shadow_c.ml:39-40` already took
   it; this is the third and last surface to do so.

   D82 made it load-bearing rather than tidy. Every element access in this module is
   now bounds-checked, so a bad index raises `Invalid_argument` instead of reading
   past an array -- and an exception reaching `caml_callback` from a C++ caller built
   `-fno-exceptions` is undefined behaviour. The check is only an improvement if what
   it raises cannot cross the boundary. `Out_of_memory` and `Stack_overflow` are
   caught for the same reason they are in `retrainer.ml`: a status the caller can act
   on beats undefined behaviour.

   The guard costs a closure, which is why the five registered wrappers below are not
   annotated and the arithmetic they call still is -- `shadow_c.ml:21-24`'s finding,
   unchanged. *)
let guard f = try f () with _ -> err_internal

(* Seed a reproducible starting model. A real deployment uplinks one; this rung is
   about the crossing, not about where the weights come from. *)
let[@zero_alloc strict] seed_from salt =
  let s = ref (12345 + salt) in
  for i = 0 to Deep_f32.n_params - 1 do
    s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
    F32.set Deep_f32.p i
      (0.1 *. ((float_of_int (!s mod 2000) /. 1000.0) -. 1.0))
  done;
  for i = 0 to Deep_f32.hs - 1 do
    F32.set Deep_f32.h_init0 i 0.0;
    F32.set Deep_f32.h_init1 i 0.0
  done;
  for i = 0 to Deep_f32.n_out - 1 do
    F32.set Deep_f32.coeff_y i 1.0
  done

let init salt = guard (fun () -> seed_from salt; Deep_f32.adam_reset (); ok)

(* The caller's window, copied in. Fixed extent, checked before the copy.

   D85: THE WINDOW NOW CARRIES ITS TARGET. It is (t_max + P) rows of `ins` values:
   the first t_max rows are the input sequence and the last P rows are the future
   block the forecast is trained against -- lstm.py's SequenceSampler pair, laid out
   in the head's (prediction, channel) order, which is row-major over those P rows.
   Until D85 it was t_max rows and nothing to learn from. *)
let[@zero_alloc strict] copy_window (w : f32) n_in =
  for i = 0 to n_in - 1 do
    F32.set Deep_f32.x_seq i (F32.get w i)
  done;
  for i = 0 to Deep_f32.n_out - 1 do
    F32.set Deep_f32.tgt i (F32.get w (n_in + i))
  done

let load_window (w : f32) =
  guard (fun () ->
    let n = Bigarray.Array1.dim w in
    let n_in = Deep_f32.t_max * Deep_f32.ins in
    if n <> n_in + Deep_f32.n_out then err_shape
    else begin copy_window w n_in; ok end)

let run budget t_steps admit =
  guard (fun () ->
    if budget <= 0 || t_steps <= 0 || t_steps > Deep_f32.t_max then err_budget
    else begin Deep_f32.run_cycle budget t_steps admit; ok end)

(* Negative is a refusal here too: Retrainer.cpp:225 already reads a negative
   step count as "the cycle did not run". *)
let steps () = guard (fun () -> Deep_f32.steps_taken.(0))

(* The best-weights copy is what a cycle offers, not the live parameters. D73 c.2. *)
let[@zero_alloc strict] copy_best (out : f32) n =
  for i = 0 to n - 1 do
    F32.set out i (F32.get Deep_f32.best i)
  done

let export (out : f32) =
  guard (fun () ->
    let n = Bigarray.Array1.dim out in
    if n <> Deep_f32.n_params then err_shape
    else begin copy_best out n; ok end)

(* (!) THE SAME NARROW OPT-OUT E1 TOOK, AND FOR THE SAME REASON -- retrainer.ml:102-114
   is the precedent and it is not re-argued here. OxCaml's stdlib flags
   Callback.register as multidomain-unsafe and points at Callback.Safe.register,
   whose signature demands a portable closure. The thing it warns about cannot
   happen by construction: this process is SINGLE-DOMAIN, which is D70 consequence
   2 and consequence 9 -- a requirement, not a preference. Domain.spawn is never
   called. If that ever stops being true this opt-out is wrong and the alert should
   be allowed to fire.

   Everything else builds at -warn-error +a -alert @all, so this is the one
   exception in this module and any other alert is a build failure. *)
let[@alert "-unsafe_multidomain"] () =
  Callback.register "sentinel_cyc_init" init;
  Callback.register "sentinel_cyc_load" load_window;
  Callback.register "sentinel_cyc_run" run;
  Callback.register "sentinel_cyc_steps" steps;
  Callback.register "sentinel_cyc_export" export
