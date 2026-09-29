(* D85: 56's healthy-window rule, crossed into C for the retrainer component.

   window56.ml is the rule and is unchanged in what it decides; this module is only its
   status-returning surface, in cycle_c.ml's and shadow_c.ml's shape: every entry point
   is wrapped in `guard`, so an exception becomes a status and never reaches a
   -fno-exceptions caller. The arguments are checked here -- window56's `push` takes 0
   or 1 and nothing else (window56.ml:48), and that is enforced at the boundary rather
   than trusted. *)
open Acc_int

let ok = 0
let err_arg = -1
let err_internal = -6

let guard f = try f () with _ -> err_internal

let push lim emit =
  guard (fun () ->
    if (lim <> 0 && lim <> 1) || (emit <> 0 && emit <> 1) then err_arg
    else begin Window56.push lim emit; ok end)

(* 1 admits, 0 refuses -- window56's `admits_all`, the rule 56 scored. *)
let admits () = guard (fun () -> Window56.admits_all ())

let reset () = guard (fun () -> Window56.reset (); ok)

(* The flying model's calibrated nominal rate, in parts per million. Once, at init. *)
let configure ppm =
  guard (fun () ->
    if ppm <= 0 then err_arg else begin Window56.set_nominal_ppm ppm; ok end)

(* Read-only, for telemetry: emissions and limit ticks in the current window. *)
let emits () = guard (fun () -> Window56.emits ())
let limits () = guard (fun () -> Window56.limits ())

(* The same narrow opt-out cycle_c.ml and shadow_c.ml take, for the same reason:
   this process is single-domain by requirement (D70 c.2, c.9). *)
let[@alert "-unsafe_multidomain"] () =
  Callback.register "sentinel_win_push" push;
  Callback.register "sentinel_win_admits" (fun () -> admits ());
  Callback.register "sentinel_win_reset" (fun () -> reset ());
  Callback.register "sentinel_win_configure" configure;
  Callback.register "sentinel_win_emits" (fun () -> emits ());
  Callback.register "sentinel_win_limits" (fun () -> limits ())
