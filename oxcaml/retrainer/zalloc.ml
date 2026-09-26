(* E3: [@zero_alloc strict] on real arithmetic. docs/MODELS.md 47.9, X7 to X9.
 *
 * (!) WRITTEN IN NATURAL OCaml FIRST, ON PURPOSE. X7 asks whether `strict` holds on the
 * arithmetic a training step actually has, without pervasive `assume` or inlining
 * gymnastics. Writing it pre-contorted to please the checker would answer a different
 * question. There is NO `assume` anywhere in this file, and 47.9 X9 counts any that
 * appear.
 *
 * A matrix multiply and a gradient step, both on preallocated flat float arrays. OCaml's
 * `float array` is unboxed, so the storage itself allocates once at module init -- which
 * is CPP-1's shape, allocation at initialisation and none after. *)


(* D82: every element access below is bounds-checked. `acc.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and every `[@zero_alloc strict]` site still
   holds. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc

let n = 16

(* Preallocated once, at module initialisation. Nothing below allocates these. *)
let a = Array.make (n * n) 0.0
let b = Array.make (n * n) 0.0
let c = Array.make (n * n) 0.0

let w = Array.make (n * n) 0.0
let x = Array.make n 0.0
let y = Array.make n 0.0
let pred = Array.make n 0.0
let grad = Array.make n 0.0

(* -- the matrix multiply ------------------------------------------------------------ *)
let[@zero_alloc strict] matmul () =
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      let s = ref 0.0 in
      for k = 0 to n - 1 do
        s := !s +. (Array.unsafe_get a ((i * n) + k))
                   *. (Array.unsafe_get b ((k * n) + j))
      done;
      Array.unsafe_set c ((i * n) + j) !s
    done
  done

(* -- the forward pass: pred = W x ---------------------------------------------------- *)
let[@zero_alloc strict] forward () =
  for i = 0 to n - 1 do
    let s = ref 0.0 in
    for j = 0 to n - 1 do
      s := !s +. (Array.unsafe_get w ((i * n) + j)) *. (Array.unsafe_get x j)
    done;
    Array.unsafe_set pred i !s
  done

(* -- one gradient step: squared error, SGD ------------------------------------------- *)
let[@zero_alloc strict] grad_step lr =
  forward ();
  for i = 0 to n - 1 do
    let d = (Array.unsafe_get pred i) -. (Array.unsafe_get y i) in
    Array.unsafe_set grad i (2.0 *. d)
  done;
  for i = 0 to n - 1 do
    let gi = Array.unsafe_get grad i in
    for j = 0 to n - 1 do
      let idx = (i * n) + j in
      let upd = (Array.unsafe_get w idx) -. (lr *. gi *. (Array.unsafe_get x j)) in
      Array.unsafe_set w idx upd
    done
  done

(* A scalar readback, so the C side can check the arithmetic did something.
 *
 * (!) THIS ONE FUNCTION NEEDED ITS SIGNATURE CHANGED, AND IT IS THE ONLY ACCOMMODATION IN
 * THIS FILE. Written the obvious way -- `let[@zero_alloc strict] checksum () = ... ; !s`,
 * returning the float -- the checker refused it with, verbatim:
 *
 *     Error: allocation of 16 bytes for float
 *
 * and it was right: a float RETURNED across a non-inlined boundary is boxed. The
 * arithmetic was never the problem. Note what did NOT allocate: the `ref 0.0`
 * accumulators in every function here, including this one, which flambda2 unboxes.
 *
 * The fix writes the result into a preallocated slot instead of returning it. That is a
 * change of signature, not a contortion of the body, and it is the SAME shape
 * docs/MODELS.md 47.6 already requires at the C boundary -- scalars out through
 * caller-owned memory. It is counted as an accommodation at 47.13.4 regardless, because
 * X7's band is about what `strict` demands and this is one of the things it demanded.
 *
 * OxCaml's unboxed `float#` is the language-native alternative and is deliberately NOT
 * used here: E3 asks what plain OCaml costs under `strict`, and reaching for an extension
 * would answer a different question. Registered at 47.14. *)
let out = Array.make 1 0.0

let[@zero_alloc strict] checksum () =
  let s = ref 0.0 in
  for i = 0 to (n * n) - 1 do
    s := !s +. (Array.unsafe_get c i) +. (Array.unsafe_get w i)
  done;
  Array.unsafe_set out 0 !s
