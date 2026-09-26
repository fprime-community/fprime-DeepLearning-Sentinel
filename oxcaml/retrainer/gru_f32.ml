(* Section 54: the GRU cell in FLOAT32, under [@zero_alloc strict].
 *
 * (!) 54.2 + 54.2a PERMITTED SET, and nothing outside it is used:
 *   %addfloat32  %mulfloat32  %divfloat32  %floatoffloat32  %float32offloat
 *   expf  tanhf   (C externals, [@@unboxed] [@@noalloc])
 * Subtraction is add + an exact sign flip and comparison is an exact widening (54.2a), so
 * neither %subfloat32 nor %negfloat32 is used even though both exist.
 *
 * (!) STORAGE IS Bigarray float32. 54.3 measured a boxed `float32 array` failing strict --
 * float32 is a boxed type, so an OCaml array of them allocates on every write. A Bigarray
 * holds the F32 bits and its OCaml interface is `float`, so every float32 here is a LOCAL
 * temporary, which flambda2 keeps unboxed.
 *
 * (!) THE ACCUMULATION ORDER IS flight/src/Gru.cpp:15-25's, NOT 48's. `affine` accumulates
 * from 0.0F and adds the bias LAST; 48's cell starts the accumulator AT the bias. The two
 * differ in the last bits, which is invisible at F64-against-F32 and decisive for FT3. *)


(* D82: every element access below is bounds-checked. `acc.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and every `[@zero_alloc strict]` site still
   holds. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc

external to_f : float32 -> float = "%floatoffloat32"
external of_f : float -> float32 = "%float32offloat"
external add : float32 -> float32 -> float32 = "%addfloat32"
external mul : float32 -> float32 -> float32 = "%mulfloat32"
external div : float32 -> float32 -> float32 = "%divfloat32"
external expf : float32 -> float32 = "caml_expf_float32" "expf" [@@unboxed] [@@noalloc]
external tanhf : float32 -> float32 = "caml_tanhf_float32" "tanhf" [@@unboxed] [@@noalloc]

let hs = 80
let ins = 16
let gw = 240

type f32 = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
let mk n : f32 = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n
let[@inline] get (a : f32) i = of_f (F32.get a i)
let[@inline] set (a : f32) i (v : float32) = F32.set a i (to_f v)

let w_ih = mk (gw * ins)
let w_hh = mk (gw * hs)
let b_ih = mk gw
let b_hh = mk gw
let x = mk ins
let h = mk hs
let proj = mk gw
let rec_ = mk gw
let r = mk hs
let z = mk hs
let n = mk hs
let h_new = mk hs

let g_w_ih = mk (gw * ins)
let g_w_hh = mk (gw * hs)
let g_b_ih = mk gw
let g_b_hh = mk gw
let g_x = mk ins
let g_h = mk hs
let g_h_new = mk hs

(* 54.2a: -b is an exact sign flip, so a + (b * -1) IS a float32 subtraction. *)
let[@inline] sub a b = add a (mul b (of_f (-1.0)))

(* flight/src/Gru.cpp:28-34, branch and all. The comparison is an exact widening. *)
let[@inline] sigmoid v =
  if to_f v >= 0.0 then div (of_f 1.0) (add (of_f 1.0) (expf (mul v (of_f (-1.0)))))
  else let e = expf v in div e (add (of_f 1.0) e)

(* flight/src/Gru.cpp:15-25: accumulate from ZERO, add the bias LAST. *)
let[@zero_alloc strict] forward () =
  for gi = 0 to gw - 1 do
    let acc = ref (of_f 0.0) in
    for j = 0 to ins - 1 do
      acc := add !acc (mul (get w_ih ((gi * ins) + j)) (get x j))
    done;
    set proj gi (add !acc (get b_ih gi))
  done;
  for gi = 0 to gw - 1 do
    let acc = ref (of_f 0.0) in
    for j = 0 to hs - 1 do
      acc := add !acc (mul (get w_hh ((gi * hs) + j)) (get h j))
    done;
    set rec_ gi (add !acc (get b_hh gi))
  done;
  for i = 0 to hs - 1 do
    let ri = sigmoid (add (get proj i) (get rec_ i)) in
    let zi = sigmoid (add (get proj (hs + i)) (get rec_ (hs + i))) in
    let rn = get rec_ ((2 * hs) + i) in
    let ni = tanhf (add (get proj ((2 * hs) + i)) (mul ri rn)) in
    let hi = get h i in
    set r i ri; set z i zi; set n i ni;
    set h_new i (add (mul (sub hi ni) zi) ni)
  done

let[@zero_alloc strict] zero_grads () =
  for i = 0 to (gw * ins) - 1 do set g_w_ih i (of_f 0.0) done;
  for i = 0 to (gw * hs) - 1 do set g_w_hh i (of_f 0.0) done;
  for i = 0 to gw - 1 do set g_b_ih i (of_f 0.0); set g_b_hh i (of_f 0.0) done;
  for i = 0 to ins - 1 do set g_x i (of_f 0.0) done;
  for i = 0 to hs - 1 do set g_h i (of_f 0.0) done

(* 48's backward, in float32. There is no flight reference for a backward pass, so the
   order follows 48's rather than flight/'s. *)
let[@zero_alloc strict] backward () =
  let one = of_f 1.0 in
  for i = 0 to hs - 1 do
    let gh = get g_h_new i in
    let hi = get h i and ri = get r i and zi = get z i and ni = get n i in
    let rn = get rec_ ((2 * hs) + i) in
    let d_z = mul gh (sub hi ni) in
    let d_n = mul gh (sub one zi) in
    let d_h_direct = mul gh zi in
    let d_pre_n = mul d_n (sub one (mul ni ni)) in
    let d_r = mul d_pre_n rn in
    let d_rec_n = mul d_pre_n ri in
    let d_pre_z = mul (mul d_z zi) (sub one zi) in
    let d_pre_r = mul (mul d_r ri) (sub one ri) in
    set g_b_ih i (add (get g_b_ih i) d_pre_r);
    set g_b_hh i (add (get g_b_hh i) d_pre_r);
    set g_b_ih (hs + i) (add (get g_b_ih (hs + i)) d_pre_z);
    set g_b_hh (hs + i) (add (get g_b_hh (hs + i)) d_pre_z);
    set g_b_ih ((2 * hs) + i) (add (get g_b_ih ((2 * hs) + i)) d_pre_n);
    set g_b_hh ((2 * hs) + i) (add (get g_b_hh ((2 * hs) + i)) d_rec_n);
    for j = 0 to ins - 1 do
      let xj = get x j in
      let ir = (i * ins) + j and iz = ((hs + i) * ins) + j in
      let inn = (((2 * hs) + i) * ins) + j in
      set g_w_ih ir (add (get g_w_ih ir) (mul d_pre_r xj));
      set g_w_ih iz (add (get g_w_ih iz) (mul d_pre_z xj));
      set g_w_ih inn (add (get g_w_ih inn) (mul d_pre_n xj));
      set g_x j (add (get g_x j)
                   (add (mul d_pre_r (get w_ih ir))
                      (add (mul d_pre_z (get w_ih iz)) (mul d_pre_n (get w_ih inn)))))
    done;
    for j = 0 to hs - 1 do
      let hj = get h j in
      let ir = (i * hs) + j and iz = ((hs + i) * hs) + j in
      let inn = (((2 * hs) + i) * hs) + j in
      set g_w_hh ir (add (get g_w_hh ir) (mul d_pre_r hj));
      set g_w_hh iz (add (get g_w_hh iz) (mul d_pre_z hj));
      set g_w_hh inn (add (get g_w_hh inn) (mul d_rec_n hj));
      set g_h j (add (get g_h j)
                   (add (mul d_pre_r (get w_hh ir))
                      (add (mul d_pre_z (get w_hh iz)) (mul d_rec_n (get w_hh inn)))))
    done;
    set g_h i (add (get g_h i) d_h_direct)
  done
