(* Section 50: backpropagation through time over a FIXED TAPE, under [@zero_alloc strict].
 *
 * Gru_cell is NOT edited. This module drives its three annotated functions over a sequence,
 * carrying every intermediate on a tape claimed once at module level -- CPP-1's shape.
 *
 * (!) 50.2's FAILURE MODE 1, REGISTERED IN ADVANCE: the tape is a SINGLE FLAT float array
 * with computed indices, not a float array array, because the latter is an array of pointers
 * and is where `strict` was predicted to object.
 *
 * (!) 50.2's FAILURE MODE 2: no helper here returns a float. 48's G1 established that a float
 * returned across a function boundary boxes, so the sequence loss is written into a
 * one-element array rather than returned -- the same accommodation 47.13.4 recorded for
 * `checksum`.
 *
 * (!) AND THE TAPE IS FIVE ARRAYS PER STEP, NOT THE FOUR 50.1 DECLARED. Gru_cell.backward
 * reads `rec_` at the third gate block as well as r, z, n and the incoming h, so rec_n has to
 * be on the tape too. 50.1 under-specified this and the OBSERVED section says so. *)


(* D82: every element access below is bounds-checked. `acc.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and every `[@zero_alloc strict]` site still
   holds. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc

let hs = Gru_cell.h_size
let ins = Gru_cell.in_size

let t_max = 250                  (* the flown window, src/sentinel_models/lstm.py:90 *)
let tape_w = 5                   (* r, z, n, rec_n, h_in *)

(* -- everything claimed once, at init ------------------------------------------------- *)
let tape = Array.make (t_max * tape_w * hs) 0.0
let x_seq = Array.make (t_max * ins) 0.0
let g_x_seq = Array.make (t_max * ins) 0.0
let coeff_seq = Array.make (t_max * hs) 0.0
let h0 = Array.make hs 0.0
let g_h0 = Array.make hs 0.0
let carry = Array.make hs 0.0
let loss_out = Array.make 1 0.0  (* a float RETURN would box; a slot does not *)

let[@inline] tix t k i = (((t * tape_w) + k) * hs) + i

(* zero only the per-step buffers. Gru_cell.zero_grads would also clear the weight
   gradients, which is exactly what must NOT happen between timesteps. *)
let[@zero_alloc strict] zero_step_grads () =
  for j = 0 to ins - 1 do Array.unsafe_set Gru_cell.g_x j 0.0 done;
  for i = 0 to hs - 1 do Array.unsafe_set Gru_cell.g_h i 0.0 done

(* -- forward over the sequence, writing the tape --------------------------------------- *)
let[@zero_alloc strict] forward_seq t_steps =
  Array.unsafe_set loss_out 0 0.0;
  for t = 0 to t_steps - 1 do
    for j = 0 to ins - 1 do
      Array.unsafe_set Gru_cell.x j (Array.unsafe_get x_seq ((t * ins) + j))
    done;
    if t = 0 then
      for i = 0 to hs - 1 do
        Array.unsafe_set Gru_cell.h i (Array.unsafe_get h0 i)
      done
    else
      for i = 0 to hs - 1 do
        Array.unsafe_set Gru_cell.h i (Array.unsafe_get Gru_cell.h_new i)
      done;
    (* h_in is what backward will need at this step *)
    for i = 0 to hs - 1 do
      Array.unsafe_set tape (tix t 4 i) (Array.unsafe_get Gru_cell.h i)
    done;
    Gru_cell.forward ();
    for i = 0 to hs - 1 do
      Array.unsafe_set tape (tix t 0 i) (Array.unsafe_get Gru_cell.r i);
      Array.unsafe_set tape (tix t 1 i) (Array.unsafe_get Gru_cell.z i);
      Array.unsafe_set tape (tix t 2 i) (Array.unsafe_get Gru_cell.n i);
      Array.unsafe_set tape (tix t 3 i)
        (Array.unsafe_get Gru_cell.rec_ ((2 * hs) + i))
    done;
    (* the loss enters at EVERY step -- 50.1 *)
    for i = 0 to hs - 1 do
      Array.unsafe_set loss_out 0
        ((Array.unsafe_get loss_out 0)
         +. ((Array.unsafe_get coeff_seq ((t * hs) + i))
             *. (Array.unsafe_get Gru_cell.h_new i)))
    done
  done

(* -- the backward sweep through time ---------------------------------------------------- *)
let[@zero_alloc strict] backward_seq t_steps =
  Gru_cell.zero_grads ();
  for i = 0 to hs - 1 do Array.unsafe_set carry i 0.0 done;
  for t = t_steps - 1 downto 0 do
    for j = 0 to ins - 1 do
      Array.unsafe_set Gru_cell.x j (Array.unsafe_get x_seq ((t * ins) + j))
    done;
    for i = 0 to hs - 1 do
      Array.unsafe_set Gru_cell.r i (Array.unsafe_get tape (tix t 0 i));
      Array.unsafe_set Gru_cell.z i (Array.unsafe_get tape (tix t 1 i));
      Array.unsafe_set Gru_cell.n i (Array.unsafe_get tape (tix t 2 i));
      Array.unsafe_set Gru_cell.rec_ ((2 * hs) + i) (Array.unsafe_get tape (tix t 3 i));
      Array.unsafe_set Gru_cell.h i (Array.unsafe_get tape (tix t 4 i));
      (* dL/dh_t = the loss's own term, plus everything the future sent back *)
      Array.unsafe_set Gru_cell.g_h_new i
        ((Array.unsafe_get coeff_seq ((t * hs) + i)) +. (Array.unsafe_get carry i))
    done;
    zero_step_grads ();
    Gru_cell.backward ();
    for j = 0 to ins - 1 do
      Array.unsafe_set g_x_seq ((t * ins) + j) (Array.unsafe_get Gru_cell.g_x j)
    done;
    for i = 0 to hs - 1 do
      Array.unsafe_set carry i (Array.unsafe_get Gru_cell.g_h i)
    done
  done;
  for i = 0 to hs - 1 do Array.unsafe_set g_h0 i (Array.unsafe_get carry i) done
