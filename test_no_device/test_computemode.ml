(* Regression test for UNCATEGORIZED compute-mode handling.
   Uses Cuda.Device.computemode_of_int to route raw integers through the full
   conversion chain (cu_computemode_of_int -> computemode_of_cu), then checks
   sexp_of_computemode output.  No GPU device or Cu.init() required. *)

let check label expected sexp =
  let got = Sexplib0.Sexp.to_string sexp in
  if got = expected then Printf.printf "%s: PASS\n" label
  else Printf.printf "%s: FAIL (expected %S, got %S)\n" label expected got

let () =
  (* Known CUDA compute-mode constants routed through the full conversion path. *)
  check "raw 0 -> DEFAULT"
    "DEFAULT"
    (Cuda.Device.sexp_of_computemode (Cuda.Device.computemode_of_int 0));
  check "raw 2 -> PROHIBITED"
    "PROHIBITED"
    (Cuda.Device.sexp_of_computemode (Cuda.Device.computemode_of_int 2));
  check "raw 3 -> EXCLUSIVE_PROCESS"
    "EXCLUSIVE_PROCESS"
    (Cuda.Device.sexp_of_computemode (Cuda.Device.computemode_of_int 3));
  (* Value 1 = legacy CU_COMPUTEMODE_EXCLUSIVE removed in CUDA 8.0.
     computemode_of_cu previously raised invalid_arg here; now returns UNCATEGORIZED.
     If src/cuda.ml:236 is reverted to invalid_arg, computemode_of_int 1 raises
     and this test fails at that line rather than at the sexp comparison. *)
  check "raw 1 -> UNCATEGORIZED 1"
    "(UNCATEGORIZED 1)"
    (Cuda.Device.sexp_of_computemode (Cuda.Device.computemode_of_int 1));
  check "raw 42 -> UNCATEGORIZED 42"
    "(UNCATEGORIZED 42)"
    (Cuda.Device.sexp_of_computemode (Cuda.Device.computemode_of_int 42))
