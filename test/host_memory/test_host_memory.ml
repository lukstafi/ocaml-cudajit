(* Tests Host_memory.register (cuMemHostRegister_v2) and Host_memory.unregister
   (cuMemHostUnregister) on deliberately non-page-aligned bigarray slices, including transfers,
   double-registration refusal, flags, and bounds checks. *)

let failures = ref 0

let check label ok =
  if not ok then incr failures;
  Printf.printf "%s: %b\n" label ok

let check_sub label ok =
  if not ok then (
    incr failures;
    Printf.eprintf "FAIL: %s\n" label);
  ok

let is_cuda_error ~message ~status_name f =
  match f () with
  | () -> false
  | exception Cuda.Cuda_error { message = got_msg; status } ->
      got_msg = message && Cuda.sexp_of_result status = Sexplib0.Sexp.Atom status_name

let is_invalid_arg f = match f () with () -> false | exception Invalid_argument _ -> true
let expected_value i = Float.of_int ((i * 7) + 3)

let genarray_address arr =
  Ctypes.raw_address_of_ptr (Ctypes.to_voidp (Ctypes.bigarray_start Ctypes.genarray arr))

let pick_unaligned_offset backing =
  let base_addr = genarray_address (Bigarray.genarray_of_array1 backing) in
  let page_size = 4096n in
  let offset1_addr = Nativeint.add base_addr 4n in
  if Nativeint.rem offset1_addr page_size <> 0n then 1 else 2

let () =
  let module Cu = Cuda in
  let module Host = Cu.Host_memory in
  Cu.init ();
  let device = Cu.Device.get ~ordinal:0 in
  let context = Cu.Context.create [] device in
  let props = Cu.Device.get_attributes device in

  (* Allocate a multi-page Float32 buffer and take a slice whose start address and byte length (2052
     bytes, 513 floats) are both guaranteed not 4096-byte page-aligned. *)
  let total_elems = 2048 in
  let slice_len = 513 in
  let backing = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout total_elems in
  Bigarray.Array1.fill backing (-999.0);
  let slice_offset = pick_unaligned_offset backing in
  let slice = Bigarray.Array1.sub backing slice_offset slice_len in
  let slice_gen = Bigarray.genarray_of_array1 slice in
  let slice_addr = genarray_address slice_gen in
  let slice_bytes = Bigarray.Genarray.size_in_bytes slice_gen in
  check "slice start is non-page-aligned" (Nativeint.rem slice_addr 4096n <> 0n);
  check "slice byte length is non-page-aligned" (slice_bytes mod 4096 <> 0);

  (* Verify sexp serialization of all flags and OCaml-side bounds checks, including one-past-end
     offsets so adjacent slices cannot be unregistered through a neighbor's end. *)
  let empty_gen =
    Bigarray.genarray_of_array1 (Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout 0)
  in
  let sexps_ok =
    List.map Host.sexp_of_register_flag [ PORTABLE; DEVICEMAP; IOMEMORY; READ_ONLY ]
    = [
        Sexplib0.Sexp.Atom "PORTABLE";
        Sexplib0.Sexp.Atom "DEVICEMAP";
        Sexplib0.Sexp.Atom "IOMEMORY";
        Sexplib0.Sexp.Atom "READ_ONLY";
      ]
  in
  check "sexp_of_register_flag covers all variants" sexps_ok;
  let bounds_ok =
    is_invalid_arg (fun () -> Host.register ~host_offset:(-1) slice_gen)
    && is_invalid_arg (fun () -> Host.register ~host_offset:slice_len slice_gen)
    && is_invalid_arg (fun () -> Host.register ~host_offset:(slice_len + 1) slice_gen)
    && is_invalid_arg (fun () -> Host.register ~length:(-1) slice_gen)
    && is_invalid_arg (fun () -> Host.register ~host_offset:10 ~length:slice_len slice_gen)
    && is_invalid_arg (fun () -> Host.unregister ~host_offset:(-1) slice_gen)
    && is_invalid_arg (fun () -> Host.unregister ~host_offset:slice_len slice_gen)
    && is_invalid_arg (fun () -> Host.unregister ~host_offset:(slice_len + 1) slice_gen)
    && is_invalid_arg (fun () -> Host.unregister empty_gen)
    && is_invalid_arg (fun () -> Host.register_unsafe ~ptr:Ctypes.null ~size_in_bytes:(-1) ())
  in
  check "out-of-bounds offsets and negative lengths rejected" bounds_ok;

  (* Keep golden stdout identical across devices: run the full registration suite when
     host_register_supported is true, or assert CUDA_ERROR_NOT_SUPPORTED when false. *)
  let host_reg_ok =
    if not props.host_register_supported then
      check_sub "unsupported device rejects Host_memory.register with CUDA_ERROR_NOT_SUPPORTED"
        (is_cuda_error ~message:"cu_mem_host_register" ~status_name:"CUDA_ERROR_NOT_SUPPORTED"
           (fun () -> Host.register slice_gen))
    else (
      for i = 0 to slice_len - 1 do
        slice.{i} <- expected_value i
      done;

      (* 1. Register the non-page-aligned slice, verify H2D/D2H round-trip, double-register refusal,
         and unregister. *)
      let dptr = Cu.Deviceptr.mem_alloc ~size_in_bytes:slice_bytes in
      Fun.protect
        ~finally:(fun () -> Cu.Deviceptr.mem_free dptr)
        (fun () ->
          Host.register slice_gen;
          let double_register_refused =
            Fun.protect
              ~finally:(fun () -> Host.unregister slice_gen)
              (fun () ->
                let refused =
                  is_cuda_error ~message:"cu_mem_host_register"
                    ~status_name:"CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED" (fun () ->
                      Host.register slice_gen)
                in
                Cu.Deviceptr.memcpy_H_to_D ~dst:dptr ~src:slice_gen ();
                for i = 0 to slice_len - 1 do
                  slice.{i} <- 0.0
                done;
                Cu.Deviceptr.memcpy_D_to_H ~dst:slice_gen ~src:dptr ();
                refused)
          in
          let sync_values_ok = ref true in
          for i = 0 to slice_len - 1 do
            if slice.{i} <> expected_value i then sync_values_ok := false
          done;
          let sentinels_ok = ref true in
          for i = 0 to slice_offset - 1 do
            if backing.{i} <> -999.0 then sentinels_ok := false
          done;
          for i = slice_offset + slice_len to total_elems - 1 do
            if backing.{i} <> -999.0 then sentinels_ok := false
          done;
          let c1 =
            check_sub "sync H2D/D2H round-trip through registered unaligned slice" !sync_values_ok
          in
          let c2 = check_sub "neighboring backing elements untouched" !sentinels_ok in
          let c3 =
            check_sub "double registration refused with CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED"
              double_register_refused
          in
          let c4 =
            check_sub
              "unregistering an unregistered slice fails with CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED"
              (is_cuda_error ~message:"cu_mem_host_unregister"
                 ~status_name:"CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED" (fun () ->
                   Host.unregister slice_gen))
          in

          (* 2. Re-register with PORTABLE and DEVICEMAP flags and exercise async stream copies. *)
          let stream = Cu.Stream.create () in
          Host.register ~flags:[ Host.PORTABLE; Host.DEVICEMAP ] slice_gen;
          Fun.protect
            ~finally:(fun () -> Host.unregister slice_gen)
            (fun () ->
              for i = 0 to slice_len - 1 do
                slice.{i} <- Float.of_int ((i * 11) + 5)
              done;
              Cu.Stream.memcpy_H_to_D ~dst:dptr ~src:slice_gen stream;
              Cu.Stream.synchronize stream;
              for i = 0 to slice_len - 1 do
                slice.{i} <- 0.0
              done;
              Cu.Stream.memcpy_D_to_H ~dst:slice_gen ~src:dptr stream;
              Cu.Stream.synchronize stream);
          let async_values_ok = ref true in
          for i = 0 to slice_len - 1 do
            if slice.{i} <> Float.of_int ((i * 11) + 5) then async_values_ok := false
          done;
          let c5 =
            check_sub "re-register with PORTABLE+DEVICEMAP and async stream round-trip"
              !async_values_ok
          in

          (* 3. Exercise ?host_offset and ?length directly on the backing Genarray. *)
          let backing_gen = Bigarray.genarray_of_array1 backing in
          Host.register ~flags:[ Host.PORTABLE ] ~host_offset:slice_offset ~length:slice_len
            backing_gen;
          let offset_double_refused =
            Fun.protect
              ~finally:(fun () -> Host.unregister ~host_offset:slice_offset backing_gen)
              (fun () ->
                let refused =
                  is_cuda_error ~message:"cu_mem_host_register"
                    ~status_name:"CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED" (fun () ->
                      Host.register ~host_offset:slice_offset ~length:slice_len backing_gen)
                in
                for i = 0 to slice_len - 1 do
                  backing.{slice_offset + i} <- expected_value i
                done;
                Cu.Deviceptr.memcpy_H_to_D ~host_offset:slice_offset ~length:slice_len ~dst:dptr
                  ~src:backing_gen ();
                for i = 0 to slice_len - 1 do
                  backing.{slice_offset + i} <- 0.0
                done;
                Cu.Deviceptr.memcpy_D_to_H ~host_offset:slice_offset ~length:slice_len
                  ~dst:backing_gen ~src:dptr ();
                refused)
          in
          let offset_values_ok = ref true in
          for i = 0 to slice_len - 1 do
            if backing.{slice_offset + i} <> expected_value i then offset_values_ok := false
          done;
          let c6 =
            check_sub "host_offset/length register, copy, and double-register refusal"
              (!offset_values_ok && offset_double_refused)
          in

          (* 4. Exercise READ_ONLY flag according to device capability. *)
          let ro_buf = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout slice_len in
          for i = 0 to slice_len - 1 do
            ro_buf.{i} <- expected_value i
          done;
          let ro_gen = Bigarray.genarray_of_array1 ro_buf in
          let ro_ok =
            if props.read_only_host_register_supported then (
              Host.register ~flags:[ Host.READ_ONLY ] ro_gen;
              Fun.protect
                ~finally:(fun () -> Host.unregister ro_gen)
                (fun () ->
                  Cu.Deviceptr.memcpy_H_to_D ~dst:dptr ~src:ro_gen ();
                  let readback =
                    Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout slice_len
                  in
                  Cu.Deviceptr.memcpy_D_to_H
                    ~dst:(Bigarray.genarray_of_array1 readback)
                    ~src:dptr ();
                  let ok = ref true in
                  for i = 0 to slice_len - 1 do
                    if readback.{i} <> expected_value i then ok := false
                  done;
                  !ok))
            else
              is_cuda_error ~message:"cu_mem_host_register" ~status_name:"CUDA_ERROR_NOT_SUPPORTED"
                (fun () -> Host.register ~flags:[ Host.READ_ONLY ] ro_gen)
          in
          let c7 = check_sub "READ_ONLY flag matches device capability" ro_ok in

          (* 5. Exercise register_unsafe / unregister_unsafe on the unaligned slice pointer. *)
          let slice_voidp = Ctypes.to_voidp (Ctypes.bigarray_start Ctypes.genarray slice_gen) in
          Host.register_unsafe ~flags:[ Host.PORTABLE ] ~ptr:slice_voidp ~size_in_bytes:slice_bytes
            ();
          let unsafe_double_refused =
            Fun.protect
              ~finally:(fun () -> Host.unregister_unsafe ~ptr:slice_voidp)
              (fun () ->
                is_cuda_error ~message:"cu_mem_host_register"
                  ~status_name:"CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED" (fun () ->
                    Host.register_unsafe ~ptr:slice_voidp ~size_in_bytes:slice_bytes ()))
          in
          ignore (Sys.opaque_identity slice_gen);
          let c8 =
            check_sub
              "register_unsafe/unregister_unsafe on unaligned slice and double-register refusal"
              unsafe_double_refused
          in
          c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8))
  in
  check "host registration matches device capability" host_reg_ok;

  ignore (Sys.opaque_identity context);
  if !failures > 0 then exit 1
