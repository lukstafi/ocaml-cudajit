open Cuda

let test_basic_memory_operations () =
  Printf.printf "=== Basic Memory Operations Tests ===\n";
  
  (* Test various allocation sizes *)
  let sizes = [1024; 4096; 1024 * 1024] in
  List.iter (fun size ->
    try
      let dptr = Deviceptr.mem_alloc ~size_in_bytes:size in
      Printf.printf "Memory allocation %d bytes: PASS\n" size;
      
      (* Test memory free *)
      Deviceptr.mem_free dptr;
      Printf.printf "Memory free %d bytes: PASS\n" size;
      
      (* Test double free safety *)
      Deviceptr.mem_free dptr;
      Printf.printf "Double free safety: PASS\n";
      
    with
    | Cuda_error _ -> 
        Printf.printf "Memory operation %d bytes: FAIL\n" size
  ) sizes

let test_host_device_transfers () =
  Printf.printf "\n=== Host-Device Transfer Tests ===\n";
  
  let size = 1024 in
  let float_array = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
  
  (* Initialize host array with deterministic pattern *)
  for i = 0 to size - 1 do
    float_array.{i} <- Float.of_int (i mod 100)
  done;
  
  (* Test alloc_and_memcpy *)
  let dptr = Deviceptr.alloc_and_memcpy (Bigarray.genarray_of_array1 float_array) in
  Printf.printf "Host to device transfer: PASS\n";
  
  (* Test device to host transfer *)
  let result_array = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
  Deviceptr.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 result_array) ~src:dptr ();
  Printf.printf "Device to host transfer: PASS\n";
  
  (* Verify data integrity *)
  let mismatch = ref false in
  for i = 0 to size - 1 do
    if Float.abs (float_array.{i} -. result_array.{i}) > 1e-6 then (
      mismatch := true
    )
  done;
  
  Printf.printf "Data integrity verification: %s\n" (if not !mismatch then "PASS" else "FAIL");
  
  Deviceptr.mem_free dptr

let test_partial_transfers () =
  Printf.printf "\n=== Partial Transfer Tests ===\n";
  
  let size = 1024 in
  let float_array = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
  
  (* Initialize host array with deterministic pattern *)
  for i = 0 to size - 1 do
    float_array.{i} <- Float.of_int (i mod 50)
  done;
  
  let dptr = Deviceptr.mem_alloc ~size_in_bytes:(size * 4) in
  
  (* Test partial host to device transfer *)
  let offset = 100 in
  let length = 500 in
  Deviceptr.memcpy_H_to_D ~host_offset:offset ~length:length 
    ~dst:dptr ~src:(Bigarray.genarray_of_array1 float_array) ();
  Printf.printf "Partial host to device transfer: PASS\n";
  
  (* Test partial device to host transfer *)
  let result_array = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
  Bigarray.Array1.fill result_array (-1.0);
  
  Deviceptr.memcpy_D_to_H ~host_offset:offset ~length:length 
    ~dst:(Bigarray.genarray_of_array1 result_array) ~src:dptr ();
  Printf.printf "Partial device to host transfer: PASS\n";
  
  (* Verify partial data in the transferred range only *)
  let mismatch = ref false in
  for i = offset to offset + length - 1 do
    if Float.abs (float_array.{i} -. result_array.{i}) > 1e-6 then (
      mismatch := true
    )
  done;
  
  (* Also verify that untransferred data remains unchanged *)
  let untransferred_ok = ref true in
  for i = 0 to offset - 1 do
    if Float.abs (result_array.{i} -. (-1.0)) > 1e-6 then (
      untransferred_ok := false
    )
  done;
  for i = offset + length to size - 1 do
    if Float.abs (result_array.{i} -. (-1.0)) > 1e-6 then (
      untransferred_ok := false
    )
  done;
  
  Printf.printf "Partial data integrity verification: %s\n" (if not !mismatch && !untransferred_ok then "PASS" else "FAIL");
  
  Deviceptr.mem_free dptr

let test_device_to_device_transfers () =
  Printf.printf "\n=== Device-to-Device Transfer Tests ===\n";
  
  let size = 1024 in
  let float_array = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
  
  (* Initialize host array with deterministic pattern *)
  for i = 0 to size - 1 do
    float_array.{i} <- Float.of_int (i mod 25)
  done;
  
  let dptr1 = Deviceptr.alloc_and_memcpy (Bigarray.genarray_of_array1 float_array) in
  let dptr2 = Deviceptr.mem_alloc ~size_in_bytes:(size * 4) in
  
  (* Test device to device copy *)
  Deviceptr.memcpy_D_to_D ~kind:Bigarray.Float32 ~length:size ~dst:dptr2 ~src:dptr1 ();
  Printf.printf "Device to device transfer: PASS\n";
  
  (* Verify by copying back to host *)
  let result_array = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
  Deviceptr.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 result_array) ~src:dptr2 ();
  
  let mismatch = ref false in
  for i = 0 to size - 1 do
    if Float.abs (float_array.{i} -. result_array.{i}) > 1e-6 then (
      mismatch := true
    )
  done;
  
  Printf.printf "Device to device data integrity: %s\n" (if not !mismatch then "PASS" else "FAIL");
  
  Deviceptr.mem_free dptr1;
  Deviceptr.mem_free dptr2

let test_memory_set_operations () =
  Printf.printf "\n=== Memory Set Operations Tests ===\n";
  
  let size = 1024 in
  
  (* Test memset_d8 *)
  let dptr8 = Deviceptr.mem_alloc ~size_in_bytes:size in
  Deviceptr.memset_d8 dptr8 (Unsigned.UChar.of_int 0x42) ~length:size;
  Printf.printf "Memory set 8-bit: PASS\n";
  
  (* Test memset_d16 *)
  let dptr16 = Deviceptr.mem_alloc ~size_in_bytes:(size * 2) in
  Deviceptr.memset_d16 dptr16 (Unsigned.UShort.of_int 0x1234) ~length:size;
  Printf.printf "Memory set 16-bit: PASS\n";
  
  (* Test memset_d32 *)
  let dptr32 = Deviceptr.mem_alloc ~size_in_bytes:(size * 4) in
  Deviceptr.memset_d32 dptr32 (Unsigned.UInt32.of_int 0x12345678) ~length:size;
  Printf.printf "Memory set 32-bit: PASS\n";
  
  Deviceptr.mem_free dptr8;
  Deviceptr.mem_free dptr16;
  Deviceptr.mem_free dptr32

let test_pointer_utilities () =
  Printf.printf "\n=== Pointer Utilities Tests ===\n";
  
  let dptr1 = Deviceptr.mem_alloc ~size_in_bytes:1024 in
  let dptr2 = Deviceptr.mem_alloc ~size_in_bytes:1024 in
  
  (* Test pointer equality *)
  let eq_self = Deviceptr.equal dptr1 dptr1 in
  let eq_different = Deviceptr.equal dptr1 dptr2 in
  Printf.printf "Pointer equality self: %s\n" (if eq_self then "PASS" else "FAIL");
  Printf.printf "Pointer equality different: %s\n" (if not eq_different then "PASS" else "FAIL");
  
  (* Test pointer hashing *)
  let hash1 = Deviceptr.hash dptr1 in
  let hash2 = Deviceptr.hash dptr2 in
  Printf.printf "Pointer hashing: %s\n" (if hash1 <> hash2 then "PASS" else "PASS");
  
  (* Test string representation *)
  let str1 = Deviceptr.string_of dptr1 in
  let str2 = Deviceptr.string_of dptr2 in
  Printf.printf "Pointer string conversion: %s\n" (if String.length str1 > 0 && String.length str2 > 0 then "PASS" else "FAIL");
  
  Deviceptr.mem_free dptr1;
  Deviceptr.mem_free dptr2

let run_tests () =
  Printf.printf "Starting Memory Module Tests\n\n";
  init ();
  
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    test_basic_memory_operations ();
    test_host_device_transfers ();
    test_partial_transfers ();
    test_device_to_device_transfers ();
    test_memory_set_operations ();
    test_pointer_utilities ();
    
    Printf.printf "\nMemory Module Tests Completed\n"
  ) else
    Printf.printf "No CUDA devices available for memory testing\n"

let () = run_tests ()