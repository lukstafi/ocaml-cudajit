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

let test_device_offset_transfers () =
  Printf.printf "\n=== Device-Side Offset Transfer Tests ===\n";

  (* A larger allocation; the payload lands at a non-zero device byte offset within it. *)
  let total = 256 in
  let payload_len = 64 in
  let offset_elems = 64 in
  let offset_bytes = offset_elems * 4 (* Float32 *) in

  let payload = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout payload_len in
  for i = 0 to payload_len - 1 do
    payload.{i} <- Float.of_int ((i + 1) * 3)
  done;

  let dptr = Deviceptr.mem_alloc ~size_in_bytes:(total * 4) in
  (* Sentinel-fill the whole allocation so neighbouring bytes are known (0.0). *)
  Deviceptr.memset_d32 dptr (Unsigned.UInt32.of_int 0) ~length:total;

  (* Host -> device into a non-zero device offset. *)
  Deviceptr.memcpy_H_to_D ~length:payload_len ~dst_offset:offset_bytes ~dst:dptr
    ~src:(Bigarray.genarray_of_array1 payload) ();

  (* Read it back via the complementary device src_offset. *)
  let readback = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout payload_len in
  Bigarray.Array1.fill readback (-1.0);
  Deviceptr.memcpy_D_to_H ~length:payload_len ~src_offset:offset_bytes
    ~dst:(Bigarray.genarray_of_array1 readback) ~src:dptr ();
  let offset_roundtrip_ok = ref true in
  for i = 0 to payload_len - 1 do
    if Float.abs (payload.{i} -. readback.{i}) > 1e-6 then offset_roundtrip_ok := false
  done;
  Printf.printf "Host-to-device at non-zero device offset: %s\n"
    (if !offset_roundtrip_ok then "PASS" else "FAIL");

  (* Read back the whole allocation to confirm the bytes landed only at the sub-region. *)
  let whole = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout total in
  Bigarray.Array1.fill whole (-1.0);
  Deviceptr.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 whole) ~src:dptr ();
  let neighbours_ok = ref true in
  for i = 0 to total - 1 do
    let expected =
      if i >= offset_elems && i < offset_elems + payload_len then payload.{i - offset_elems}
      else 0.0
    in
    if Float.abs (whole.{i} -. expected) > 1e-6 then neighbours_ok := false
  done;
  Printf.printf "Neighbouring bytes untouched (H-to-D offset): %s\n"
    (if !neighbours_ok then "PASS" else "FAIL");

  (* Device -> device between two distinct non-zero offsets. *)
  let dptr2 = Deviceptr.mem_alloc ~size_in_bytes:(total * 4) in
  Deviceptr.memset_d32 dptr2 (Unsigned.UInt32.of_int 0) ~length:total;
  let dst2_offset_elems = 128 in
  Deviceptr.memcpy_D_to_D ~kind:Bigarray.Float32 ~length:payload_len
    ~src_offset:offset_bytes ~dst_offset:(dst2_offset_elems * 4) ~dst:dptr2 ~src:dptr ();
  let whole2 = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout total in
  Bigarray.Array1.fill whole2 (-1.0);
  Deviceptr.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 whole2) ~src:dptr2 ();
  let d2d_ok = ref true in
  for i = 0 to total - 1 do
    let expected =
      if i >= dst2_offset_elems && i < dst2_offset_elems + payload_len then
        payload.{i - dst2_offset_elems}
      else 0.0
    in
    if Float.abs (whole2.{i} -. expected) > 1e-6 then d2d_ok := false
  done;
  Printf.printf "Device-to-device between non-zero offsets: %s\n"
    (if !d2d_ok then "PASS" else "FAIL");

  Deviceptr.mem_free dptr;
  Deviceptr.mem_free dptr2

let test_region_memset () =
  Printf.printf "\n=== Region (Offset) Memset Tests ===\n";
  let n = 256 in
  let dptr = Deviceptr.mem_alloc ~size_in_bytes:n in
  (* Fill the whole allocation with 0xAA. *)
  Deviceptr.memset_d8 dptr (Unsigned.UChar.of_int 0xAA) ~length:n;
  (* Zero-fill bytes [64..127] using a non-zero byte offset. *)
  let off = 64 in
  Deviceptr.memset_d8 ~offset:off dptr (Unsigned.UChar.of_int 0x00) ~length:64;
  let host = Bigarray.Array1.create Bigarray.Char Bigarray.C_layout n in
  Deviceptr.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 host) ~src:dptr ();
  let ok = ref true in
  for i = 0 to n - 1 do
    let expected = if i >= off && i < off + 64 then '\x00' else '\xAA' in
    if Char.code host.{i} <> Char.code expected then ok := false
  done;
  Printf.printf "Non-zero-offset memset_d8 targets correct sub-region: %s\n"
    (if !ok then "PASS" else "FAIL");
  Deviceptr.mem_free dptr

let fill_kernel_src = {|
extern "C" __global__ void fill_floats(float *buf, float val, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) buf[tid] = val;
}
|}

let test_region_kernel_launch () =
  Printf.printf "\n=== Region (Tensor_at) Kernel Launch Tests ===\n";
  let prog =
    Nvrtc.compile_to_ptx ~cu_src:fill_kernel_src ~name:"fill_floats"
      ~options:[] ~with_debug:false
  in
  let module_ = Module.load_data_ex prog [] in
  let func = Module.get_function module_ ~name:"fill_floats" in
  let slab_elems = 512 in
  let region_start = 256 in (* sub-region starts at element index 256 *)
  let region_len = 64 in
  let offset_bytes = region_start * 4 in (* Float32 = 4 bytes per element *)
  let slab = Deviceptr.mem_alloc ~size_in_bytes:(slab_elems * 4) in
  (* Zero-fill the whole slab so neighbours are known. *)
  Deviceptr.memset_d32 slab (Unsigned.UInt32.of_int 0) ~length:slab_elems;
  (* Launch kernel targeting [region_start..region_start+region_len) via Tensor_at. *)
  let region = Deviceptr.offset slab ~bytes:offset_bytes in
  Stream.launch_kernel func ~grid_dim_x:1 ~block_dim_x:128 ~shared_mem_bytes:0
    Stream.no_stream
    [ Stream.Tensor_at region; Stream.Single 42.0; Stream.Int region_len ];
  Context.synchronize ();
  let host = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout slab_elems in
  Bigarray.Array1.fill host (-1.0);
  Deviceptr.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 host) ~src:slab ();
  let ok = ref true in
  for i = 0 to slab_elems - 1 do
    let expected =
      if i >= region_start && i < region_start + region_len then 42.0 else 0.0
    in
    if Float.abs (host.{i} -. expected) > 1e-6 then ok := false
  done;
  Printf.printf "Kernel launch via Tensor_at writes correct sub-region: %s\n"
    (if !ok then "PASS" else "FAIL");
  Deviceptr.mem_free slab

(* Complements [test_region_memset] (sync, d8): exercises the *asynchronous*
   [Stream.memset_d32 ?offset] path, which AC 4 covers but the sync test does not. *)
let test_region_memset_async () =
  Printf.printf "\n=== Region (Offset) Async Memset Tests ===\n";
  let total = 256 (* uint32 elements *) in
  let off_elems = 64 in
  let off_bytes = off_elems * 4 in
  let region_len = 64 in
  let stream = Stream.create () in
  let dptr = Deviceptr.mem_alloc ~size_in_bytes:(total * 4) in
  (* Sentinel-fill the whole allocation with 0xFFFFFFFF, then zero a non-zero-offset sub-region. *)
  Stream.memset_d32 dptr (Unsigned.UInt32.of_int32 (-1l)) ~length:total stream;
  Stream.memset_d32 ~offset:off_bytes dptr (Unsigned.UInt32.of_int 0) ~length:region_len stream;
  Stream.synchronize stream;
  let host = Bigarray.Array1.create Bigarray.Int32 Bigarray.C_layout total in
  Bigarray.Array1.fill host 0x5A5A5A5Al;
  Deviceptr.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 host) ~src:dptr ();
  let region_ok = ref true in
  let neighbours_ok = ref true in
  for i = 0 to total - 1 do
    if i >= off_elems && i < off_elems + region_len then (
      if host.{i} <> 0l then region_ok := false)
    else if host.{i} <> -1l then neighbours_ok := false
  done;
  Printf.printf "Async non-zero-offset memset_d32 targets correct sub-region: %s\n"
    (if !region_ok then "PASS" else "FAIL");
  Printf.printf "Neighbouring bytes untouched (async offset memset): %s\n"
    (if !neighbours_ok then "PASS" else "FAIL");
  Deviceptr.mem_free dptr

(* AC 5: byte offsets on [memcpy_peer]. The proposal's build host is single-GPU, so we copy with the
   current context as both ends (peer-to-self) -- the byte-offset arithmetic under test is identical
   to the cross-device case. (cuMemcpyPeer accepts identical src/dst contexts on this host.) *)
let test_region_peer_copy () =
  Printf.printf "\n=== Region Peer-Copy Offset Tests ===\n";
  let ctx = Context.get_current () in
  let total = 256 (* int32 elements *) in
  let len = 64 in
  let src_off_elems = 64 in
  let dst_off_elems = 128 in
  let payload = Bigarray.Array1.create Bigarray.Int32 Bigarray.C_layout len in
  for i = 0 to len - 1 do
    payload.{i} <- Int32.of_int ((i + 1) * 7)
  done;
  let src = Deviceptr.mem_alloc ~size_in_bytes:(total * 4) in
  let dst = Deviceptr.mem_alloc ~size_in_bytes:(total * 4) in
  Deviceptr.memset_d32 src (Unsigned.UInt32.of_int 0) ~length:total;
  Deviceptr.memset_d32 dst (Unsigned.UInt32.of_int32 (-1l)) ~length:total;
  (* Stage the payload at the source's non-zero byte offset, then peer-copy it to a *different*
     non-zero offset in [dst]. *)
  Deviceptr.memcpy_H_to_D ~length:len ~dst_offset:(src_off_elems * 4) ~dst:src
    ~src:(Bigarray.genarray_of_array1 payload) ();
  Deviceptr.memcpy_peer ~kind:Bigarray.Int32 ~length:len ~src_offset:(src_off_elems * 4)
    ~dst_offset:(dst_off_elems * 4) ~dst ~dst_ctx:ctx ~src ~src_ctx:ctx ();
  let host = Bigarray.Array1.create Bigarray.Int32 Bigarray.C_layout total in
  Bigarray.Array1.fill host 0x5A5A5A5Al;
  Deviceptr.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 host) ~src:dst ();
  let region_ok = ref true in
  let neighbours_ok = ref true in
  for i = 0 to total - 1 do
    if i >= dst_off_elems && i < dst_off_elems + len then (
      if host.{i} <> payload.{i - dst_off_elems} then region_ok := false)
    else if host.{i} <> -1l then neighbours_ok := false
  done;
  Printf.printf "Peer copy between non-zero offsets: %s\n" (if !region_ok then "PASS" else "FAIL");
  Printf.printf "Neighbouring bytes untouched (peer copy offset): %s\n"
    (if !neighbours_ok then "PASS" else "FAIL");
  Deviceptr.mem_free src;
  Deviceptr.mem_free dst

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
    test_device_offset_transfers ();
    test_region_memset ();
    test_region_memset_async ();
    test_region_kernel_launch ();
    test_region_peer_copy ();
    test_memory_set_operations ();
    test_pointer_utilities ();
    
    Printf.printf "\nMemory Module Tests Completed\n"
  ) else
    Printf.printf "No CUDA devices available for memory testing\n"

let () = run_tests ()