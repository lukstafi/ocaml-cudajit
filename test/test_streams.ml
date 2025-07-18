open Cuda

let simple_kernel = 
  {|
extern "C" __global__ void vector_add(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}
|}

let test_stream_creation () =
  Printf.printf "=== Stream Creation Tests ===\n";
  
  (* Test default stream creation *)
  let stream1 = Stream.create () in
  Printf.printf "Created default stream\n";
  
  (* Test stream creation with non-blocking flag *)
  let stream2 = Stream.create ~non_blocking:true () in
  Printf.printf "Created non-blocking stream\n";
  ignore stream2;
  
  (* Test stream creation with priority *)
  let stream3 = Stream.create ~lower_priority:1 () in
  Printf.printf "Created stream with lower priority\n";
  ignore stream3;
  
  (* Test stream creation with both flags *)
  let stream4 = Stream.create ~non_blocking:true ~lower_priority:2 () in
  Printf.printf "Created non-blocking stream with lower priority\n";
  ignore stream4;
  
  (* Test no_stream (NULL stream) *)
  let null_stream = Stream.no_stream in
  Printf.printf "Obtained NULL stream\n";
  ignore null_stream;
  
  (* Test stream properties *)
  let ctx = Stream.get_context stream1 in
  Printf.printf "Stream context retrieval: PASS\n";
  ignore ctx;
  
  let stream_id = Stream.get_id stream1 in
  Printf.printf "Stream ID retrieval: PASS\n";
  ignore stream_id

let test_stream_memory_operations () =
  Printf.printf "\n=== Stream Memory Operations Tests ===\n";
  
  let stream = Stream.create () in
  let size = 1024 in
  
  (* Test asynchronous memory allocation *)
  let dptr = Stream.mem_alloc stream ~size_in_bytes:(size * 4) in
  Printf.printf "Successfully allocated %d bytes asynchronously\n" (size * 4);
  
  (* Test asynchronous memory operations *)
  Stream.memset_d32 dptr (Unsigned.UInt32.of_int 0x12345678) ~length:size stream;
  Printf.printf "Successfully set memory asynchronously\n";
  
  (* Test stream synchronization *)
  Stream.synchronize stream;
  Printf.printf "Stream synchronized\n";
  
  (* Test asynchronous memory free *)
  Stream.mem_free stream dptr;
  Printf.printf "Successfully freed memory asynchronously\n";
  
  Stream.synchronize stream

let test_async_host_device_transfers () =
  Printf.printf "\n=== Async Host-Device Transfer Tests ===\n";
  
  let stream = Stream.create () in
  let size = 1024 in
  
  (* Create and initialize host data *)
  let host_array = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
  for i = 0 to size - 1 do
    host_array.{i} <- Float.of_int i
  done;
  
  let dptr = Stream.mem_alloc stream ~size_in_bytes:(size * 4) in
  
  (* Test asynchronous host to device transfer *)
  Stream.memcpy_H_to_D ~dst:dptr ~src:(Bigarray.genarray_of_array1 host_array) stream;
  Printf.printf "Successfully transferred data H->D asynchronously\n";
  
  (* Test asynchronous device to host transfer *)
  let result_array = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
  Stream.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 result_array) ~src:dptr stream;
  Printf.printf "Successfully transferred data D->H asynchronously\n";
  
  (* Wait for transfers to complete *)
  Stream.synchronize stream;
  
  (* Verify data integrity *)
  let mismatch = ref false in
  for i = 0 to size - 1 do
    if Float.abs (host_array.{i} -. result_array.{i}) > 1e-6 then (
      mismatch := true;
      Printf.printf "Async transfer mismatch at index %d: expected %f, got %f\n" 
        i host_array.{i} result_array.{i}
    )
  done;
  
  if not !mismatch then
    Printf.printf "Async transfer data integrity verified for %d elements\n" size;
  
  Stream.mem_free stream dptr;
  Stream.synchronize stream

let test_async_device_to_device_transfers () =
  Printf.printf "\n=== Async Device-to-Device Transfer Tests ===\n";
  
  let stream = Stream.create () in
  let size = 1024 in
  
  let dptr1 = Stream.mem_alloc stream ~size_in_bytes:(size * 4) in
  let dptr2 = Stream.mem_alloc stream ~size_in_bytes:(size * 4) in
  
  (* Initialize first device pointer *)
  Stream.memset_d32 dptr1 (Unsigned.UInt32.of_int 0xDEADBEEF) ~length:size stream;
  
  (* Test asynchronous device to device transfer *)
  Stream.memcpy_D_to_D ~kind:Bigarray.Float32 ~length:size ~dst:dptr2 ~src:dptr1 stream;
  Printf.printf "Successfully transferred data D->D asynchronously\n";
  
  (* Wait for operation to complete *)
  Stream.synchronize stream;
  
  (* Verify by copying back to host *)
  let result_array = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
  Stream.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 result_array) ~src:dptr2 stream;
  Stream.synchronize stream;
  
  (* Check if the pattern was transferred correctly *)
  let expected_value = Int32.bits_of_float (Int32.float_of_bits 0xDEADBEEFl) in
  let pattern_correct = ref true in
  for i = 0 to min 10 (size - 1) do
    let actual_value = Int32.bits_of_float result_array.{i} in
    if actual_value <> expected_value then (
      pattern_correct := false;
      Printf.printf "Pattern mismatch at index %d\n" i
    )
  done;
  
  if !pattern_correct then
    Printf.printf "Device-to-device transfer pattern verified\n";
  
  Stream.mem_free stream dptr1;
  Stream.mem_free stream dptr2;
  Stream.synchronize stream

let test_kernel_launches () =
  Printf.printf "\n=== Kernel Launch Tests ===\n";
  
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    let stream = Stream.create () in
    let size = 1024 in
    
    (* Compile kernel *)
    let prog = Nvrtc.compile_to_ptx ~cu_src:simple_kernel ~name:"vector_add" 
                ~options:["--use_fast_math"] ~with_debug:false in
    let module_ = Module.load_data_ex prog [] in
    let kernel = Module.get_function module_ ~name:"vector_add" in
    
    (* Allocate device memory *)
    let dptr_a = Stream.mem_alloc stream ~size_in_bytes:(size * 4) in
    let dptr_b = Stream.mem_alloc stream ~size_in_bytes:(size * 4) in
    let dptr_c = Stream.mem_alloc stream ~size_in_bytes:(size * 4) in
    
    (* Initialize input arrays *)
    Stream.memset_d32 dptr_a (Unsigned.UInt32.of_int 0x3F800000) ~length:size stream; (* 1.0f *)
    Stream.memset_d32 dptr_b (Unsigned.UInt32.of_int 0x40000000) ~length:size stream; (* 2.0f *)
    
    (* Launch kernel with different grid configurations *)
    let grid_sizes = [1; 4; 16; 32] in
    List.iter (fun grid_size ->
      let block_size = size / grid_size in
      if block_size > 0 && block_size <= 1024 then (
        Printf.printf "Launching kernel with grid_size=%d, block_size=%d\n" grid_size block_size;
        
        Stream.launch_kernel kernel ~grid_dim_x:grid_size ~block_dim_x:block_size
          ~shared_mem_bytes:0 stream
          [Tensor dptr_a; Tensor dptr_b; Tensor dptr_c; Int size];
        
        Stream.synchronize stream;
        Printf.printf "Kernel launch completed\n"
      )
    ) grid_sizes;
    
    (* Verify result *)
    let result_array = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
    Stream.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 result_array) ~src:dptr_c stream;
    Stream.synchronize stream;
    
    let correct_results = ref true in
    for i = 0 to min 10 (size - 1) do
      if Float.abs (result_array.{i} -. 3.0) > 1e-6 then (
        correct_results := false;
        Printf.printf "Kernel result mismatch at index %d: expected 3.0, got %f\n" 
          i result_array.{i}
      )
    done;
    
    if !correct_results then
      Printf.printf "Kernel execution results verified (1.0 + 2.0 = 3.0)\n";
    
    Stream.mem_free stream dptr_a;
    Stream.mem_free stream dptr_b;
    Stream.mem_free stream dptr_c;
    Stream.synchronize stream
  ) else
    Printf.printf "No devices available for kernel launch testing\n"

let test_stream_queries () =
  Printf.printf "\n=== Stream Query Tests ===\n";
  
  let stream = Stream.create () in
  let size = 1000 in
  let dptr = Stream.mem_alloc stream ~size_in_bytes:(size * 4) in
  
  (* Check stream status before operations *)
  let ready_before = Stream.is_ready stream in
  Printf.printf "Stream ready before operations: %b\n" ready_before;
  
  (* Launch asynchronous operations *)
  Stream.memset_d32 dptr (Unsigned.UInt32.of_int 0x12345678) ~length:size stream;
  Stream.memset_d32 dptr (Unsigned.UInt32.of_int 0x87654321) ~length:size stream;
  
  (* Check stream status during operations *)
  let ready_during = Stream.is_ready stream in
  Printf.printf "Stream ready during operations: %b\n" ready_during;
  
  (* Wait for completion *)
  Stream.synchronize stream;
  
  (* Check stream status after completion *)
  let ready_after = Stream.is_ready stream in
  Printf.printf "Stream ready after completion: %b\n" ready_after;
  
  Stream.mem_free stream dptr;
  Stream.synchronize stream

let test_concurrent_streams () =
  Printf.printf "\n=== Concurrent Stream Tests ===\n";
  
  let num_streams = 4 in
  let streams = Array.init num_streams (fun _ -> Stream.create ()) in
  let size = 1000 in
  
  (* Create device pointers for each stream *)
  let dptrs = Array.map (fun stream -> 
    Stream.mem_alloc stream ~size_in_bytes:(size * 4)
  ) streams in
  
  (* Launch concurrent operations on all streams *)
  Array.iteri (fun i stream ->
    let pattern = Int32.shift_left (Int32.of_int i) 24 in
    Stream.memset_d32 dptrs.(i) (Unsigned.UInt32.of_int32 pattern) ~length:size stream;
    Printf.printf "Launched operation on stream %d\n" i
  ) streams;
  
  (* Check stream readiness *)
  Array.iteri (fun i stream ->
    let ready = Stream.is_ready stream in
    Printf.printf "Stream %d ready: %b\n" i ready
  ) streams;
  
  (* Synchronize all streams *)
  Array.iteri (fun i stream ->
    Stream.synchronize stream;
    Printf.printf "Stream %d synchronized\n" i
  ) streams;
  
  (* Clean up *)
  Array.iteri (fun i stream ->
    Stream.mem_free stream dptrs.(i);
    Stream.synchronize stream
  ) streams;
  
  Printf.printf "Concurrent streams test completed\n"

let run_tests () =
  Printf.printf "Starting Stream Module Tests\n\n";
  init ();
  
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    test_stream_creation ();
    test_stream_memory_operations ();
    test_async_host_device_transfers ();
    test_async_device_to_device_transfers ();
    test_kernel_launches ();
    test_stream_queries ();
    test_concurrent_streams ();
    
    Printf.printf "\nStream Module Tests Completed\n"
  ) else
    Printf.printf "No CUDA devices available for stream testing\n"

let () = run_tests ()