open Cuda

let vector_add_kernel = 
  {|
extern "C" __global__ void vector_add(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}
|}

let reduction_kernel = 
  {|
extern "C" __global__ void reduction(float *input, float *output, int n) {
  extern __shared__ float sdata[];
  
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Load data into shared memory
  sdata[tid] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();
  
  // Perform reduction in shared memory
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // Write result for this block to global memory
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}
|}

let test_multi_stream_pipeline () =
  Printf.printf "=== Multi-Stream Pipeline Test ===\n";
  
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    (* Compile kernel *)
    let prog = Nvrtc.compile_to_ptx ~cu_src:vector_add_kernel ~name:"vector_add" 
                ~options:["--use_fast_math"] ~with_debug:false in
    let module_ = Module.load_data_ex prog [] in
    let kernel = Module.get_function module_ ~name:"vector_add" in
    
    let num_streams = 3 in
    let streams = Array.init num_streams (fun _ -> Stream.create ()) in
    let size_per_stream = 1024 in
    
    (* Create host data *)
    let host_a = Array.init num_streams (fun _ ->
      let arr = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size_per_stream in
      for i = 0 to size_per_stream - 1 do
        arr.{i} <- Float.of_int i
      done;
      arr
    ) in
    
    let host_b = Array.init num_streams (fun _ ->
      let arr = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size_per_stream in
      for i = 0 to size_per_stream - 1 do
        arr.{i} <- Float.of_int (i * 2)
      done;
      arr
    ) in
    
    let host_c = Array.init num_streams (fun _ ->
      Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size_per_stream
    ) in
    
    (* Allocate device memory for each stream *)
    let dev_a = Array.map (fun stream ->
      Stream.mem_alloc stream ~size_in_bytes:(size_per_stream * 4)
    ) streams in
    
    let dev_b = Array.map (fun stream ->
      Stream.mem_alloc stream ~size_in_bytes:(size_per_stream * 4)
    ) streams in
    
    let dev_c = Array.map (fun stream ->
      Stream.mem_alloc stream ~size_in_bytes:(size_per_stream * 4)
    ) streams in
    
    (* Create events for synchronization *)
    let events = Array.init num_streams (fun _ -> Event.create ~enable_timing:true ()) in
    
    (* Pipeline: Transfer -> Compute -> Transfer back *)
    Array.iteri (fun i stream ->
      Printf.printf "Starting pipeline for stream %d\n" i;
      
      (* Transfer input data to device *)
      Stream.memcpy_H_to_D ~dst:dev_a.(i) ~src:(Bigarray.genarray_of_array1 host_a.(i)) stream;
      Stream.memcpy_H_to_D ~dst:dev_b.(i) ~src:(Bigarray.genarray_of_array1 host_b.(i)) stream;
      
      (* Record event before computation *)
      Event.record events.(i) stream;
      
      (* Launch kernel *)
      let grid_size = (size_per_stream + 255) / 256 in
      let block_size = 256 in
      Stream.launch_kernel kernel ~grid_dim_x:grid_size ~block_dim_x:block_size
        ~shared_mem_bytes:0 stream
        [Tensor dev_a.(i); Tensor dev_b.(i); Tensor dev_c.(i); Int size_per_stream];
      
      (* Transfer result back to host *)
      Stream.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 host_c.(i)) ~src:dev_c.(i) stream;
      
      Printf.printf "Pipeline launched for stream %d\n" i
    ) streams;
    
    (* Wait for all streams to complete *)
    Array.iteri (fun i stream ->
      Stream.synchronize stream;
      Printf.printf "Stream %d completed\n" i
    ) streams;
    
    (* Verify results *)
    let all_correct = ref true in
    Array.iteri (fun stream_idx result_array ->
      for i = 0 to min 10 (size_per_stream - 1) do
        let expected = host_a.(stream_idx).{i} +. host_b.(stream_idx).{i} in
        let actual = result_array.{i} in
        if Float.abs (expected -. actual) > 1e-6 then (
          all_correct := false;
          Printf.printf "Stream %d: mismatch at index %d: expected %f, got %f\n" 
            stream_idx i expected actual
        )
      done
    ) host_c;
    
    if !all_correct then
      Printf.printf "All pipeline results verified correctly\n";
    
    (* Check timing *)
    Array.iteri (fun i event ->
      if Event.query event then
        Printf.printf "Stream %d computation completed\n" i
    ) events;
    
    (* Clean up *)
    Array.iteri (fun i stream ->
      Stream.mem_free stream dev_a.(i);
      Stream.mem_free stream dev_b.(i);
      Stream.mem_free stream dev_c.(i);
      Stream.synchronize stream
    ) streams;
    
    Printf.printf "Multi-stream pipeline test completed\n"
  ) else
    Printf.printf "No devices available for multi-stream pipeline testing\n"

let test_event_driven_workflow () =
  Printf.printf "\n=== Event-Driven Workflow Test ===\n";
  
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    let stream1 = Stream.create () in
    let stream2 = Stream.create () in
    let stream3 = Stream.create () in
    
    let size = 1024 in
    let dptr1 = Stream.mem_alloc stream1 ~size_in_bytes:(size * 4) in
    let dptr2 = Stream.mem_alloc stream2 ~size_in_bytes:(size * 4) in
    let dptr3 = Stream.mem_alloc stream3 ~size_in_bytes:(size * 4) in
    
    (* Create events for workflow coordination *)
    let event1 = Event.create () in
    let event2 = Event.create () in
    
    (* Stage 1: Initialize data on stream1 *)
    Stream.memset_d32 dptr1 (Unsigned.UInt32.of_int 0x3F800000) ~length:size stream1; (* 1.0f *)
    Event.record event1 stream1;
    Printf.printf "Stage 1: Data initialized on stream1\n";
    
    (* Stage 2: Wait for stage 1, then process on stream2 *)
    Event.wait stream2 event1;
    Stream.memcpy_D_to_D ~kind:Bigarray.Float32 ~length:size ~dst:dptr2 ~src:dptr1 stream2;
    Stream.memset_d32 dptr2 (Unsigned.UInt32.of_int 0x40000000) ~length:size stream2; (* 2.0f *)
    Event.record event2 stream2;
    Printf.printf "Stage 2: Data processed on stream2\n";
    
    (* Stage 3: Wait for stage 2, then finalize on stream3 *)
    Event.wait stream3 event2;
    Stream.memcpy_D_to_D ~kind:Bigarray.Float32 ~length:size ~dst:dptr3 ~src:dptr2 stream3;
    Printf.printf "Stage 3: Data finalized on stream3\n";
    
    (* Wait for all stages to complete *)
    Event.synchronize event1;
    Event.synchronize event2;
    Stream.synchronize stream3;
    
    Printf.printf "Event-driven workflow completed\n";
    
    (* Verify final result *)
    let result_array = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
    Stream.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 result_array) ~src:dptr3 stream3;
    Stream.synchronize stream3;
    
    let correct = ref true in
    for i = 0 to min 10 (size - 1) do
      if Float.abs (result_array.{i} -. 2.0) > 1e-6 then (
        correct := false;
        Printf.printf "Final result mismatch at index %d: expected 2.0, got %f\n" 
          i result_array.{i}
      )
    done;
    
    if !correct then
      Printf.printf "Event-driven workflow results verified\n";
    
    (* Clean up *)
    Stream.mem_free stream1 dptr1;
    Stream.mem_free stream2 dptr2;
    Stream.mem_free stream3 dptr3;
    Stream.synchronize stream1;
    Stream.synchronize stream2;
    Stream.synchronize stream3
  ) else
    Printf.printf "No devices available for event-driven workflow testing\n"

let test_reduction_workflow () =
  Printf.printf "\n=== Reduction Workflow Test ===\n";
  
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    (* Compile reduction kernel *)
    let prog = Nvrtc.compile_to_ptx ~cu_src:reduction_kernel ~name:"reduction" 
                ~options:["--use_fast_math"] ~with_debug:false in
    let module_ = Module.load_data_ex prog [] in
    let kernel = Module.get_function module_ ~name:"reduction" in
    
    let stream = Stream.create () in
    let size = 1024 in
    let block_size = 256 in
    let grid_size = (size + block_size - 1) / block_size in
    
    (* Create host input data *)
    let host_input = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout size in
    for i = 0 to size - 1 do
      host_input.{i} <- 1.0 (* Sum should be 1024.0 *)
    done;
    
    (* Allocate device memory *)
    let dev_input = Stream.mem_alloc stream ~size_in_bytes:(size * 4) in
    let dev_output = Stream.mem_alloc stream ~size_in_bytes:(grid_size * 4) in
    
    (* Transfer input data *)
    Stream.memcpy_H_to_D ~dst:dev_input ~src:(Bigarray.genarray_of_array1 host_input) stream;
    
    (* Launch reduction kernel *)
    let shared_mem_bytes = block_size * 4 in (* 4 bytes per float *)
    Stream.launch_kernel kernel ~grid_dim_x:grid_size ~block_dim_x:block_size
      ~shared_mem_bytes stream
      [Tensor dev_input; Tensor dev_output; Int size];
    
    (* Transfer result back *)
    let host_output = Bigarray.Array1.create Bigarray.Float32 Bigarray.C_layout grid_size in
    Stream.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 host_output) ~src:dev_output stream;
    
    (* Wait for completion *)
    Stream.synchronize stream;
    
    (* Compute final sum on host *)
    let final_sum = ref 0.0 in
    for i = 0 to grid_size - 1 do
      final_sum := !final_sum +. host_output.{i}
    done;
    
    Printf.printf "Reduction result: %f (expected: %f)\n" !final_sum (Float.of_int size);
    
    if Float.abs (!final_sum -. Float.of_int size) < 1e-3 then
      Printf.printf "Reduction workflow verified successfully\n"
    else
      Printf.printf "Reduction workflow failed verification\n";
    
    (* Clean up *)
    Stream.mem_free stream dev_input;
    Stream.mem_free stream dev_output;
    Stream.synchronize stream
  ) else
    Printf.printf "No devices available for reduction workflow testing\n"

let test_resource_lifecycle () =
  Printf.printf "\n=== Resource Lifecycle Test ===\n";
  
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    Printf.printf "Testing resource creation and cleanup cycles\n";
    
    (* Test multiple allocation/deallocation cycles *)
    for cycle = 1 to 5 do
      Printf.printf "Resource cycle %d\n" cycle;
      
      (* Create multiple streams *)
      let streams = Array.init 4 (fun _ -> Stream.create ()) in
      
      (* Allocate memory on each stream *)
      let dptrs = Array.map (fun stream ->
        Stream.mem_alloc stream ~size_in_bytes:(1024 * 4)
      ) streams in
      
      (* Perform operations *)
      Array.iteri (fun i stream ->
        Stream.memset_d32 dptrs.(i) (Unsigned.UInt32.of_int (i * 0x11111111)) ~length:1024 stream
      ) streams;
      
      (* Create events *)
      let events = Array.map (fun stream ->
        let event = Event.create ~enable_timing:true () in
        Event.record event stream;
        event
      ) streams in
      
      (* Wait for completion *)
      Array.iter Event.synchronize events;
      
      (* Clean up explicitly *)
      Array.iteri (fun i stream ->
        Stream.mem_free stream dptrs.(i);
        Stream.synchronize stream
      ) streams;
      
      Printf.printf "Cycle %d completed\n" cycle
    done;
    
    (* Force garbage collection *)
    Gc.full_major ();
    Printf.printf "Resource lifecycle test completed\n"
  ) else
    Printf.printf "No devices available for resource lifecycle testing\n"

let test_error_handling () =
  Printf.printf "\n=== Error Handling Test ===\n";
  
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    (* Test invalid memory allocation *)
    (try
       let huge_size = Int.max_int / 2 in
       let _ = Deviceptr.mem_alloc ~size_in_bytes:huge_size in
       Printf.printf "ERROR: Should have failed with huge allocation\n"
     with
     | Cuda_error {message; _} -> 
         Printf.printf "Correctly handled huge allocation error: %s\n" message);
    
    (* Test invalid kernel compilation *)
    (try
       let invalid_kernel = "invalid C++ code here" in
       let _ = Nvrtc.compile_to_ptx ~cu_src:invalid_kernel ~name:"invalid" 
                 ~options:[] ~with_debug:false in
       Printf.printf "ERROR: Should have failed with invalid kernel\n"
     with
     | Nvrtc.Nvrtc_error {message; _} -> 
         Printf.printf "Correctly handled invalid kernel error: %s\n" message);
    
    (* Test invalid function name *)
    (try
       let prog = Nvrtc.compile_to_ptx ~cu_src:vector_add_kernel ~name:"vector_add" 
                    ~options:[] ~with_debug:false in
       let module_ = Module.load_data_ex prog [] in
       let _ = Module.get_function module_ ~name:"nonexistent_function" in
       Printf.printf "ERROR: Should have failed with invalid function name\n"
     with
     | Cuda_error {message; _} -> 
         Printf.printf "Correctly handled invalid function name error: %s\n" message);
    
    Printf.printf "Error handling test completed\n"
  ) else
    Printf.printf "No devices available for error handling testing\n"

let run_tests () =
  Printf.printf "Starting Integration Tests\n\n";
  init ();
  
  test_multi_stream_pipeline ();
  test_event_driven_workflow ();
  test_reduction_workflow ();
  test_resource_lifecycle ();
  test_error_handling ();
  
  Printf.printf "\nIntegration Tests Completed\n"

let () = run_tests ()