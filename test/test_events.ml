open Cuda

let simple_kernel = 
  {|
extern "C" __global__ void delay_kernel(float *data, int iterations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int i = 0; i < iterations; i++) {
    sum += (float)i;
  }
  if (idx < 1000) {
    data[idx] = sum;
  }
}
|}

let test_basic_event_operations () =
  Printf.printf "=== Basic Event Operations Tests ===\n";
  
  let stream = Stream.create () in
  
  (* Test event creation with different flags *)
  let event1 = Event.create () in
  let event2 = Event.create ~enable_timing:true () in
  let event3 = Event.create ~blocking_sync:true () in
  let event4 = Event.create ~enable_timing:true ~blocking_sync:true () in
  
  Printf.printf "Successfully created 4 events with different flags\n";
  
  (* Test event recording *)
  Event.record event1 stream;
  Event.record event2 stream;
  Event.record event3 stream;
  Event.record event4 stream;
  
  Printf.printf "Successfully recorded events on stream\n";
  
  (* Test event querying *)
  let ready1 = Event.query event1 in
  let ready2 = Event.query event2 in
  Printf.printf "Event 1 ready: %b, Event 2 ready: %b\n" ready1 ready2;
  
  (* Test event synchronization *)
  Event.synchronize event1;
  Event.synchronize event2;
  Printf.printf "Successfully synchronized events\n";
  
  (* Check if events are ready after sync *)
  let ready_after_sync = Event.query event1 in
  Printf.printf "Event 1 ready after sync: %b\n" ready_after_sync

let test_event_timing () =
  Printf.printf "\n=== Event Timing Tests ===\n";
  
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    (* Compile a simple delay kernel *)
    let prog = Nvrtc.compile_to_ptx ~cu_src:simple_kernel ~name:"delay_kernel" 
                ~options:["--use_fast_math"] ~with_debug:false in
    let module_ = Module.load_data_ex prog [] in
    let kernel = Module.get_function module_ ~name:"delay_kernel" in
    
    let stream = Stream.create () in
    let size = 1000 in
    let dptr = Deviceptr.mem_alloc ~size_in_bytes:(size * 4) in
    
    (* Create timing events *)
    let start_event = Event.create ~enable_timing:true () in
    let end_event = Event.create ~enable_timing:true () in
    
    (* Record start event *)
    Event.record start_event stream;
    
    (* Launch kernel with some work *)
    Stream.launch_kernel kernel ~grid_dim_x:10 ~block_dim_x:100 
      ~shared_mem_bytes:0 stream
      [Tensor dptr; Int 100000];
    
    (* Record end event *)
    Event.record end_event stream;
    
    (* Wait for completion *)
    Stream.synchronize stream;
    
    (* Check events are complete *)
    let start_ready = Event.query start_event in
    let end_ready = Event.query end_event in
    Printf.printf "Event completion check: %s\n" (if start_ready && end_ready then "PASS" else "FAIL");
    
    if start_ready && end_ready then (
      let elapsed = Event.elapsed_time ~start:start_event ~end_:end_event in
      Printf.printf "Timing measurement: %s\n" (if elapsed >= 0.0 then "PASS" else "FAIL");
    ) else (
      Printf.printf "Timing measurement: FAIL\n"
    );
    
    Deviceptr.mem_free dptr
  ) else
    Printf.printf "No devices available for event timing testing\n"

let test_event_stream_synchronization () =
  Printf.printf "\n=== Event Stream Synchronization Tests ===\n";
  
  let stream1 = Stream.create () in
  let stream2 = Stream.create () in
  
  let size = 1000 in
  let dptr1 = Deviceptr.mem_alloc ~size_in_bytes:(size * 4) in
  let dptr2 = Deviceptr.mem_alloc ~size_in_bytes:(size * 4) in
  
  (* Create synchronization event *)
  let sync_event = Event.create () in
  
  (* Launch work on stream1 *)
  Stream.memset_d32 dptr1 (Unsigned.UInt32.of_int 0x12345678) ~length:size stream1;
  
  (* Record event on stream1 *)
  Event.record sync_event stream1;
  
  (* Make stream2 wait for stream1's event *)
  Event.wait stream2 sync_event;
  
  (* Launch work on stream2 that depends on stream1 *)
  Stream.memcpy_D_to_D ~kind:Bigarray.Float32 ~length:size ~dst:dptr2 ~src:dptr1 stream2;
  
  (* Synchronize both streams *)
  Stream.synchronize stream1;
  Stream.synchronize stream2;
  
  Printf.printf "Successfully synchronized streams using events\n";
  
  Deviceptr.mem_free dptr1;
  Deviceptr.mem_free dptr2

let test_delimited_events () =
  Printf.printf "\n=== Delimited Event Tests ===\n";
  
  let stream = Stream.create () in
  let size = 1000 in
  let dptr = Deviceptr.mem_alloc ~size_in_bytes:(size * 4) in
  
  (* Test delimited event creation and recording *)
  let delimited_event = Delimited_event.record stream in
  Printf.printf "Created delimited event\n";
  
  (* Check initial state *)
  let initially_released = Delimited_event.is_released delimited_event in
  Printf.printf "Initially released: %b\n" initially_released;
  
  (* Launch some work *)
  Stream.memset_d32 dptr (Unsigned.UInt32.of_int 0xDEADBEEF) ~length:size stream;
  
  (* Query event status *)
  let ready = Delimited_event.query delimited_event in
  Printf.printf "Delimited event ready: %b\n" ready;
  
  (* Synchronize the event *)
  Delimited_event.synchronize delimited_event;
  Printf.printf "Synchronized delimited event\n";
  
  (* Check if released after synchronization *)
  let released_after_sync = Delimited_event.is_released delimited_event in
  Printf.printf "Released after sync: %b\n" released_after_sync;
  
  (* Test with another stream *)
  let stream2 = Stream.create () in
  let delimited_event2 = Delimited_event.record stream2 in
  
  (* Make stream wait for delimited event *)
  Stream.memset_d32 dptr (Unsigned.UInt32.of_int 0xCAFEBABE) ~length:size stream2;
  Delimited_event.wait stream delimited_event2;
  
  (* Synchronize stream to release the delimited event *)
  Stream.synchronize stream;
  
  let released_after_stream_sync = Delimited_event.is_released delimited_event2 in
  Printf.printf "Delimited event released after stream sync: %b\n" released_after_stream_sync;
  
  Deviceptr.mem_free dptr

let test_multiple_events_timing () =
  Printf.printf "\n=== Multiple Events Timing Tests ===\n";
  
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    let stream = Stream.create () in
    let size = 1000 in
    let dptr = Deviceptr.mem_alloc ~size_in_bytes:(size * 4) in
    
    (* Create multiple timing events *)
    let events = Array.init 5 (fun _ -> Event.create ~enable_timing:true ()) in
    
    (* Record events with operations between them *)
    Event.record events.(0) stream;
    
    Stream.memset_d32 dptr (Unsigned.UInt32.of_int 0x11111111) ~length:size stream;
    Event.record events.(1) stream;
    
    Stream.memset_d32 dptr (Unsigned.UInt32.of_int 0x22222222) ~length:size stream;
    Event.record events.(2) stream;
    
    Stream.memset_d32 dptr (Unsigned.UInt32.of_int 0x33333333) ~length:size stream;
    Event.record events.(3) stream;
    
    Stream.memset_d32 dptr (Unsigned.UInt32.of_int 0x44444444) ~length:size stream;
    Event.record events.(4) stream;
    
    (* Wait for all events to complete *)
    Stream.synchronize stream;
    
    (* Check all events completed successfully *)
    let all_ready = ref true in
    for i = 0 to 4 do
      if not (Event.query events.(i)) then
        all_ready := false
    done;
    
    Printf.printf "Multiple events completion: %s\n" (if !all_ready then "PASS" else "FAIL");
    
    (* Test timing measurements work *)
    if !all_ready then (
      let elapsed = Event.elapsed_time ~start:events.(0) ~end_:events.(4) in
      Printf.printf "Total timing measurement: %s\n" (if elapsed >= 0.0 then "PASS" else "FAIL");
    ) else (
      Printf.printf "Total timing measurement: FAIL\n"
    );
    
    Deviceptr.mem_free dptr
  ) else
    Printf.printf "No devices available for multiple events timing testing\n"

let run_tests () =
  Printf.printf "Starting Event Module Tests\n\n";
  init ();
  
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    test_basic_event_operations ();
    test_event_timing ();
    test_event_stream_synchronization ();
    test_delimited_events ();
    test_multiple_events_timing ();
    
    Printf.printf "\nEvent Module Tests Completed\n"
  ) else
    Printf.printf "No CUDA devices available for event testing\n"

let () = run_tests ()