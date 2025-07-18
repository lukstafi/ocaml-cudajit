open Cuda

let test_context_creation () =
  Printf.printf "=== Context Creation Tests ===\n";
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    
    (* Test context creation with different flags *)
    let ctx1 = Context.create [Context.SCHED_AUTO] device in
    Printf.printf "Created context with SCHED_AUTO flag\n";
    ignore ctx1;
    
    let ctx2 = Context.create [Context.SCHED_YIELD] device in
    Printf.printf "Created context with SCHED_YIELD flag\n";
    ignore ctx2;
    
    let ctx3 = Context.create [Context.SCHED_BLOCKING_SYNC] device in
    Printf.printf "Created context with SCHED_BLOCKING_SYNC flag\n";
    ignore ctx3;
    
    (* Test primary context *)
    let primary_ctx = Context.get_primary device in
    Printf.printf "Obtained primary context\n";
    ignore primary_ctx;
    
    Printf.printf "All context creation tests passed\n"
  ) else
    Printf.printf "No devices available for context testing\n"

let test_context_stack () =
  Printf.printf "\n=== Context Stack Tests ===\n";
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx1 = Context.create [Context.SCHED_AUTO] device in
    let ctx2 = Context.create [Context.SCHED_YIELD] device in
    
    (* Test context stack operations *)
    Context.push_current ctx1;
    Printf.printf "Pushed context 1 to stack\n";
    
    let current_ctx = Context.get_current () in
    Printf.printf "Retrieved current context\n";
    ignore current_ctx;
    
    Context.set_current ctx2;
    Printf.printf "Set current context to context 2\n";
    
    let popped_ctx = Context.pop_current () in
    Printf.printf "Popped context from stack\n";
    ignore popped_ctx;
    
    Printf.printf "Context stack tests completed\n"
  ) else
    Printf.printf "No devices available for context stack testing\n"

let test_context_limits () =
  Printf.printf "\n=== Context Limits Tests ===\n";
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    (* Test different context limits *)
    let limits = [
      Context.STACK_SIZE;
      Context.PRINTF_FIFO_SIZE;
      Context.MALLOC_HEAP_SIZE;
      Context.DEV_RUNTIME_SYNC_DEPTH;
    ] in
    
    List.iter (fun limit ->
      try
        let current_limit = Context.get_limit limit in
        Printf.printf "Current %s: %d\n" 
          (Sexplib0.Sexp.to_string_hum (Context.sexp_of_limit limit)) current_limit;
        
        (* Test setting a new limit (conservative values) *)
        match limit with
        | Context.STACK_SIZE -> 
            Context.set_limit limit (max 1024 current_limit);
            Printf.printf "Set stack size limit\n"
        | Context.PRINTF_FIFO_SIZE -> 
            Context.set_limit limit (max 1024 current_limit);
            Printf.printf "Set printf fifo size limit\n"
        | Context.MALLOC_HEAP_SIZE -> 
            Context.set_limit limit (max 1024 current_limit);
            Printf.printf "Set malloc heap size limit\n"
        | Context.DEV_RUNTIME_SYNC_DEPTH -> 
            Context.set_limit limit (max 1 current_limit);
            Printf.printf "Set device runtime sync depth limit\n"
        | _ -> ()
      with
      | Cuda_error {message; _} -> 
          Printf.printf "Error with limit %s: %s\n" 
            (Sexplib0.Sexp.to_string_hum (Context.sexp_of_limit limit)) message
    ) limits;
    
    Printf.printf "Context limits tests completed\n"
  ) else
    Printf.printf "No devices available for context limits testing\n"

let test_context_flags () =
  Printf.printf "\n=== Context Flags Tests ===\n";
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO; Context.LMEM_RESIZE_TO_MAX] device in
    Context.set_current ctx;
    
    let flags = Context.get_flags () in
    Printf.printf "Context flags: %s\n" (Sexplib0.Sexp.to_string_hum (Context.sexp_of_flags flags));
    
    Printf.printf "Context flags tests completed\n"
  ) else
    Printf.printf "No devices available for context flags testing\n"

let test_context_synchronization () =
  Printf.printf "\n=== Context Synchronization Tests ===\n";
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    
    Printf.printf "Testing context synchronization\n";
    Context.synchronize ();
    Printf.printf "Context synchronization completed\n"
  ) else
    Printf.printf "No devices available for context synchronization testing\n"

let run_tests () =
  Printf.printf "Starting Context Module Tests\n\n";
  init ();
  
  test_context_creation ();
  test_context_stack ();
  test_context_limits ();
  test_context_flags ();
  test_context_synchronization ();
  
  Printf.printf "\nContext Module Tests Completed\n"

let () = run_tests ()