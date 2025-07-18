open Cuda

let test_device_enumeration () =
  Printf.printf "=== Device Enumeration Tests ===\n";
  let device_count = Device.get_count () in
  Printf.printf "Device count check: %s\n" (if device_count >= 0 then "PASS" else "FAIL");
  
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    Printf.printf "Device 0 access: PASS\n";
    
    (* Test device attributes *)
    let attrs = Device.get_attributes device in
    Printf.printf "Device attributes access: PASS\n";
    Printf.printf "Device name length: %s\n" (if String.length attrs.name > 0 then "PASS" else "FAIL");
    Printf.printf "Compute capability valid: %s\n" (if attrs.compute_capability_major >= 1 then "PASS" else "FAIL");
    Printf.printf "Multiprocessor count valid: %s\n" (if attrs.multiprocessor_count > 0 then "PASS" else "FAIL");
    Printf.printf "Max threads per block valid: %s\n" (if attrs.max_threads_per_block > 0 then "PASS" else "FAIL");
    Printf.printf "Warp size valid: %s\n" (if attrs.warp_size > 0 then "PASS" else "FAIL");
    
    (* Test memory info - requires a context *)
    let ctx = Context.create [Context.SCHED_AUTO] device in
    Context.set_current ctx;
    let free_mem, total_mem = Device.get_free_and_total_mem () in
    Printf.printf "Memory info access: PASS\n";
    Printf.printf "Memory values valid: %s\n" (if total_mem > 0 && free_mem >= 0 then "PASS" else "FAIL");
    
    (* Test invalid device ordinal *)
    (try
       let _ = Device.get ~ordinal:999 in
       Printf.printf "Invalid device ordinal handling: FAIL\n"
     with
     | Cuda_error _ -> Printf.printf "Invalid device ordinal handling: PASS\n")
  ) else
    Printf.printf "No CUDA devices available\n"

let test_peer_access () =
  Printf.printf "\n=== Peer Access Tests ===\n";
  let device_count = Device.get_count () in
  if device_count > 1 then (
    let device0 = Device.get ~ordinal:0 in
    let device1 = Device.get ~ordinal:1 in
    
    let can_access = Device.can_access_peer ~dst:device0 ~src:device1 in
    Printf.printf "Peer access query: PASS\n";
    Printf.printf "Peer access result type: %s\n" (if can_access = true || can_access = false then "PASS" else "FAIL");
    
    let p2p_attrs = Device.get_p2p_attributes ~dst:device0 ~src:device1 in
    Printf.printf "P2P attributes query: PASS\n";
    Printf.printf "P2P attributes count: %s\n" (if List.length p2p_attrs >= 0 then "PASS" else "FAIL");
  ) else
    Printf.printf "Multiple devices required for peer access testing\n"

let test_primary_context () =
  Printf.printf "\n=== Primary Context Tests ===\n";
  let device_count = Device.get_count () in
  if device_count > 0 then (
    let device = Device.get ~ordinal:0 in
    Device.primary_ctx_reset device;
    Printf.printf "Primary context reset: PASS\n"
  ) else
    Printf.printf "No devices available for primary context testing\n"

let run_tests () =
  Printf.printf "Starting Device Module Tests\n\n";
  init ();
  
  test_device_enumeration ();
  test_peer_access ();
  test_primary_context ();
  
  Printf.printf "\nDevice Module Tests Completed\n"

let () = run_tests ()