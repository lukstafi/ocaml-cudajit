(* Stream-ordered memory pool and graph-allocator statistics.
   Allocates and frees asynchronously and verifies that the byte-count readings
   and release-threshold behavior move as the CUDA driver documents, printing
   only device-independent claims. *)

let failures = ref 0

let check label holds =
  if not holds then incr failures;
  Printf.printf "%s: %b\n" label holds

let () =
  let module Cu = Cuda in
  let module U64 = Unsigned.UInt64 in
  Cu.init ();
  let device = Cu.Device.get ~ordinal:0 in
  let props = Cu.Device.get_attributes device in
  check "device supports memory pools" props.memory_pools_supported;
  let context = Cu.Context.create [] device in
  let default_pool = Cu.Mem_pool.get_default device in
  let current_pool = Cu.Mem_pool.get_current device in
  let pool_attr = Cu.Mem_pool.get_attribute in
  let graph_attr = Cu.Graph.get_mem_attribute device in
  let zero = U64.zero in
  let alloc_bytes = 4 * 1024 * 1024 in
  let alloc_u64 = U64.of_int alloc_bytes in

  (* Exercise sexp serializers on the new types. *)
  let pool_sexp = Sexplib0.Sexp.to_string (Cu.Mem_pool.sexp_of_t default_pool) in
  check "pool sexp is formatted as pointer" (String.length pool_sexp > 1 && pool_sexp.[0] = '@');
  check "pool attribute sexps match constructors"
    (List.for_all
       (fun (a, expected) ->
         String.equal (Sexplib0.Sexp.to_string (Cu.Mem_pool.sexp_of_attribute a)) expected)
       [
         (Cu.Mem_pool.RELEASE_THRESHOLD, "RELEASE_THRESHOLD");
         (RESERVED_MEM_CURRENT, "RESERVED_MEM_CURRENT");
         (RESERVED_MEM_HIGH, "RESERVED_MEM_HIGH");
         (USED_MEM_CURRENT, "USED_MEM_CURRENT");
         (USED_MEM_HIGH, "USED_MEM_HIGH");
       ]);
  check "graph mem_attribute sexps match constructors"
    (List.for_all
       (fun (a, expected) ->
         String.equal (Sexplib0.Sexp.to_string (Cu.Graph.sexp_of_mem_attribute a)) expected)
       [
         (Cu.Graph.USED_MEM_CURRENT, "USED_MEM_CURRENT");
         (USED_MEM_HIGH, "USED_MEM_HIGH");
         (RESERVED_MEM_CURRENT, "RESERVED_MEM_CURRENT");
         (RESERVED_MEM_HIGH, "RESERVED_MEM_HIGH");
       ]);

  (* Initial state before any async or graph allocation in this process. *)
  check "initial release threshold is 0"
    (U64.equal (pool_attr default_pool RELEASE_THRESHOLD) zero);
  check "initial current pool threshold matches default"
    (U64.equal (pool_attr current_pool RELEASE_THRESHOLD) zero);
  check "initial pool used current is 0" (U64.equal (pool_attr default_pool USED_MEM_CURRENT) zero);
  check "initial pool used high is 0" (U64.equal (pool_attr default_pool USED_MEM_HIGH) zero);
  check "initial pool reserved current is 0"
    (U64.equal (pool_attr default_pool RESERVED_MEM_CURRENT) zero);
  check "initial pool reserved high is 0"
    (U64.equal (pool_attr default_pool RESERVED_MEM_HIGH) zero);

  let stream = Cu.Stream.create () in

  (* 1. Default threshold (0): async allocation raises used and reserved; async free + sync drops
     both current readings back to 0 while keeping high-water marks. *)
  let p1 = Cu.Stream.mem_alloc stream ~size_in_bytes:alloc_bytes in
  Cu.Stream.synchronize stream;
  let used_alloc = pool_attr default_pool USED_MEM_CURRENT in
  let used_high_alloc = pool_attr default_pool USED_MEM_HIGH in
  let reserved_alloc = pool_attr default_pool RESERVED_MEM_CURRENT in
  let reserved_high_alloc = pool_attr default_pool RESERVED_MEM_HIGH in
  check "after alloc: pool used covers allocation" (U64.compare used_alloc alloc_u64 >= 0);
  check "after alloc: current pool matches default pool"
    (U64.equal (pool_attr current_pool USED_MEM_CURRENT) used_alloc);
  check "after alloc: pool used high >= used current" (U64.compare used_high_alloc used_alloc >= 0);
  check "after alloc: pool reserved >= used" (U64.compare reserved_alloc used_alloc >= 0);
  check "after alloc: pool reserved high >= reserved current"
    (U64.compare reserved_high_alloc reserved_alloc >= 0);

  Cu.Stream.mem_free stream p1;
  Cu.Stream.synchronize stream;
  check "after free+sync (threshold 0): pool used is 0"
    (U64.equal (pool_attr default_pool USED_MEM_CURRENT) zero);
  check "after free+sync (threshold 0): pool reserved is 0"
    (U64.equal (pool_attr default_pool RESERVED_MEM_CURRENT) zero);
  check "after free+sync (threshold 0): pool used high retained"
    (U64.compare (pool_attr default_pool USED_MEM_HIGH) used_high_alloc >= 0);
  check "after free+sync (threshold 0): pool reserved high retained"
    (U64.compare (pool_attr default_pool RESERVED_MEM_HIGH) reserved_high_alloc >= 0);

  (* 2. High release threshold (UINT64_MAX): free + sync keeps backing memory reserved until the
     threshold is lowered back to 0 and synchronized. *)
  Cu.Mem_pool.set_release_threshold default_pool U64.max_int;
  check "release threshold reads back UINT64_MAX"
    (U64.equal (pool_attr default_pool RELEASE_THRESHOLD) U64.max_int);
  check "current pool reflects updated release threshold"
    (U64.equal (pool_attr current_pool RELEASE_THRESHOLD) U64.max_int);

  let p2 = Cu.Stream.mem_alloc stream ~size_in_bytes:alloc_bytes in
  Cu.Stream.synchronize stream;
  Cu.Stream.mem_free stream p2;
  Cu.Stream.synchronize stream;
  check "after free+sync (threshold max): pool used is 0"
    (U64.equal (pool_attr default_pool USED_MEM_CURRENT) zero);
  check "after free+sync (threshold max): pool reserved retained"
    (U64.compare (pool_attr default_pool RESERVED_MEM_CURRENT) alloc_u64 >= 0);

  Cu.Mem_pool.set_release_threshold default_pool zero;
  check "release threshold restored to 0"
    (U64.equal (pool_attr default_pool RELEASE_THRESHOLD) zero);
  Cu.Stream.synchronize stream;
  check "after lowering threshold to 0 + sync: pool reserved released"
    (U64.equal (pool_attr default_pool RESERVED_MEM_CURRENT) zero);

  (* 3. Graph asynchronous allocator statistics. *)
  check "initial graph used current is 0" (U64.equal (graph_attr USED_MEM_CURRENT) zero);
  check "initial graph used high is 0" (U64.equal (graph_attr USED_MEM_HIGH) zero);
  check "initial graph reserved current is 0" (U64.equal (graph_attr RESERVED_MEM_CURRENT) zero);
  check "initial graph reserved high is 0" (U64.equal (graph_attr RESERVED_MEM_HIGH) zero);

  let capture_alloc bytes =
    Cu.Graph.begin_capture stream;
    let gp = Cu.Stream.mem_alloc stream ~size_in_bytes:bytes in
    Cu.Stream.memset_d8 gp (Unsigned.UChar.of_int 0x42) ~length:bytes stream;
    let graph = Cu.Graph.end_capture stream in
    let exec = Cu.Graph.instantiate graph in
    Cu.Graph.destroy graph;
    (gp, exec)
  in
  let gp1, exec1 = capture_alloc alloc_bytes in
  Cu.Graph.launch exec1 stream;
  Cu.Stream.synchronize stream;

  let g1_used = graph_attr USED_MEM_CURRENT in
  let g1_used_high = graph_attr USED_MEM_HIGH in
  let g1_res = graph_attr RESERVED_MEM_CURRENT in
  let g1_res_high = graph_attr RESERVED_MEM_HIGH in
  check "after first graph alloc launch: graph used covers allocation"
    (U64.compare g1_used alloc_u64 >= 0);
  check "after first graph alloc launch: graph used high >= used current"
    (U64.compare g1_used_high g1_used >= 0);
  check "after first graph alloc launch: graph reserved >= used" (U64.compare g1_res g1_used >= 0);
  check "after first graph alloc launch: graph reserved high >= reserved current"
    (U64.compare g1_res_high g1_res >= 0);

  (* A second live graph allocation larger than the driver's initial chunk forces both current and
     high-water readings to grow strictly above the first graph's footprint. *)
  let second_bytes = 40 * 1024 * 1024 in
  let total_u64 = U64.of_int (alloc_bytes + second_bytes) in
  let gp2, exec2 = capture_alloc second_bytes in
  Cu.Graph.launch exec2 stream;
  Cu.Stream.synchronize stream;

  let g2_used = graph_attr USED_MEM_CURRENT in
  let g2_used_high = graph_attr USED_MEM_HIGH in
  let g2_res = graph_attr RESERVED_MEM_CURRENT in
  let g2_res_high = graph_attr RESERVED_MEM_HIGH in
  check "after second graph alloc launch: graph used covers combined allocations"
    (U64.compare g2_used total_u64 >= 0);
  check "after second graph alloc launch: graph used grew" (U64.compare g2_used g1_used > 0);
  check "after second graph alloc launch: graph used high >= used current"
    (U64.compare g2_used_high g2_used >= 0);
  check "after second graph alloc launch: graph reserved >= used" (U64.compare g2_res g2_used >= 0);
  check "after second graph alloc launch: graph reserved high >= reserved current"
    (U64.compare g2_res_high g2_res >= 0);

  (* Free the graph allocations and destroy the executable graphs; high-water marks persist and
     the reserved >= used and current <= high invariants hold regardless of whether the driver
     retains backing chunks until cuDeviceGraphMemTrim. *)
  Cu.Stream.mem_free stream gp2;
  Cu.Stream.mem_free stream gp1;
  Cu.Stream.synchronize stream;
  Cu.Graph.exec_destroy exec2;
  Cu.Graph.exec_destroy exec1;
  Cu.Context.synchronize ();
  let g3_used = graph_attr USED_MEM_CURRENT in
  let g3_used_high = graph_attr USED_MEM_HIGH in
  let g3_res = graph_attr RESERVED_MEM_CURRENT in
  let g3_res_high = graph_attr RESERVED_MEM_HIGH in
  check "after destroying graphs: graph used high retained"
    (U64.compare g3_used_high g2_used_high >= 0);
  check "after destroying graphs: graph reserved high retained"
    (U64.compare g3_res_high g2_res_high >= 0);
  check "after destroying graphs: graph used <= used high" (U64.compare g3_used g3_used_high <= 0);
  check "after destroying graphs: graph reserved <= reserved high"
    (U64.compare g3_res g3_res_high <= 0);
  check "after destroying graphs: graph reserved >= used" (U64.compare g3_res g3_used >= 0);

  ignore (Sys.opaque_identity context);
  if !failures > 0 then exit 1
