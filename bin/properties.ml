module Cu = Cuda

let () =
  Cu.init ();
  let num_gpus = Cu.Device.get_count () in
  Format.printf "\n# GPUs: %d\n%!" num_gpus;
  let gpus = List.init num_gpus (fun ordinal -> Cu.Device.get ~ordinal) in
  List.iteri
    (fun ordinal dev ->
      let props = Cu.Device.get_attributes dev in
      Cu.Context.set_current @@ Cu.Context.get_primary dev;
      let ctx_flags = Cu.Context.get_flags () in
      let free, total = Cu.Device.get_free_and_total_mem () in
      Format.printf "GPU #%d:@ Free mem: %d,@ total mem: %d,@ context properties:@ %a@\n%!" ordinal
        free total Sexplib0.Sexp.pp_hum
        (Cu.Context.sexp_of_flags ctx_flags);
      Format.printf "Attributes:@ %a\n%%!" Sexplib0.Sexp.pp_hum (Cu.Device.sexp_of_attributes props);
      Format.printf "Default limits:@ %a\n%%!" Sexplib0.Sexp.pp_hum
        (Sexplib0.Sexp_conv.(
           sexp_of_list (fun li ->
               sexp_of_pair Cu.Context.sexp_of_limit sexp_of_int (li, Cu.Context.get_limit li)))
           [
             STACK_SIZE;
             PRINTF_FIFO_SIZE;
             MALLOC_HEAP_SIZE;
             DEV_RUNTIME_SYNC_DEPTH;
             DEV_RUNTIME_PENDING_LAUNCH_COUNT;
             MAX_L2_FETCH_GRANULARITY;
             PERSISTING_L2_CACHE_SIZE;
           ]))
    gpus
