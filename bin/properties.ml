module Cu = Cudajit

let () =
  Cu.init ();
  let num_gpus = Cu.device_get_count () in
  Format.printf "\n# GPUs: %d\n%!" num_gpus;
  let gpus = List.init num_gpus (fun ordinal -> Cu.device_get ~ordinal) in
  List.iteri
    (fun ordinal dev ->
      let props = Cu.device_get_attributes dev in
      Cu.ctx_set_current @@ Cu.device_primary_ctx_retain dev;
      let ctx_flags = Cu.ctx_get_flags () in
      Format.printf "GPU #%d:@ %a@\nContext properties:@ %a@\n%!" ordinal Sexplib0.Sexp.pp_hum
        (Cu.sexp_of_device_attributes props)
        Sexplib0.Sexp.pp_hum (Cu.sexp_of_ctx_flags ctx_flags);
      Format.printf "Default limits:@ %a\n%%!" Sexplib0.Sexp.pp_hum
        (Sexplib0.Sexp_conv.(
           sexp_of_list (fun li ->
               sexp_of_pair Cu.sexp_of_limit sexp_of_int (li, Cu.ctx_get_limit li)))
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
