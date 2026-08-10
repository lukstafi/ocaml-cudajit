module Cu = Cuda

(* [CUlimit]s are optional: a driver rejects the ones a device does not implement with
   [CUDA_ERROR_UNSUPPORTED_LIMIT] (e.g. the device-runtime limits on GPUs without device-side
   launch). [result] is abstract, so recognize that status through its sexp. *)
let is_unsupported_limit status =
  match Cu.sexp_of_result status with
  | Sexplib0.Sexp.Atom "CUDA_ERROR_UNSUPPORTED_LIMIT" -> true
  | _ -> false

let limits =
  [
    Cu.Context.STACK_SIZE;
    PRINTF_FIFO_SIZE;
    MALLOC_HEAP_SIZE;
    DEV_RUNTIME_SYNC_DEPTH;
    DEV_RUNTIME_PENDING_LAUNCH_COUNT;
    MAX_L2_FETCH_GRANULARITY;
    PERSISTING_L2_CACHE_SIZE;
  ]

(* [None] for a limit this device does not support; other CUDA errors still propagate. *)
let query_limit li =
  match Cu.Context.get_limit li with
  | value -> Some value
  | exception Cu.Cuda_error { status; _ } when is_unsupported_limit status -> None

let sexp_of_limits queried =
  Sexplib0.Sexp_conv.(
    sexp_of_list (fun (li, value) ->
        sexp_of_pair Cu.Context.sexp_of_limit
          (function Some v -> sexp_of_int v | None -> Sexplib0.Sexp.Atom "UNSUPPORTED")
          (li, value)))
    queried

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
      Format.printf "Attributes:@ %a\n%!" Sexplib0.Sexp.pp_hum (Cu.Device.sexp_of_attributes props);
      let queried = List.map (fun li -> (li, query_limit li)) limits in
      Format.printf "Default limits:@ %a\n%!" Sexplib0.Sexp.pp_hum (sexp_of_limits queried);
      match List.filter (fun (_, value) -> value = None) queried with
      | [] -> ()
      | unsupported ->
          Format.printf "Limits unsupported by this device's driver:@ %a\n%!" Sexplib0.Sexp.pp_hum
            (Sexplib0.Sexp_conv.sexp_of_list Cu.Context.sexp_of_limit @@ List.map fst unsupported))
    gpus
