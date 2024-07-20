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
        Sexplib0.Sexp.pp_hum (Cu.sexp_of_ctx_flags ctx_flags))
    gpus
