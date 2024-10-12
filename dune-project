(lang dune 3.11)

(using ctypes 0.3)

(name cudajit)

(version 0.5.1)

(generate_opam_files true)

(source
 (github lukstafi/ocaml-cudajit))

(authors "Lukasz Stafiniak")

(maintainers "Lukasz Stafiniak <lukstafi@gmail.com>")

(license MIT)

(documentation
 https://github.com/lukstafi/ocaml-cudajit/blob/master/README.md)

(package
 (name cudajit)
 (synopsis
  "Bindings to the `cuda` and `nvrtc` libraries with a unified interface")
 (description
  "Bindings to manually selected parts of `lcuda` and `lnvrtc`, with a few types and conversion functions to facilitate use.")
 (depends
  (ocaml
   (>= 4.12))
  dune
  (ctypes
   (>= 0.14.0))
  ctypes-foreign
  sexplib0
  ppx_sexp_conv
  ppx_expect
  conf-cuda)
 (tags
  (cuda jit nvrtc)))
