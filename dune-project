(lang dune 3.11)

(using ctypes 0.3)

(name cudajit)

(version 0.1.1)

(generate_opam_files true)

(source
 (github lukstafi/ocaml-cudajit))

(authors "Lukasz Stafiniak")

(maintainers "Lukasz Stafiniak")

(license LICENSE)

(documentation
 https://github.com/lukstafi/ocaml-cudajit/blob/master/README.md)

(package
 (name cudajit)
 (synopsis
  "Bindings to the `cuda` and `nvrtc` libraries with a unified interface")
 (description
  "Bindings to manually selected parts of `lcuda` and `lnvrtc`, with a few types and conversion functions to facilitate use.")
 ;  TODO: prepare conf-cuda-toolkit according to the instructions:
 ; https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu
 ; https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-wsl
 ; https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-debian
 ; and all the others: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#overview
 (depends
  ocaml
  dune
  ctypes
  ctypes-foreign
  ; conf-cuda-toolkit
  sexplib0
  ppx_sexp_conv
  ppx_expect
  )
 (tags
  (cuda jit nvrtc)))
