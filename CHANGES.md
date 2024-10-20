## [0.5.1] current

### Added

- `get_free_and_total_mem`.
- Multiple missing `sexp_of` conversions.
- `cuda_call_hook` to help in debugging.
- `is_success` functions.

## [0.5.0] 2024-09-25

### Added

- CUDA events.
- Delimited events: they are owned by a stream they record, and are automatically destroyed after synchronization.

### Changed

- Partitioned the API into modules.
- Removed `destroy` functions from the interface, attaching them as finalizers.

### Fixed

- Fixed broken types for `can_access_peer` and `get_p2p_attributes`.

## [0.4.1] 2024-09-12

### Fixed

- Pass the $CUDA_PATH/include path to the nvrtc compiler; otherwise it will not `#include` anything.
- Work around `Ctypes.bigarray_start` and `typ_of_bigarray_kind` because `ctypes` does not support half precision.

## [0.4.0] 2024-07-21

### Added

- Previously commented out parts, that require a newer version of the CUDA API.
- Interface file `cudajit.mli` with documentation.
- Expose context limits. Print default limits in `bin/properties`.
- `sexp_of_kernel_param`

### Changed

- Dropped `JIT_` prefix for `jit_option` values.
- Self-contained types in the interface, with some corrections and renaming.
- Formatting: line length 100.

### Fixed

- A major bug, exacerbated by the asynchronous functionaliy of v0.3 -- functions performing asynchronous calls should keep the call arguments alive; the user should only forget (or free) the arguments after the calls complete (e.g. after synchronizing a stream).
  - Only `launch_kernel` needed fixing as I don't think other async functions allocate passed arguments.
  - We hanlde this internally so no API change!

## [0.3.0] 2024-07-05

### Added

- Support for streams (except `cuStreamWaitEvent` and graph capture).
- Support for asynchronous copying, including `cuMemcpyPeerAsync`.

### Changed

- Renamed `byte_size` to `size_in_bytes`.

## [0.2.0] 2024-05-18

### Added

- Support for peer-to-peer device-to-device copying.
- Support for context flags.

### Changed

- `ctx_create` properly handles context flags.

## [0.1.1] 2024-05-09

### Added

- Continuous Integration on GitHub thanks to GitHub action Jimver/cuda-toolkit, but only PTX compilation.

### Fixed

- Test target should erase compiler versions.

## [0.1.0] 2023-10-28

### Added

- Initial stand-alone release. For earlier changes, see e.g. [ocannl/cudajit @ 2 months ago](https://github.com/lukstafi/ocannl/tree/560ad1caeefe0bdfd85d0393a29a4721d11ee742/cudajit)

### Fixed

- To be defensive, pass `-I` and `-L` arguments to the compiler and linker with the default paths on linux-like systems.
