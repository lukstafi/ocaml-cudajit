## [0.4.0] 2024-07-20

### Added

- Previously commented out parts, that require a newer version of the CUDA API.
- TODO: Interface file `cudajit.mli`.

### Fixed

- A major bug, exacerbated by the asynchronous functionaliy of v0.3 -- functions performing asynchronous calls should keep the call arguments alive; the user should only forget (or free) the arguments after the calls complete (e.g. after synchronizing a stream).
  - Only `launch_kernel` needed fixing as I don't think other functions allocate passed arguments.
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
