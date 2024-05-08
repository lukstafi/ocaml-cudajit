## [0.1.1] 2024-05-08

### Added

- Continuous Integration on GitHub thanks to GitHub action Jimver/cuda-toolkit.

### Fixed

- Test target should erase compiler versions.

## [0.1.0] 2023-10-28

### Added

- Initial stand-alone release. For earlier changes, see e.g. [ocannl/cudajit @ 2 months ago](https://github.com/lukstafi/ocannl/tree/560ad1caeefe0bdfd85d0393a29a4721d11ee742/cudajit)

### Fixed

- To be defensive, pass `-I` and `-L` arguments to the compiler and linker with the default paths on linux-like systems.
