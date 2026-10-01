# String_Vext v0.1.0

Initial source-available, noncommercial release by qiuyong.

## Included

- GPU String path solver with full triclinic Lennard-Jones image handling.
- Exact CPU Vext generator with 60^3 spatial cells and 64 orientations by default.
- Python command-line entry points, native source, build scripts, and checksums.
- COF validation case under `tests/cof/`, not a restriction on framework type.

## Validation

The release includes a synthetic nonorthogonal-cell test and COF regression
reports: 50 full-grid materials (100 gas fields), 300 same-geometry String
path comparisons, and a six-path BODY-X GPU pilot. The largest accessible
Vext orientation difference after float32 storage was 0.000122 K. Two sample
input-to-NPZ timings were 25.3x and 23.1x faster than the old CPU DIRECT path.

## Compatibility and scope

The attached wheel bundles reference binaries for Linux x86-64. The String
binary targets `sm_89`, needs a compatible CUDA runtime/cuFFT, and requires
glibc >=2.38. Rebuild from source for other GPUs or older systems. The CPU
Vext reference binary is also included. GitHub Actions packages the
reference binaries; it does not compile CUDA String or run GPU tests. Vext
currently supports neutral,
symmetric two-site LJ guests; it is not a general charged/multisite model.

Only three small `input.dat` examples are included. No production CIF,
input, trajectory, Vext, or diffusion dataset is part of the release.

License: PolyForm Noncommercial 1.0.0. Commercial use is not permitted without
separate written permission from qiuyong.
