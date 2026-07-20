# CUDA to SYCL migration task

Translate `main.cu` into a semantically equivalent SYCL 2020 program and save
the final source as `main.sycl.cpp` in this directory. The input is CUDA only;
no pre-generated `.dp.cpp` or SYCLomatic output is provided.

You may read `main.cu`, `CMakeLists.txt`, and this task. You may use
`sycl_build.sh` and `sycl_run.sh` to compile and run your translation. The
benchmark verifier, metadata, tolerances, and reference implementation are not
available in this workspace and must not be searched for or reconstructed from
the repository.

Requirements:

- Preserve deterministic input generation, dimensions, seeds, and algorithmic
  semantics from `main.cu`.
- Accept an optional output path as `argv[1]`.
- Write the same whitespace-separated numeric result sequence as the CUDA
  program.
- Do not embed a captured reference output or replace the computation with
  constants.
- Keep all required code in `main.sycl.cpp` unless the build wrapper explicitly
  permits additional files.

Declare completion only after the final source has been written.
