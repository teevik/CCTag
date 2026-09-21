# SYCL stub execution backend

This scaffold implements the `ExecutionBackend` interface and is selected by
`CCTAG_PIPELINE=portable` and `CCTAG_PORTABLE_BACKEND=sycl`. The regular `cctagDetection`
entry point drives it through the existing Context and host sequence.

All eight detection stages are empty and every image returns no detection candidates.
`load` runs a synchronous SYCL kernel that writes `42` and checks the result, proving that
the selected execution backend launches device code. It prints the platform, device and a
stub warning. There is no CPU stage delegation or detection algorithm yet; that work remains
in [issue #130](https://github.com/teevik/master-thesis/issues/130).

Stage storage and `wait` are empty: the smoke kernel completes inside `load`, and no other
stage submits work. Calls without a probe and with a timing-only probe work. Stage snapshot
requests throw `std::logic_error` because no intermediate outputs exist. Stub timings are
not detection performance measurements. `CCTAG_SYCL_DEVICE=cpu|gpu` selects the platform for
public API calls, defaulting to `cpu`; requesting a missing platform fails without fallback.

## Meeting demo with Nix

From the parent `master-thesis` repository:

```sh
nix run '.?submodules=1#portable-sycl-smoke' -- cpu
LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  nix run '.?submodules=1#portable-sycl-smoke' -- gpu
```

The demo uses the public detection API with a nonempty 64×64 image, prints and checks all
eight stage events, checks the empty result, then repeats detection without a probe on the
same context. It defaults to `cpu`; `--help` prints usage.

The package pins AdaptiveCpp 25.10.0, LLVM 20 and the `generic` compilation flow using the
repository's locked nixpkgs. It builds a static CCTag library with the SYCL stub selected;
only `backend.cpp` uses SYCL compilation. CPU execution is checked during the build. GPU
execution needs the host's NVIDIA driver; the command above exposes its NixOS library
directory. AdaptiveCpp may print buffer-model and first-run JIT notices.

## CMake build

With CCTag's dependencies, AdaptiveCpp 25.10 and its matching Clang toolchain available,
from the CCTag repository:

```sh
cmake -S . -B build-sycl-stub \
  -DCMAKE_CXX_COMPILER=clang++ -DACPP_TARGETS=generic \
  -DCCTAG_PIPELINE=portable -DCCTAG_PORTABLE_BACKEND=sycl \
  -DCCTAG_BUILD_APPS=OFF -DCCTAG_WITH_CUDA=OFF -DCCTAG_BUILD_TESTS=ON
cmake --build build-sycl-stub
ctest --test-dir build-sycl-stub --output-on-failure
./build-sycl-stub/Linux-x86_64/portable_sycl_smoke gpu
```

The executable's platform directory follows CCTag's usual CMake convention. Set
`AdaptiveCpp_DIR` to its `lib/cmake/AdaptiveCpp` directory if CMake cannot find the package.
The CPU backend keeps its existing tests; the SYCL build runs the public-API smoke check
instead of claiming detector correctness for the empty stages.
