# Disposable SYCL gradient experiment

Evidence for [Prototype: validate the SYCL gradient and CPU delegation on required platforms](https://github.com/teevik/master-thesis/issues/97).
This branch is throwaway decision evidence. It is not the production execution backend.

The experiment is gated first by compiling and executing the existing shared gradient
element function with AdaptiveCpp 25.10.0 and LLVM 20.1.8. The accompanying umbrella
branch `prototype/sycl-gradient` carries the pinned Nix recipe and run evidence.

The full experiment will exercise one persistent execution state, per-level CPU staging
and explicit device USM, typed borrowed whole-image inputs, and one shared host-state
owner. The ordinary host-call schedule is the control for the asynchronous comparison.

Current preflight covers a 37×23 source with row stride 43 and independently specified
nearest-even rounding examples. Comparing the shared function with itself on the host
is only a compiler/execution check; reference-fed isolation remains required.
