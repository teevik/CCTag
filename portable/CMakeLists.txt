cmake_minimum_required(VERSION 3.20)

set(CCTAG_PORTABLE_BACKEND "cpu" CACHE STRING "Execution backend of the portable pipeline: cpu or sycl (stub)")
set_property(CACHE CCTAG_PORTABLE_BACKEND PROPERTY STRINGS cpu sycl)
if(NOT CCTAG_PORTABLE_BACKEND MATCHES "^(cpu|sycl)$")
  message(FATAL_ERROR "CCTAG_PORTABLE_BACKEND must be 'cpu' or 'sycl', got '${CCTAG_PORTABLE_BACKEND}'")
endif()

# Separate sanitizer names with commas, e.g. address,undefined
set(CCTAG_PORTABLE_SANITIZERS "" CACHE STRING "Sanitizers for the portable pipeline and its tests")

add_library(CCTag
    ${PROJECT_SOURCE_DIR}/src/cctag/CCTagMarkersBank.cpp
    ${PROJECT_SOURCE_DIR}/src/cctag/Params.cpp
    ${PROJECT_SOURCE_DIR}/src/cctag/geometry/Ellipse.cpp
    src/ICCTag.cpp
    src/host/ellipse.cpp
    src/host/markers.cpp)

if(CCTAG_PORTABLE_BACKEND STREQUAL "cpu")
  target_sources(CCTag PRIVATE
    src/backends/cpu/buffers.cpp
    src/backends/cpu/pyramid.cpp
    src/backends/cpu/gradient.cpp
    src/backends/cpu/edges.cpp
    src/backends/cpu/edge_points.cpp
    src/backends/cpu/vote.cpp
    src/backends/cpu/linking.cpp
    src/backends/cpu/candidates.cpp
    src/backends/cpu/markers.cpp)
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

function(cctag_portable_configure target)
  target_compile_features(${target} PRIVATE cxx_std_20)
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
  if(NOT MSVC)
    # Disable implicit FMA contraction so rounding does not depend on the compiler.
    # Kernels call std::fma where they need a fused result.
    target_compile_options(${target} PRIVATE -ffp-contract=off -Wmisleading-indentation)
  endif()
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/src)
  if(CCTAG_PORTABLE_SANITIZERS)
    # Fail tests on undefined behaviour
    target_compile_options(${target} PRIVATE
        -fsanitize=${CCTAG_PORTABLE_SANITIZERS} -fno-sanitize-recover=all -fno-omit-frame-pointer)
    # Link the sanitizer runtime for static library consumers too
    target_link_options(${target} PUBLIC -fsanitize=${CCTAG_PORTABLE_SANITIZERS})
  endif()
endfunction()

cctag_portable_configure(CCTag)
string(TOUPPER "${CCTAG_PORTABLE_BACKEND}" PORTABLE_BACKEND_UPPER)
target_compile_definitions(CCTag PRIVATE CCTAG_PORTABLE_BACKEND_${PORTABLE_BACKEND_UPPER}=1)

target_include_directories(CCTag
    PUBLIC
      "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
      "$<BUILD_INTERFACE:${generated_dir}>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

target_link_libraries(CCTag
    PUBLIC
      opencv_core
      Eigen3::Eigen
      Boost::filesystem
      Boost::serialization
    PRIVATE
      opencv_imgproc
      OpenMP::OpenMP_CXX
      Threads::Threads)

if(CCTAG_PORTABLE_BACKEND STREQUAL "sycl")
  message(WARNING "The SYCL execution backend is a stub: all detection stages are empty.")
  add_subdirectory(src/backends/sycl)
endif()

# These suites validate the implemented CPU stages, not the empty SYCL scaffold.
if(CCTAG_BUILD_TESTS AND CCTAG_PORTABLE_BACKEND STREQUAL "cpu")
  find_package(ut 2.3.1 REQUIRED CONFIG)

  # Match the scalar OpenCV path used to produce the reference snapshots
  set(CCTAG_PORTABLE_OPENCV_CPU_DISABLE "AVX512_SKX,AVX2,FP16,AVX,SSE4_2,SSE4_1"
      CACHE STRING "OPENCV_CPU_DISABLE for the portable pipeline's tests")

  function(cctag_portable_add_test name)
    set(target portable_${name})
    add_executable(${target} ${ARGN})
    cctag_portable_configure(${target})
    target_link_libraries(${target} PRIVATE CCTag Boost::ut opencv_imgproc OpenMP::OpenMP_CXX)
    add_test(NAME ${target} COMMAND ${target})
    # Let OpenMP threads sleep between tasks to avoid spinning on tiny test images
    set_tests_properties(${target} PROPERTIES
        ENVIRONMENT "OPENCV_CPU_DISABLE=${CCTAG_PORTABLE_OPENCV_CPU_DISABLE};OMP_WAIT_POLICY=passive")
  endfunction()

  # Headers with embedded unit tests. Compiling them also gives clangd a compile command.
  set(unit_test_headers
      src/kernels/frontend.hpp
      src/kernels/pcg32.hpp
      src/host/candidates.hpp
      src/host/context.hpp
      src/host/detect.hpp)
  set_source_files_properties(${unit_test_headers} PROPERTIES LANGUAGE CXX)
  cctag_portable_add_test(unit
      ${unit_test_headers}
      src/host/ellipse.cpp
      src/backends/cpu/pyramid.cpp
      src/backends/cpu/gradient.cpp
      src/backends/cpu/edges.cpp
      src/backends/cpu/edge_points.cpp
      src/backends/cpu/vote.cpp
      src/backends/cpu/linking.cpp
      src/backends/cpu/markers.cpp
      test/main.cpp
      test/detection_api.cpp
      test/support/reference_snapshot.cpp)
  target_compile_definitions(portable_unit PRIVATE CCTAG_TEST)

  # Contracts that need retained CUDA captures from CCTAG_REFERENCE_SNAPSHOTS. A separate
  # test lets CTest skip it (exit code 77) when the captures are absent, without skipping
  # the unit suite
  cctag_portable_add_test(snapshot_contracts
      test/snapshot_contracts.cpp
      test/support/reference_snapshot.cpp)
  set_tests_properties(portable_snapshot_contracts PROPERTIES SKIP_RETURN_CODE 77)

  # Writes isolated-stage and full-pipeline snapshots for one CUDA reference capture.
  # The thesis repository's snapshot comparator checks them, this target does not.
  add_executable(portable_cuda_isolation test/stage_isolation.cpp test/support/reference_snapshot.cpp)
  cctag_portable_configure(portable_cuda_isolation)
  target_link_libraries(portable_cuda_isolation PRIVATE CCTag opencv_imgproc OpenMP::OpenMP_CXX)
  install(TARGETS portable_cuda_isolation RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
