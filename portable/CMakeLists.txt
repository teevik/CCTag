cmake_minimum_required(VERSION 3.20)

set(CCTAG_PORTABLE_BACKEND "cpu" CACHE STRING "Execution backend of the portable pipeline: cpu")
set_property(CACHE CCTAG_PORTABLE_BACKEND PROPERTY STRINGS cpu)
if(NOT CCTAG_PORTABLE_BACKEND STREQUAL "cpu")
  message(FATAL_ERROR "CCTAG_PORTABLE_BACKEND must be 'cpu', got '${CCTAG_PORTABLE_BACKEND}'")
endif()

# Separate sanitizer names with commas, e.g. address,undefined
set(CCTAG_PORTABLE_SANITIZERS "" CACHE STRING "Sanitizers for the portable pipeline and its tests")

add_library(CCTag
    ${PROJECT_SOURCE_DIR}/src/cctag/CCTagMarkersBank.cpp
    ${PROJECT_SOURCE_DIR}/src/cctag/Params.cpp
    ${PROJECT_SOURCE_DIR}/src/cctag/geometry/Ellipse.cpp
    src/ICCTag.cpp
    src/backends/cpu/buffers.cpp
    src/backends/cpu/pyramid.cpp
    src/backends/cpu/gradient.cpp
    src/backends/cpu/edges.cpp)

find_package(OpenMP REQUIRED COMPONENTS CXX)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

function(cctag_portable_configure target)
  target_compile_features(${target} PRIVATE cxx_std_20)
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
  if(NOT MSVC)
    # Disable fused multiply-add to preserve exact reference snapshot comparisons
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

if(CCTAG_BUILD_TESTS)
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

  # Give clangd a compile command for each header
  set(unit_test_headers
      src/kernels/canny.hpp
      src/kernels/gradient.hpp
      src/kernels/thinning.hpp
      src/host/context.hpp
      src/host/detect.hpp)
  set_source_files_properties(${unit_test_headers} PROPERTIES LANGUAGE CXX)
  cctag_portable_add_test(unit
      ${unit_test_headers}
      src/backends/cpu/edges.cpp
      test/main.cpp
      test/detection_api.cpp
      test/support/reference_snapshot.cpp)
  target_compile_definitions(portable_unit PRIVATE CCTAG_TEST)

  # Keep stage-isolation tests separate so an unset snapshot directory only skips those tests
  cctag_portable_add_test(stage_isolation
      test/stage_isolation.cpp
      test/support/reference_snapshot.cpp)
  set_tests_properties(portable_stage_isolation PROPERTIES SKIP_RETURN_CODE 77)
endif()
