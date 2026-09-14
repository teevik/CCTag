# The portable pipeline: the detection stages rewritten behind a compile-time execution backend.
# See README.md in this directory. Production kernels/ code uses only the standard library so its
# element functions compile for a device too. Host code and guarded tests use OpenCV, Eigen and
# Boost as the legacy pipeline does.
cmake_minimum_required(VERSION 3.20)

# The execution backend `cctagDetection` is compiled against. Only `cpu` exists today; a device
# backend adds its name here and its sources below.
set(CCTAG_PORTABLE_BACKEND "cpu" CACHE STRING "Execution backend of the portable pipeline: cpu")
set_property(CACHE CCTAG_PORTABLE_BACKEND PROPERTY STRINGS cpu)
if(NOT CCTAG_PORTABLE_BACKEND MATCHES "^(cpu)$")
  message(FATAL_ERROR "CCTAG_PORTABLE_BACKEND must be 'cpu', got '${CCTAG_PORTABLE_BACKEND}'")
endif()
# Comma-separated -fsanitize= list (e.g. "address,undefined"), applied to the library and its tests
set(CCTAG_PORTABLE_SANITIZERS "" CACHE STRING "Sanitizers for the portable pipeline and its tests")

set(PORTABLE_SHARED_SOURCES
    ${PROJECT_SOURCE_DIR}/src/cctag/CCTagMarkersBank.cpp
    ${PROJECT_SOURCE_DIR}/src/cctag/Params.cpp
    ${PROJECT_SOURCE_DIR}/src/cctag/geometry/Ellipse.cpp)

set(PORTABLE_SOURCES
    src/ICCTag.cpp
    src/backends/cpu/buffers.cpp
    src/backends/cpu/pyramid.cpp
    src/backends/cpu/gradient.cpp)

add_library(CCTag
    ${PORTABLE_SHARED_SOURCES}
    ${PORTABLE_SOURCES})

# No fused multiply-add to allow for bit-exact comparisons
set(PORTABLE_EXACT_FLOAT_OPTIONS "")
if(NOT MSVC)
  set(PORTABLE_EXACT_FLOAT_OPTIONS -ffp-contract=off)
endif()

# Warn on misleading indentation
set(PORTABLE_WARNING_OPTIONS "")
if(NOT MSVC)
  set(PORTABLE_WARNING_OPTIONS -Wmisleading-indentation)
endif()

# The CPU backend's kernels are `#pragma omp parallel for` loops; OMP_NUM_THREADS=1 is the sequential
# run of the same binary. cmake/Config.cmake.in re-finds OpenMP for consumers of the installed package.
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Common settings for every portable target, library and tests alike
function(cctag_portable_configure target)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  target_compile_options(${target} PRIVATE ${PORTABLE_EXACT_FLOAT_OPTIONS} ${PORTABLE_WARNING_OPTIONS})
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/src)
  if(CCTAG_PORTABLE_SANITIZERS)
    # Undefined behaviour aborts the test instead of printing and returning success
    target_compile_options(${target} PRIVATE
        -fsanitize=${CCTAG_PORTABLE_SANITIZERS} -fno-sanitize-recover=all -fno-omit-frame-pointer)
    # PUBLIC so that everything linking the library links the sanitizer runtime too
    target_link_options(${target} PUBLIC -fsanitize=${CCTAG_PORTABLE_SANITIZERS})
  endif()
endfunction()

cctag_portable_configure(CCTag)
string(TOUPPER "${CCTAG_PORTABLE_BACKEND}" PORTABLE_BACKEND_UPPER)
target_compile_definitions(CCTag PRIVATE CCTAG_PORTABLE_BACKEND_${PORTABLE_BACKEND_UPPER}=1)
target_link_libraries(CCTag PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(CCTag
    PUBLIC
      "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
      "$<BUILD_INTERFACE:${generated_dir}>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

target_link_libraries(CCTag
    PUBLIC
      ${OpenCV_LIBS}
      Eigen3::Eigen
      Boost::date_time
      Boost::filesystem
      Boost::serialization
      Boost::timer)

if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  target_link_libraries(CCTag PRIVATE pthread dl)
endif()

if(CCTAG_BUILD_TESTS)
  function(cctag_portable_add_test name source)
    set(target portable_${name})
    set_source_files_properties("${source}" PROPERTIES LANGUAGE CXX)
    add_executable(${target} test/main.cpp "${source}")
    cctag_portable_configure(${target})
    target_compile_definitions(${target} PRIVATE BOOST_TEST_DYN_LINK ${ARGN})
    target_link_libraries(${target} PRIVATE CCTag Boost::unit_test_framework OpenMP::OpenMP_CXX)
    add_test(NAME ${target} COMMAND ${target} --catch_system_error=yes)
  endfunction()

  cctag_portable_add_test(detection_api test/detection_api.cpp)
  cctag_portable_add_test(gradient_element src/kernels/gradient.hpp
      CCTAG_TEST_KERNELS_GRADIENT)
  cctag_portable_add_test(context src/host/context.hpp CCTAG_TEST_HOST_CONTEXT)
  cctag_portable_add_test(host_sequence src/host/detect.hpp
      CCTAG_TEST_HOST_SEQUENCE)

  # Stage-isolation tests (one per stage function); they skip unless $CCTAG_REFERENCE_SNAPSHOTS is set
  add_library(cctag_portable_test_support STATIC test/support/reference_snapshot.cpp)
  target_include_directories(cctag_portable_test_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_link_libraries(cctag_portable_test_support PUBLIC CCTag)
  cctag_portable_configure(cctag_portable_test_support)
  # Compile the reader directly into its own tests, so it is defined once in that executable.
  cctag_portable_add_test(snapshot_support test/support/reference_snapshot.cpp
      CCTAG_TEST_REFERENCE_SNAPSHOT)
  cctag_portable_add_test(stage_isolation test/stage_isolation.cpp)
  target_link_libraries(portable_stage_isolation PRIVATE cctag_portable_test_support)

  # OpenCV's SIMD paths sum in a different order from its scalar path and can differ in the last bit.
  # The reference snapshots are produced with them masked, the element functions reproduce the scalar
  # results, so every portable test runs under the same mask.
  set(CCTAG_PORTABLE_OPENCV_CPU_DISABLE "AVX512_SKX,AVX2,FP16,AVX,SSE4_2,SSE4_1"
      CACHE STRING "OPENCV_CPU_DISABLE for the portable pipeline's tests")
  # OMP_WAIT_POLICY: the tests run thousands of OpenMP regions on tiny images; with the default
  # spin-wait on a loaded machine that turns 40 ms into 15 s.
  get_property(portable_tests DIRECTORY PROPERTY TESTS)
  set_tests_properties(${portable_tests} PROPERTIES
      ENVIRONMENT "OPENCV_CPU_DISABLE=${CCTAG_PORTABLE_OPENCV_CPU_DISABLE};OMP_WAIT_POLICY=passive")
  # Without the snapshot store the isolation cases skip; make ctest report that as "Not Run" rather
  # than as a pass. Boost.Test only logs the skip at the test_suite level.
  set_property(TEST portable_stage_isolation APPEND PROPERTY
      ENVIRONMENT "BOOST_TEST_LOG_LEVEL=test_suite")
  set_tests_properties(portable_stage_isolation PROPERTIES
      SKIP_REGULAR_EXPRESSION "is skipped because precondition failed")
endif()
