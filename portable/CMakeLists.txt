# The portable pipeline. kernels/ stays third-party-free so its element functions compile for a
# device; host code may use OpenCV, Eigen and Boost (ADR 0006).
set(CCTAG_PORTABLE_BACKEND "cpu" CACHE STRING "Execution backend of the portable pipeline: cpu or stub")
set_property(CACHE CCTAG_PORTABLE_BACKEND PROPERTY STRINGS cpu stub)
if(NOT CCTAG_PORTABLE_BACKEND MATCHES "^(cpu|stub)$")
  message(FATAL_ERROR "CCTAG_PORTABLE_BACKEND must be 'cpu' or 'stub', got '${CCTAG_PORTABLE_BACKEND}'")
endif()
option(CCTAG_PORTABLE_PARALLEL "Compile the portable pipeline's kernels with OpenMP pragmas" ON)

set(PORTABLE_SHARED_SOURCES
    ${PROJECT_SOURCE_DIR}/src/cctag/CCTagMarkersBank.cpp
    ${PROJECT_SOURCE_DIR}/src/cctag/Params.cpp
    ${PROJECT_SOURCE_DIR}/src/cctag/geometry/Ellipse.cpp)

set(PORTABLE_SOURCES
    src/ICCTag.cpp
    src/backends/cpu/buffers.cpp
    src/backends/cpu/pyramid.cpp
    src/backends/cpu/gradient.cpp
    src/backends/stub/backend.cpp)

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

set(CCTAG_PORTABLE_OPENMP OFF)
if(CCTAG_PORTABLE_PARALLEL)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    set(CCTAG_PORTABLE_OPENMP ON)
  else()
    message(STATUS "CCTag: OpenMP not found, the portable pipeline's kernels run sequentially")
  endif()
endif()

# Consumed by cmake/Config.cmake.in so that find_package(CCTag) links the OpenMP runtime
set(CCTAG_PORTABLE_OPENMP ${CCTAG_PORTABLE_OPENMP} PARENT_SCOPE)

# Common settings for every portable target, library and tests alike
function(cctag_portable_configure target)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  target_compile_options(${target} PRIVATE ${PORTABLE_EXACT_FLOAT_OPTIONS} ${PORTABLE_WARNING_OPTIONS})
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  if(CCTAG_PORTABLE_OPENMP)
    target_compile_definitions(${target} PRIVATE CCTAG_PORTABLE_PARALLEL=1)
  endif()
endfunction()

cctag_portable_configure(CCTag)
string(TOUPPER "${CCTAG_PORTABLE_BACKEND}" PORTABLE_BACKEND_UPPER)
target_compile_definitions(CCTag PRIVATE CCTAG_PORTABLE_BACKEND_${PORTABLE_BACKEND_UPPER}=1)
if(CCTAG_PORTABLE_OPENMP)
  target_link_libraries(CCTag PRIVATE OpenMP::OpenMP_CXX)
endif()

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
  add_executable(cctag_portable_smoke test/smoke.cpp)
  target_link_libraries(cctag_portable_smoke PRIVATE CCTag)
  add_test(NAME portable_pipeline_returns_no_detection_candidates COMMAND cctag_portable_smoke)

  # One Boost test executable per test/<name>.cpp, named portable_<name>, each suite is a ctest
  function(cctag_portable_add_test name)
    add_boost_test(SOURCE test/${name}.cpp LINK ${ARGN} PREFIX portable)
    cctag_portable_configure(portable_${name})
    if(CCTAG_PORTABLE_OPENMP)
      target_link_libraries(portable_${name} OpenMP::OpenMP_CXX)
    endif()
  endfunction()

  cctag_portable_add_test(kernels CCTag)
  cctag_portable_add_test(seam CCTag)

  # Stage-isolation tests (one per stage function), they skip unless $CCTAG_REFERENCE_SNAPSHOTS is set
  add_library(cctag_portable_test_support STATIC test/reference_snapshot.cpp)
  target_include_directories(cctag_portable_test_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_link_libraries(cctag_portable_test_support PUBLIC CCTag)
  cctag_portable_configure(cctag_portable_test_support)
  cctag_portable_add_test(stage_isolation cctag_portable_test_support)

  # The reference snapshots are produced with OpenCV's SIMD paths masked (the scalar results are the
  # ones the fork reproduces bit for bit); every portable test runs under the same mask.
  set(CCTAG_PORTABLE_OPENCV_CPU_DISABLE "AVX512_SKX,AVX2,FP16,AVX,SSE4_2,SSE4_1"
      CACHE STRING "OPENCV_CPU_DISABLE for the portable pipeline's tests")
  get_property(portable_tests DIRECTORY PROPERTY TESTS)
  set_tests_properties(${portable_tests} PROPERTIES
      ENVIRONMENT "OPENCV_CPU_DISABLE=${CCTAG_PORTABLE_OPENCV_CPU_DISABLE}")
endif()
