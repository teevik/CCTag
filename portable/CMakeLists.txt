# The portable pipeline: a ground-up implementation in src/ with one execution backend selected at
# configure time. Layout (ADR 0001): kernels/ (device-eligible element functions, standard headers
# only), host/ (the backend-agnostic host sequence), backends/<name>/ (stage functions + buffers).
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

# Every backend is compiled; only ICCTag.cpp instantiates the selected one, so a test binary can
# instantiate the host sequence on several backends side by side.
set(PORTABLE_SOURCES
    src/ICCTag.cpp
    src/backends/cpu/backend.cpp
    src/backends/stub/backend.cpp)

add_library(CCTag
    ${PORTABLE_SHARED_SOURCES}
    ${PORTABLE_SOURCES})

set_target_properties(CCTag PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

string(TOUPPER "${CCTAG_PORTABLE_BACKEND}" PORTABLE_BACKEND_UPPER)
target_compile_definitions(CCTag PRIVATE CCTAG_PORTABLE_BACKEND_${PORTABLE_BACKEND_UPPER}=1)

# The gradient stage is bit-exact only without fused multiply-add (exact-stage-arithmetic §2.4).
set(PORTABLE_EXACT_FLOAT_OPTIONS "")
if(NOT MSVC)
  set(PORTABLE_EXACT_FLOAT_OPTIONS -ffp-contract=off)
endif()
target_compile_options(CCTag PRIVATE ${PORTABLE_EXACT_FLOAT_OPTIONS})

target_include_directories(CCTag
    PUBLIC
      "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
      "$<BUILD_INTERFACE:${generated_dir}>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CCTAG_PORTABLE_OPENMP OFF)
if(CCTAG_PORTABLE_PARALLEL)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    set(CCTAG_PORTABLE_OPENMP ON)
    target_compile_definitions(CCTag PRIVATE CCTAG_PORTABLE_PARALLEL=1)
    target_link_libraries(CCTag PRIVATE OpenMP::OpenMP_CXX)
  else()
    message(STATUS "CCTag: OpenMP not found, the portable pipeline's kernels run sequentially")
  endif()
endif()
# Consumed by cmake/Config.cmake.in so that find_package(CCTag) links the OpenMP runtime.
set(CCTAG_PORTABLE_OPENMP ${CCTAG_PORTABLE_OPENMP} PARENT_SCOPE)

target_link_libraries(CCTag
    PUBLIC
      ${OpenCV_LIBS}
      Eigen3::Eigen
      Boost::date_time
      Boost::filesystem
      Boost::serialization
      Boost::timer
    PRIVATE
      TBB::tbb)

if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  target_link_libraries(CCTag PRIVATE pthread dl)
endif()

if(CCTAG_BUILD_TESTS)
  add_executable(cctag_portable_smoke test/smoke.cpp)
  target_link_libraries(cctag_portable_smoke PRIVATE CCTag)
  add_test(NAME portable_pipeline_returns_no_detection_candidates COMMAND cctag_portable_smoke)

  add_boost_test(SOURCE test/seam.cpp LINK CCTag PREFIX portable)
  target_include_directories(portable_seam PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  set_target_properties(portable_seam PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  target_compile_options(portable_seam PRIVATE ${PORTABLE_EXACT_FLOAT_OPTIONS})
  if(CCTAG_PORTABLE_OPENMP)
    target_compile_definitions(portable_seam PRIVATE CCTAG_PORTABLE_PARALLEL=1)
    target_link_libraries(portable_seam OpenMP::OpenMP_CXX)
  endif()
endif()
