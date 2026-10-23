find_package(AdaptiveCpp CONFIG REQUIRED)

# Only these sources are compiled as SYCL, the rest of CCTag stays host code. The numbered files
# are the stages of the pipeline in the order they run.
set(CCTAG_PORTABLE_SYCL_SOURCES
    Pipeline.cpp
    01_Pyramid.cpp
    02_Gradient.cpp
    03_Suppress.cpp
    04_Hysteresis.cpp
    05_Thin.cpp
    06_Link.cpp
    07_Vote.cpp
    08_Center.cpp)
add_library(cctag_portable OBJECT ${CCTAG_PORTABLE_SYCL_SOURCES})
set_target_properties(cctag_portable PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(cctag_portable PRIVATE ${PROJECT_SOURCE_DIR})
add_sycl_to_target(TARGET cctag_portable SOURCES ${CCTAG_PORTABLE_SYCL_SOURCES})

target_sources(CCTag PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Detection.cpp
    $<TARGET_OBJECTS:cctag_portable>)
target_include_directories(CCTag PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_definitions(CCTag PRIVATE CCTAG_WITH_SYCL)
target_link_libraries(CCTag PRIVATE AdaptiveCpp::acpp-rt)

# Times single stages on their own, from the same objects as the library
if(CCTAG_BUILD_PORTABLE_BENCH)
  find_package(CLI11 2 CONFIG REQUIRED)
  find_package(nlohmann_json 3.11 REQUIRED)

  add_executable(cctag-portable-bench bench/main.cpp $<TARGET_OBJECTS:cctag_portable>)
  set_target_properties(cctag-portable-bench PROPERTIES CXX_STANDARD 17)
  target_include_directories(cctag-portable-bench PRIVATE ${PROJECT_SOURCE_DIR})
  add_sycl_to_target(TARGET cctag-portable-bench SOURCES bench/main.cpp)
  target_link_libraries(cctag-portable-bench PRIVATE AdaptiveCpp::acpp-rt opencv_core
    opencv_imgcodecs CLI11::CLI11 nlohmann_json::nlohmann_json)

  install(TARGETS cctag-portable-bench RUNTIME DESTINATION bin)
endif()
