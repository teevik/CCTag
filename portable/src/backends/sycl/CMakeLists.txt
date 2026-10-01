set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp compilation targets")
find_package(AdaptiveCpp 25.10 REQUIRED CONFIG)

# Only this translation unit needs the SYCL compiler. The public API stays host code.
add_library(cctag_sycl_stub OBJECT backend.cpp)
cctag_portable_configure(cctag_sycl_stub)
set_target_properties(cctag_sycl_stub PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(cctag_sycl_stub PRIVATE ${PROJECT_SOURCE_DIR}/src ${generated_dir})
target_link_libraries(cctag_sycl_stub PRIVATE Eigen3::Eigen Boost::serialization)
target_compile_options(cctag_sycl_stub PRIVATE -fno-fast-math)
add_sycl_to_target(TARGET cctag_sycl_stub SOURCES backend.cpp)
target_sources(CCTag PRIVATE $<TARGET_OBJECTS:cctag_sycl_stub>)
target_link_libraries(CCTag PRIVATE AdaptiveCpp::acpp-rt)

add_executable(portable_sycl_smoke smoke.cpp)
cctag_portable_configure(portable_sycl_smoke)
target_link_libraries(portable_sycl_smoke PRIVATE CCTag)

if(CCTAG_BUILD_TESTS)
  # Exercise the selected backend through the public API, including a real SYCL launch.
  add_test(NAME portable_sycl_smoke_cpu COMMAND portable_sycl_smoke cpu)
endif()

install(TARGETS portable_sycl_smoke RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
