set(PORTABLE_SHARED_SOURCES
    ${PROJECT_SOURCE_DIR}/src/cctag/CCTagMarkersBank.cpp
    ${PROJECT_SOURCE_DIR}/src/cctag/Params.cpp
    ${PROJECT_SOURCE_DIR}/src/cctag/geometry/Ellipse.cpp)

add_library(CCTag
    ${PORTABLE_SHARED_SOURCES}
    src/ICCTag.cpp)

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
      Boost::timer
    PRIVATE
      TBB::tbb)

if(CCTAG_WITH_CUDA)
  target_compile_definitions(CCTag PUBLIC CCTAG_WITH_CUDA)
endif()

if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  target_link_libraries(CCTag PRIVATE pthread dl)
endif()

if(CCTAG_BUILD_TESTS)
  add_executable(cctag_portable_smoke test/smoke.cpp)
  target_link_libraries(cctag_portable_smoke PRIVATE CCTag)
  add_test(NAME portable_pipeline_returns_no_detection_candidates COMMAND cctag_portable_smoke)
endif()
