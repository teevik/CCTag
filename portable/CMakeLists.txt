find_package(AdaptiveCpp CONFIG REQUIRED)

# Only these sources are compiled as SYCL, the rest of CCTag stays host code.
add_library(cctag_portable OBJECT Pipeline.cpp)
set_target_properties(cctag_portable PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(cctag_portable PRIVATE ${PROJECT_SOURCE_DIR})
add_sycl_to_target(TARGET cctag_portable SOURCES Pipeline.cpp)

target_sources(CCTag PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Detection.cpp
    $<TARGET_OBJECTS:cctag_portable>)
target_include_directories(CCTag PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_definitions(CCTag PRIVATE CCTAG_WITH_SYCL)
target_link_libraries(CCTag PRIVATE AdaptiveCpp::acpp-rt)
