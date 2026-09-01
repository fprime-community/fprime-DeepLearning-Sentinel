####
# The fprime-sentinel F' project.
#
# Shaped after F's own TestDeploymentsProject/CMakeLists.txt at v4.3.0, which is
# how a project outside the framework tree consumes it: find_package(FPrime),
# then add the framework as a subdirectory named F-Prime.
#
# fprime_add_validation_flags() holds this project to the warning set F' itself
# validates against. The Sentinel core carries its own stricter set on top,
# applied per target in flight/CMakeLists.txt and asserted against the Makefile
# by tests/test_flight_build.py, so the core is never built quieter here than
# work item 8 verified it.
####
cmake_policy(SET CMP0048 NEW)
cmake_minimum_required(VERSION 3.26)
project(SentinelFprime VERSION 0.6.0 LANGUAGES C CXX)

set(FPRIME_INCLUDE_FRAMEWORK_CODE OFF)
find_package(FPrime REQUIRED PATHS "${CMAKE_CURRENT_LIST_DIR}/lib/fprime")
fprime_add_validation_flags()
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/lib/fprime" "F-Prime")

# The work item 8 inference core, built as the STATIC library D31 consequence 4
# wrote flight/CMakeLists.txt to provide. Plain CMake, not an F' module: it has
# no FPP and no ports, and it must keep building without F' at all.
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../flight" "sentinel-core")

# The core's F' personality is switched on from here, not from
# flight/CMakeLists.txt, which must keep building with no F' at all. PUBLIC so
# the definition and F's headers propagate to anything linking the core, and so
# the library and its users cannot disagree about what U32 is (D35).
target_compile_definitions(sentinel_core PUBLIC SENTINEL_FPRIME_TYPES)
target_link_libraries(sentinel_core PUBLIC Fw_Types)
# Order matters and F' says so plainly if it is wrong: a deployment can only
# reference modules already defined, so the component comes before the
# deployment that instantiates it.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/Monitor/")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/SentinelRef/")
