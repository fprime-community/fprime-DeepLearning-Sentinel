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

# The component and the core arrive through library.cmake, the same file any
# other project consumes them with, so this project is not a privileged consumer
# of its own component -- if the library is broken for a mission it is broken
# here too. Order matters and F' says so plainly if it is wrong: a deployment can
# only reference modules already defined.
include("${CMAKE_CURRENT_LIST_DIR}/library.cmake")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/SentinelRef/")

# The retrainer's own deployment (docs/MODELS.md 65, 47.15b's named discharger for
# C5). AFTER SentinelRef, and the order is load-bearing: SentinelRef registers the
# Retrainer module and an F' module may be registered once, so the target must
# already exist when this deployment's topology names Retrain.Retrainer.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/SentinelRetrain/")
