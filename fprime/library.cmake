####
# library.cmake for fprime-sentinel
#
# This makes fprime/ an F' library. Any project consumes the Sentinel component
# by adding this directory to `library_locations` in its settings.ini and
# nothing else -- which is what scripts/fprime_ref_patch.sh does to build
# Sentinel inside F's own reference deployment without copying a source file,
# and what a mission would do to adopt it.
#
# Modules live under the Sentinel/ namespace directory because F' says so
# plainly: "Placing container directories directly at the root of the repository
# is *strongly* forbidden" -- a library with a module called Monitor at its root
# would collide with the next library that has one
# (docs/how-to/develop/develop-fprime-libraries.md).
####

# The work item 8 inference core: plain CMake, not an F' module, because it has
# no FPP and no ports and must keep building with no F' at all. The library
# brings it so a consumer does not have to know it exists.
#
# The core's F' personality is switched on here rather than in
# flight/CMakeLists.txt, for the same reason. PUBLIC, so the definition and F's
# headers reach anything linking the core and the library and its users cannot
# disagree about what U32 is (D35).
if (NOT TARGET sentinel_core)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../flight" "sentinel-core")
    target_compile_definitions(sentinel_core PUBLIC SENTINEL_FPRIME_TYPES)
    target_link_libraries(sentinel_core PUBLIC Fw_Types)
endif()

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/Sentinel/Monitor")

# ----------------------------------------------------------------------------------
# D84: the retrainer, OPT-IN. OFF by default, and OFF means OFF.
#
# With SENTINEL_WITH_RETRAINER OFF nothing below the option is evaluated: the
# exported set is exactly `sentinel_core` and `Sentinel/Monitor`, as before D84, no
# OxCaml toolchain is needed, and no OCaml runtime reaches any binary.
# tests/test_library_export_is_opt_in.py checks that by configuring this file.
#
# ON, it registers `Sentinel/Retrainer` -- after refusing any platform OxCaml does
# not support, with one message -- at the shape SENTINEL_RETRAINER_CHANNELS and
# SENTINEL_RETRAINER_PREDICTIONS name, which `scripts/oxcaml_shape.sh` must have
# generated and gated first. The component runs in its OWN deployment, never
# beside the detector (D70 consequence 2); fprime/SentinelRetrain/ is the pattern.
#
# Status, D83's words exactly: the chosen retraining implementation,
# host-verified, not flight-qualified. The case against sits beside it in D84.
# ----------------------------------------------------------------------------------
option(SENTINEL_WITH_RETRAINER
       "Export the OxCaml retrainer (docs/DECISIONS.md D84). Needs scripts/oxcaml_setup.sh."
       OFF)
if (SENTINEL_WITH_RETRAINER)
    set(SENTINEL_RETRAINER_CHANNELS "16" CACHE STRING
        "Channels the retrainer is generated for, 1..16 (scripts/oxcaml_shape.sh)")
    set(SENTINEL_RETRAINER_PREDICTIONS "10" CACHE STRING
        "Predictions the retrainer is generated for, 1..10 (scripts/oxcaml_shape.sh)")
    include("${CMAKE_CURRENT_LIST_DIR}/Sentinel/Retrainer/retrainer_platform.cmake")
    sentinel_retrainer_check_platform("${CMAKE_SYSTEM_NAME}" "${CMAKE_SYSTEM_PROCESSOR}"
                                      "${CMAKE_CROSSCOMPILING}")
    add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/Sentinel/Retrainer")
endif()
