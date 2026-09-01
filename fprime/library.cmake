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
