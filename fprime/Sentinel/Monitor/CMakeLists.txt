####
# F Prime CMakeLists.txt:
#
# SOURCES: list of source files (to be compiled)
# AUTOCODER_INPUTS: list of files to be passed to the autocoders
# DEPENDS: list of libraries that this module depends on
#
# More information in the F' CMake API documentation:
# https://fprime.jpl.nasa.gov/latest/docs/reference/api/cmake/API/
#
####

# Module names are derived from the path from the nearest project/library/framework
# root when not specifically overridden by the developer, i.e. the module defined by
# `MyProj/Some/Path/CMakeLists.txt` will be named `MyProj_Some_Path`.

register_fprime_library(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/Monitor.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/Monitor.cpp"
    DEPENDS
        Svc_Sched
        Os
        sentinel_core
)

### Unit Tests ###
register_fprime_ut(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/Monitor.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/MonitorTestMain.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/MonitorTester.cpp"
    DEPENDS
        Svc_Sched
        Os
        sentinel_core
    UT_AUTO_HELPERS
)

# The refusal tests mutate a known-good model.bin, exactly as
# flight/test/RefusalTests.cpp does, so they need the work item 8 vectors. The
# path is a definition rather than a relative path, because a unit test's working
# directory is the build cache and not the source tree.
# SENTINEL_UT_TMP keeps the test's scratch model file out of the source tree.
# A test that dirties the tree is what tests/test_no_local_persistence.py exists
# to forbid on the Python side, and the same rule applies here.
if (TARGET Sentinel_Monitor_ut_exe)
    target_compile_definitions(Sentinel_Monitor_ut_exe PRIVATE
        "SENTINEL_VECTORS_DIR=\"${CMAKE_CURRENT_LIST_DIR}/../../../flight/test/vectors\""
        "SENTINEL_UT_TMP=\"${CMAKE_CURRENT_BINARY_DIR}\"")
endif()
