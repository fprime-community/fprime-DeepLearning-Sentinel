####
# D85: the SampleTap -- registered from fprime/library.cmake only when
# SENTINEL_WITH_RETRAINER is ON, with the Retrainer. OFF, a mission's topology never
# sees it and the detector's input path is exactly what it was.
####
register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/SampleTap.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/SampleTap.cpp"
    DEPENDS
        Sentinel_Monitor
)

register_fprime_ut(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/SampleTap.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/SampleTap.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/SampleTapTestMain.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/SampleTapTester.cpp"
    DEPENDS
        Sentinel_Monitor
    UT_AUTO_HELPERS
)
