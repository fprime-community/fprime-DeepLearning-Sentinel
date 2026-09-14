####
# The physics testbed's subsystem simulation (docs/MODELS.md 42).
#
# (!) IT IS REGISTERED FROM THE DEPLOYMENT AND NOT FROM library.cmake, ON
# PURPOSE. fprime/library.cmake exports what a mission adopting Sentinel
# consumes; a simulated battery is not that. This builds only inside
# SentinelRef, which is where the testbed lives.
#
# PowerPlant.cpp carries no F' header and is compiled here rather than as a
# separate library: it is one translation unit and splitting it would make the
# tuning driver's include path a build-system question.
####
register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/PowerSim.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/PowerSim.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/PowerPlant.cpp"
    DEPENDS
        Svc_Sched
        Sentinel_Monitor
)
