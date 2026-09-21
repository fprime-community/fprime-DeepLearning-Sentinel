####
# ExampleAdapter: F's Passive Adapter Pattern, at its smallest useful size.
#
# (!) REGISTERED HERE AND NOT IN fprime/library.cmake, for the same reason
# PowerSim and Retrainer are: that file exports what a mission adopting Sentinel
# CONSUMES, and example code is not that. A mission copies this directory; it
# does not link it.
####
register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/ChannelAdapter.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/ChannelAdapter.cpp"
    DEPENDS
        Sentinel_Monitor
        Svc_Sched
)
