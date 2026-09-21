####
# HubCounter: the counter docs/MODELS.md 70.6 owed.
#
# (!) REGISTERED HERE AND NOT IN fprime/library.cmake, for the same reason
# PowerSim, Retrainer and ExampleAdapter are: that file exports what a mission
# adopting Sentinel CONSUMES, and measurement apparatus is not that.
####
register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/HubCounter.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/HubCounter.cpp"
    DEPENDS
        Svc_Sched
)
