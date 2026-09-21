####
# ExampleSource: a deterministic synthetic feed for the worked adoption chain.
#
# (!) REGISTERED HERE AND NOT IN fprime/library.cmake, for the same reason
# PowerSim, Retrainer and ExampleAdapter are: that file exports what a mission
# adopting Sentinel CONSUMES, and example code is not that.
#
# (!) AND IT IS NOT INSTANCED BY THIS DEPLOYMENT, so `fprime-util build -p
# ./SentinelRef` does not build it -- that exits 0 having built nothing, which
# reads exactly like success. Build it by its own path:
#
#     cd fprime/SentinelRef/ExampleSource && fprime-util build
####
register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/ExampleSource.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/ExampleSource.cpp"
    DEPENDS
        SentinelRef_ExampleAdapter
        Svc_Sched
)
