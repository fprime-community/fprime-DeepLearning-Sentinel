####
# EmitProbe: LC2's detector-side log (docs/MODELS.md 78.11).
#
# (!) REGISTERED HERE AND NOT IN fprime/library.cmake, for HubCounter's reason: that
# file exports what a mission adopting Sentinel CONSUMES, and measurement apparatus is
# not that. And only with the retrainer ON: it reads the SampleTap, which exists only
# then, and it is instanced only in the loop topology.
####
register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/EmitProbe.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/EmitProbe.cpp"
    DEPENDS
        Svc_Sched
        Fw_Logger
        Sentinel_Monitor
        Sentinel_SampleTap
)
