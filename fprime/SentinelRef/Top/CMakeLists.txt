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

# D85: WITH THE RETRAINER ON, THE LOOP TOPOLOGY -- GENERATED, NOT A SECOND COPY.
# topology.fpp is today's topology and stays the only source. When
# SENTINEL_WITH_RETRAINER is ON, the loop variant is made from it by exactly two
# anchored substitutions -- the tap declared after powerSim, and PowerSim -> Monitor
# rerouted through the tap with the tap's copy into the hub's serial port -- and the
# configure stops if either anchor has moved. OFF, nothing here runs and SentinelRef is
# exactly what it was (D84 SX11).
set(_sentinelref_topology "${CMAKE_CURRENT_LIST_DIR}/topology.fpp")
set(_sentinelref_loop_inputs "")
set(_sentinelref_loop_depends "")
if (SENTINEL_WITH_RETRAINER)
    file(READ "${CMAKE_CURRENT_LIST_DIR}/topology.fpp" _topo)
    set(_a1 "    instance powerSim\n")
    set(_a2 "      powerSim.channelOut -> sentinelMonitor.channelsIn\n")
    string(FIND "${_topo}" "${_a1}" _f1)
    string(FIND "${_topo}" "${_a2}" _f2)
    if (_f1 EQUAL -1 OR _f2 EQUAL -1)
        message(FATAL_ERROR "SentinelRef/Top/topology.fpp no longer carries the two lines "
                            "the loop topology is generated from; update Top/CMakeLists.txt")
    endif()
    string(REPLACE "${_a1}" "${_a1}    instance sampleTap\n" _topo "${_topo}")
    string(REPLACE "${_a2}"
        "      # D85: through the tap -- the Monitor first and unchanged, then a copy to the hub\n      powerSim.channelOut -> sampleTap.sampleIn\n      sampleTap.sampleOut -> sentinelMonitor.channelsIn\n      sampleTap.sampleCopy -> hub.serialIn[0]\n"
        _topo "${_topo}")
    # In this module's binary directory, not a subdirectory of it: FPP names every
    # generated header by the .fpp file's path, so a loop/ subdirectory here would make
    # the topology include "SentinelRef/Top/loop/...Ac.hpp", which is never written.
    set(_sentinelref_topology "${CMAKE_CURRENT_BINARY_DIR}/topology.fpp")
    file(WRITE "${_sentinelref_topology}" "${_topo}")
    set(_sentinelref_loop_inputs "${CMAKE_CURRENT_LIST_DIR}/loop/instances_loop.fpp")
    set(_sentinelref_loop_depends Sentinel_SampleTap)
    message(STATUS "SentinelRef: the loop topology (SampleTap before the Monitor), D85")
endif()

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/instances.fpp"
        ${_sentinelref_loop_inputs}
        "${CMAKE_CURRENT_LIST_DIR}/system.fpp"
        "${_sentinelref_topology}"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/SentinelRefTopology.cpp"
    DEPENDS
        Fw_Logger
        ${_sentinelref_loop_depends}
)
