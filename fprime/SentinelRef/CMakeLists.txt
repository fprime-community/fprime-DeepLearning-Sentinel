#####
# 'SentinelRef' Deployment:
#
# This registers the 'SentinelRef' deployment to the build system. 
# Custom components that have not been added at the project-level should be added to 
# the list below.
#
#####

###
# Topology and Components
###

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/Top/")

# Add custom components to this specific deployment here.
#
# The physics testbed (docs/MODELS.md 42). It is registered HERE and not in
# fprime/library.cmake: that file exports what a mission adopting Sentinel
# consumes, and a simulated battery is not that.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/PowerSim/")

# The retrainer is NOT registered here any more. Until D84 it lived in this
# deployment's directory so the library would not export it; D84 exports it opt-in,
# from fprime/library.cmake, and it moved to fprime/Sentinel/Retrainer/. This
# deployment never instanced it and still does not: SentinelRef carries no OCaml
# runtime, which tests/test_detector_binary_has_no_ocaml_runtime.py reads by symbol.

# The Passive Adapter Pattern example (docs/DESIGN.md 8). Registered HERE for the
# same reason PowerSim is: fprime/library.cmake exports what a mission adopting
# Sentinel consumes, and example code is not that. A mission copies this directory
# rather than linking it.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/ExampleAdapter/")

# The hub tap (docs/MODELS.md 70.6). Registered HERE for the same reason the
# other three are: fprime/library.cmake exports what a mission adopting Sentinel
# consumes, and measurement apparatus is not that.
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/ExampleSource/")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/HubCounter/")

register_fprime_deployment(
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/Main.cpp"
    DEPENDS
        ${FPRIME_CURRENT_MODULE}_Top
)
