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

register_fprime_deployment(
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/Main.cpp"
    DEPENDS
        ${FPRIME_CURRENT_MODULE}_Top
)
