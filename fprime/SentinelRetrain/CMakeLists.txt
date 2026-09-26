#####
# 'SentinelRetrain' deployment: the retrainer, in its own process.
#
# (!) 47.15b names this as C5's discharger -- "a minimal retrainer deployment --
# its own Top/, a topology carrying the Retrainer and a rate group and nothing
# else, per D70 consequence 2". docs/MODELS.md 61.5a narrows stop 32 so that
# building one is permitted; reading (b), Retrainer into SentinelRef, stays
# rejected on all three of 47.15b's grounds.
#
# (!) THE Retrainer MODULE IS NOT REGISTERED HERE. fprime/library.cmake registers it
# when SENTINEL_WITH_RETRAINER is ON (docs/DECISIONS.md D84), and an F' module may be
# registered once; fprime/CMakeLists.txt includes the library first, so the target
# exists by the time this file is read. Until D84 it was registered by SentinelRef,
# and this comment argued for keeping it there to keep the UT target name
# SentinelRef_Retrainer_ut_exe that records cite. The records keep that name --
# they were true at their commits -- and D84 consequence 5 maps it to
# Sentinel_Retrainer_ut_exe.
#
# OFF, this deployment is skipped by NAME rather than by file existence: the
# option is the mission's statement of intent, and a missing object with the
# option ON is an error the Retrainer module raises itself, naming the command.
#####
if(NOT SENTINEL_WITH_RETRAINER)
    message(STATUS "SentinelRetrain: SENTINEL_WITH_RETRAINER is OFF; skipping (docs/DECISIONS.md D84)")
    return()
endif()

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/Top/")

register_fprime_deployment(
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/Main.cpp"
    DEPENDS
        ${FPRIME_CURRENT_MODULE}_Top
)
