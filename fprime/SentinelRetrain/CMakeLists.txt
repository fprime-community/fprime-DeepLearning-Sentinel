#####
# 'SentinelRetrain' deployment: the retrainer, in its own process.
#
# (!) 47.15b names this as C5's discharger -- "a minimal retrainer deployment --
# its own Top/, a topology carrying the Retrainer and a rate group and nothing
# else, per D70 consequence 2". docs/MODELS.md 61.5a narrows stop 32 so that
# building one is permitted; reading (b), Retrainer into SentinelRef, stays
# rejected on all three of 47.15b's grounds.
#
# (!) THE Retrainer MODULE IS NOT REGISTERED HERE. fprime/SentinelRef/CMakeLists.txt
# already registers it and an F' module may be registered once; fprime/CMakeLists.txt
# adds SentinelRef first, so the target exists by the time this file is read. Keeping
# the directory where it is keeps the UT target name SentinelRef_Retrainer_ut_exe and
# every record citation that names it -- docs/MODELS.md 47.15 and 61.1 among them.
#
# The ${OX_OBJ} guard is repeated here by FILE EXISTENCE rather than by
# if(TARGET ...), which is order- and sub-build-dependent: without the object the
# Retrainer module returns early, its target never exists, and a topology naming
# Retrain.Retrainer would fail at FPP time with a worse message than this one.
#####
set(OX_OBJ "${CMAKE_CURRENT_LIST_DIR}/../../oxcaml/_build/cycle_complete.o")
if(NOT EXISTS "${OX_OBJ}")
    set(OX_OBJ "${CMAKE_CURRENT_LIST_DIR}/../../oxcaml/_build/retrainer_complete.o")
endif()
if(NOT EXISTS "${OX_OBJ}")
    message(STATUS "SentinelRetrain: ${OX_OBJ} absent; skipping. Run scripts/oxcaml_s61.sh")
    return()
endif()

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/Top/")

register_fprime_deployment(
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/Main.cpp"
    DEPENDS
        ${FPRIME_CURRENT_MODULE}_Top
)
