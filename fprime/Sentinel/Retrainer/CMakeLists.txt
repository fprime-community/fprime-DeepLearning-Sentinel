####
# The retrainer (docs/MODELS.md 47, 61, 72; docs/DECISIONS.md D70, D83, D84).
#
# (!) REGISTERED FROM fprime/library.cmake, AND ONLY WHEN THE MISSION ASKS. D84 exports
# it OPT-IN: library.cmake adds this directory only when SENTINEL_WITH_RETRAINER is
# ON, after a platform check. OFF -- the default -- this file is never read and a
# mission adopting Sentinel inherits no OCaml runtime. Until D84 it was registered by
# SentinelRef instead, precisely so the library would not export it.
#
# (!) IT LINKS ONE OBJECT, NOT THE OCaml ARCHIVES, AND THE FIRST ATTEMPT GOT THAT WRONG.
# `retrainer.a` + `libasmrun.a` leaves about thirty symbols undefined -- caml_program,
# caml_globals, caml_frametable, caml_code_segments, caml_unit_deps_table and the whole
# caml_exn_* set. ocamlopt SYNTHESISES those at link time from the particular set of
# modules being linked, so they are in no archive. `ocamlopt -output-complete-obj` emits
# them, together with the compiled OCaml, the C stubs and the runtime, as a single object.
#
# (!) THE OBJECT IS THE MISSION'S SHAPE, GENERATED (D83.1 route 1, D84).
# `scripts/oxcaml_shape.sh --channels C --predictions P` generates deep_f32 at that
# shape, runs its gates, and leaves the object and `sentinel_cycle_shape.h` in
# oxcaml/_build/shape-cC-pP/. SENTINEL_RETRAINER_CHANNELS and _PREDICTIONS select it.
#
# (!) AND IT STOPS RATHER THAN SKIPS. Before D84 an absent object made this module
# skip itself with a STATUS line, because nobody had asked for it. Now a mission HAS
# asked, by switching the option on, so an absent or mismatched shape is an error
# that names the command -- a silent skip here would print `ninja: no work to do`
# and build nothing, which is the failure D83.3 found in a README.
####
set(OX_SHAPE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../oxcaml/_build/shape-c${SENTINEL_RETRAINER_CHANNELS}-p${SENTINEL_RETRAINER_PREDICTIONS}")
get_filename_component(OX_SHAPE_DIR "${OX_SHAPE_DIR}" ABSOLUTE)
set(OX_OBJ "${OX_SHAPE_DIR}/cycle_complete.o")
set(OX_SHAPE_H "${OX_SHAPE_DIR}/sentinel_cycle_shape.h")
set(OX_SRC "${CMAKE_CURRENT_LIST_DIR}/../../../oxcaml/retrainer")

if(NOT EXISTS "${OX_OBJ}" OR NOT EXISTS "${OX_SHAPE_H}")
    message(FATAL_ERROR
        "SENTINEL_WITH_RETRAINER is ON, but the retrainer has not been built at "
        "${SENTINEL_RETRAINER_CHANNELS} channels, ${SENTINEL_RETRAINER_PREDICTIONS} predictions "
        "(no ${OX_OBJ}). Run, from the repository root:\n"
        "  scripts/oxcaml_setup.sh\n"
        "  bash scripts/oxcaml_shape.sh --channels ${SENTINEL_RETRAINER_CHANNELS} "
        "--predictions ${SENTINEL_RETRAINER_PREDICTIONS}\n"
        "then regenerate. To build the detector alone, set SENTINEL_WITH_RETRAINER OFF.")
endif()

# The generated header must be the shape that was asked for. Read, not trusted.
file(STRINGS "${OX_SHAPE_H}" _sentinel_shape_c REGEX "^#define SENTINEL_CYCLE_CHANNELS ")
file(STRINGS "${OX_SHAPE_H}" _sentinel_shape_p REGEX "^#define SENTINEL_CYCLE_PREDICTIONS ")
string(REGEX REPLACE ".* ([0-9]+)U$" "\\1" _sentinel_shape_c "${_sentinel_shape_c}")
string(REGEX REPLACE ".* ([0-9]+)U$" "\\1" _sentinel_shape_p "${_sentinel_shape_p}")
if(NOT _sentinel_shape_c STREQUAL "${SENTINEL_RETRAINER_CHANNELS}"
   OR NOT _sentinel_shape_p STREQUAL "${SENTINEL_RETRAINER_PREDICTIONS}")
    message(FATAL_ERROR
        "${OX_SHAPE_H} declares ${_sentinel_shape_c} channels, ${_sentinel_shape_p} "
        "predictions; SENTINEL_RETRAINER_CHANNELS/_PREDICTIONS ask for "
        "${SENTINEL_RETRAINER_CHANNELS}/${SENTINEL_RETRAINER_PREDICTIONS}. Re-run "
        "scripts/oxcaml_shape.sh for the shape you want.")
endif()
message(STATUS "Retrainer: ${SENTINEL_RETRAINER_CHANNELS} channels, "
               "${SENTINEL_RETRAINER_PREDICTIONS} predictions, from ${OX_SHAPE_DIR}")

# D85: the component is a shell around RetrainLoop, which runs a replica of the flight
# core (sentinel_core) and takes the tap's samples (Sentinel_SampleTap's port type).
register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/Retrainer.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/Retrainer.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/RetrainLoop.cpp"
    DEPENDS
        Svc_Sched
        Fw_Logger
        Sentinel_SampleTap
        sentinel_core
)

# PUBLIC: Retrainer.hpp includes the generated sentinel_cycle_shape.h, and every
# topology that instances the component compiles Retrainer.hpp.
target_include_directories(${FPRIME_CURRENT_MODULE} PUBLIC "${OX_SHAPE_DIR}")
target_include_directories(${FPRIME_CURRENT_MODULE} PRIVATE "${OX_SRC}")
target_link_libraries(${FPRIME_CURRENT_MODULE} PUBLIC "${OX_OBJ}")

### Unit tests -- and this is the process E1 runs in (docs/MODELS.md 47.9 X1) ###
register_fprime_ut(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/Retrainer.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/Retrainer.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/RetrainLoop.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/RetrainerTestMain.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/RetrainerTester.cpp"
    DEPENDS
        Svc_Sched
        Fw_Logger
        Sentinel_SampleTap
        sentinel_core
    UT_AUTO_HELPERS
)

if (TARGET Sentinel_Retrainer_ut_exe)
    target_include_directories(Sentinel_Retrainer_ut_exe PRIVATE "${OX_SHAPE_DIR}" "${OX_SRC}")
    target_link_libraries(Sentinel_Retrainer_ut_exe PRIVATE "${OX_OBJ}")
endif()
