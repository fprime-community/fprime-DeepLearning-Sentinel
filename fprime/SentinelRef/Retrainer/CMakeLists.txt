####
# E1, the pipe (docs/MODELS.md 47, docs/DECISIONS.md D70).
#
# (!) REGISTERED FROM THE DEPLOYMENT AND NOT FROM library.cmake, ON PURPOSE, and for the
# same reason PowerSim is: fprime/library.cmake exports what a mission adopting Sentinel
# consumes, and an OCaml runtime is emphatically not that. This builds only inside
# SentinelRef.
#
# (!) IT LINKS ONE OBJECT, NOT THE OCaml ARCHIVES, AND THE FIRST ATTEMPT GOT THAT WRONG.
# `retrainer.a` + `libasmrun.a` leaves about thirty symbols undefined -- caml_program,
# caml_globals, caml_frametable, caml_code_segments, caml_unit_deps_table and the whole
# caml_exn_* set. ocamlopt SYNTHESISES those at link time from the particular set of
# modules being linked, so they are in no archive. `ocamlopt -output-complete-obj` emits
# them, together with the compiled OCaml, the C stubs and the runtime, as a single object.
# `scripts/oxcaml_e1.sh` produces it.
#
# The toolchain is gitignored and script-rebuilt (docs/MODELS.md 47.8), so if it is absent
# this module skips itself with a message rather than failing the build -- the same posture
# flight/Makefile's lint target takes towards an absent clang-tidy.
####
# 61 / E5-c adds a superset object -- E1's five entry points plus the float32
# cycle's five -- because two -output-complete-obj objects cannot be linked into
# one binary, each carrying its own runtime. E1's own object is unchanged and
# scripts/oxcaml_e1.sh still builds it; this prefers the superset when it exists.
set(OX_OBJ "${CMAKE_CURRENT_LIST_DIR}/../../../oxcaml/_build/cycle_complete.o")
if(NOT EXISTS "${OX_OBJ}")
    set(OX_OBJ "${CMAKE_CURRENT_LIST_DIR}/../../../oxcaml/_build/retrainer_complete.o")
endif()
set(OX_SRC "${CMAKE_CURRENT_LIST_DIR}/../../../oxcaml/retrainer")

if(NOT EXISTS "${OX_OBJ}")
    message(STATUS "Retrainer: ${OX_OBJ} absent; skipping. Run scripts/oxcaml_e1.sh")
    return()
endif()

register_fprime_module(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/Retrainer.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/Retrainer.cpp"
    DEPENDS
        Svc_Sched
)

target_include_directories(${FPRIME_CURRENT_MODULE} PRIVATE "${OX_SRC}")
target_link_libraries(${FPRIME_CURRENT_MODULE} PUBLIC "${OX_OBJ}")

### Unit tests -- and this is the process E1 runs in (docs/MODELS.md 47.9 X1) ###
register_fprime_ut(
    AUTOCODER_INPUTS
        "${CMAKE_CURRENT_LIST_DIR}/Retrainer.fpp"
    SOURCES
        "${CMAKE_CURRENT_LIST_DIR}/Retrainer.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/RetrainerTestMain.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/test/ut/RetrainerTester.cpp"
    DEPENDS
        Svc_Sched
    UT_AUTO_HELPERS
)

if (TARGET SentinelRef_Retrainer_ut_exe)
    target_include_directories(SentinelRef_Retrainer_ut_exe PRIVATE "${OX_SRC}")
    target_link_libraries(SentinelRef_Retrainer_ut_exe PRIVATE "${OX_OBJ}")
endif()
