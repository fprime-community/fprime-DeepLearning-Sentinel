####
# D84: where the retrainer may be switched on, checked before anything is registered.
#
# fprime/library.cmake calls this only when SENTINEL_WITH_RETRAINER is ON. It is a
# function in its own file so tests/test_library_export_is_opt_in.py can run it
# under `cmake -P` for every system and processor, without F' and without a host
# of each kind.
#
# The supported set is OxCaml's, as docs/DECISIONS.md D83 and D84 state it in the
# case against: x86-64 and arm64 Linux, arm64 macOS. No 32-bit ARM, no musl, no
# documented cross-compile recipe -- and the object is built on the host by
# scripts/oxcaml_shape.sh, so a cross-compiled F' build would link an object for
# the wrong machine. musl cannot be told apart from glibc here; the message says so.
#
# (!) ONE MESSAGE, whatever the reason, so a mission reads the same sentence
# wherever it is refused.
####
function(sentinel_retrainer_check_platform system processor crosscompiling)
    set(_ok FALSE)
    if (system STREQUAL "Darwin" AND processor MATCHES "^(arm64|aarch64)$")
        set(_ok TRUE)
    elseif (system STREQUAL "Linux" AND processor MATCHES "^(x86_64|amd64|AMD64|aarch64|arm64)$")
        set(_ok TRUE)
    endif()
    if (crosscompiling)
        set(_ok FALSE)
        set(processor "${processor}, cross-compiling")
    endif()
    if (NOT _ok)
        message(FATAL_ERROR
            "SENTINEL_WITH_RETRAINER=ON is not supported on ${system} (${processor}). "
            "The retrainer is built with OxCaml, which supports x86-64 and arm64 Linux "
            "and arm64 macOS only: no 32-bit ARM, no musl (which this check cannot "
            "detect), and no documented cross-compile recipe. This project has built it "
            "only on arm64 macOS (docs/DECISIONS.md D83, D84). Set "
            "SENTINEL_WITH_RETRAINER OFF to build the detector alone.")
    endif()
endfunction()
