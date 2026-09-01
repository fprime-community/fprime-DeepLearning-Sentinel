# The F' build, for work item 9. Work item 8 verifies through the Makefile.
#
# The flag set below is IDENTICAL to flight/Makefile's by intent, so the F' build
# cannot be quietly laxer than the one work item 8 measured against.
# `tests/test_flight_build.py` asserts the two lists agree rather than trusting
# them to; if you change one, change both and the test will tell you if you did not.
#
# Work item 9 replaces include/sentinel/Types.hpp's body with
# `#include "Fw/FPrimeBasicTypes.hpp"` and may substitute Utils::Hash for
# src/Crc32.cpp; nothing else in this directory changes. F' is pinned at v4.3.0
# (docs/DECISIONS.md D31).

cmake_minimum_required(VERSION 3.16)
project(sentinel_flight CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# SENTINEL_FLIGHT_FLAGS -- keep in step with flight/Makefile
set(SENTINEL_FLIGHT_FLAGS
    -fno-exceptions
    -fno-rtti
    -ffp-contract=off
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
    -Werror
)

set(SENTINEL_SOURCES
    src/Crc32.cpp
    src/Status.cpp
    src/ModelFile.cpp
    src/Gru.cpp
    src/Ewma.cpp
    src/Detector.cpp
    src/Baseline.cpp
)

add_library(sentinel_core STATIC ${SENTINEL_SOURCES})
target_include_directories(sentinel_core PUBLIC include)
target_compile_options(sentinel_core PRIVATE ${SENTINEL_FLIGHT_FLAGS})

# -ffast-math and -Ofast are forbidden outright: they license reassociation and
# reciprocal substitution, either of which changes results against the NumPy
# reference every measurement in this project was taken with.
foreach(forbidden "-ffast-math" "-Ofast" "-funsafe-math-optimizations")
    if(CMAKE_CXX_FLAGS MATCHES "${forbidden}")
        message(FATAL_ERROR
            "${forbidden} breaks bit-identity with the reference; see docs/MODEL_FILE.md 9")
    endif()
endforeach()

foreach(binary GoldenVectors RefusalTests DeterminismTest RoundTripDump Footprint
               BaselineVectors)
    add_executable(${binary} test/${binary}.cpp)
    target_include_directories(${binary} PRIVATE test)
    target_link_libraries(${binary} PRIVATE sentinel_core)
    target_compile_options(${binary} PRIVATE ${SENTINEL_FLIGHT_FLAGS})
endforeach()
