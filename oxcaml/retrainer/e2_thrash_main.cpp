// E2 arms B and B2: a process whose only job is to abuse a garbage collector.
// docs/MODELS.md 47.9.1 and 47.9.2.
//
// argv[1] domains (1 for B, one-per-core for B2), argv[2] milliseconds, argv[3] a file
// to write the REAL progress counter into. Bounded rather than signal-terminated: a
// signal handler cannot safely call into OCaml, and 47.9.2 requires each arm to report
// the thrasher's own progress rather than a proxy for it having been alive.
#include "sentinel_thrash.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: e2_thrash <domains> <ms> <counter_file>\n");
        return 2;
    }
    if (sentinel_retrainer_boot() != 0) {
        std::fprintf(stderr, "thrash: runtime unavailable\n");
        return 1;
    }
    const int32_t n  = static_cast<int32_t>(std::atoi(argv[1]));
    const int32_t ms = static_cast<int32_t>(std::atoi(argv[2]));
    const int32_t c  = sentinel_thrash_run_for(n, ms, argv[3]);

    std::FILE* f = std::fopen(argv[3], "w");
    if (f != nullptr) { std::fprintf(f, "%d\n", c); std::fclose(f); }
    return 0;
}
