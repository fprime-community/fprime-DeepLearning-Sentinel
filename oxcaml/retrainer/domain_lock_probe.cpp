// ======================================================================
// \title  domain_lock_probe.cpp
// \brief  reproduces "Fatal error: no domain lock held" on demand, and
//         shows the same calls succeeding on the booting thread
//
// (!) THIS IS THE NEGATIVE CONTROL FOR THE FIX AT Retrainer.cpp:97-108.
// docs/MODELS.md 65.6 recorded that the first real tick of the retrainer in a
// deployment died with "Fatal error: no domain lock held", and recorded the
// fix -- boot the runtime on the thread that will then call into it. Nothing
// reproduced the failure, so the fix was believed in one direction only.
//
// OCaml 5 grants the domain lock to whichever thread calls caml_startup. This
// probe boots on the main thread, exactly as a topology's configureTopology()
// would, and then either:
//
//   same    calls into OCaml on that same thread            -- must succeed
//   cross   calls into OCaml on a second pthread            -- must abort
//
// `cross` is the shape F' produces: Svc.ActiveRateGroup runs schedIn on its own
// task, so a runtime booted in configureTopology() is booted on one thread and
// called from another. Reaching the end of `cross` without aborting is itself a
// reportable result, and the probe says so and exits non-zero rather than
// passing quietly.
//
// No exceptions and no RTTI, and pthreads rather than std::thread, so this
// builds at the same flag set as the rest of the C++ boundary (61 / FC3).
// ======================================================================
#include "sentinel_cycle.h"

#include <pthread.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

//: The other thread's return code. Written by one thread and read after join,
//: so the join is the synchronisation and no atomic is needed.
int32_t g_crossRc = 0;

void* call_into_ocaml(void* /*unused*/)
{
    // sentinel_cycle_init is the cheapest entry point that actually crosses:
    // CAMLparam0() touches the domain state and caml_callback needs the lock.
    g_crossRc = sentinel_cycle_init(1);
    return nullptr;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* mode = (argc > 1) ? argv[1] : "same";

    // Booted HERE, on the main thread. This is the line that decides which
    // thread owns the domain for the life of the process.
    const int32_t booted = sentinel_cycle_boot();
    if (booted != SENTINEL_CYC_OK) {
        std::printf("  NO VERDICT: the runtime is unavailable (rc %d)\n",
                    static_cast<int>(booted));
        return 2;
    }
    std::printf("  runtime booted on the main thread\n");

    if (std::strcmp(mode, "cross") == 0) {
        pthread_t other;
        if (pthread_create(&other, nullptr, call_into_ocaml, nullptr) != 0) {
            std::printf("  NO VERDICT: could not create the second thread\n");
            return 2;
        }
        static_cast<void>(pthread_join(other, nullptr));

        // (!) Reaching this line means the process did NOT abort, and that is
        // the finding, not a pass. The whole fix at Retrainer.cpp:97-108 rests
        // on a cross-thread call being fatal.
        std::printf("  (!) cross-thread call RETURNED %d -- no abort\n",
                    static_cast<int>(g_crossRc));
        return 4;
    }

    const int32_t rc = sentinel_cycle_init(1);
    std::printf("  same-thread call returned %d\n", static_cast<int>(rc));
    return (rc == SENTINEL_CYC_OK) ? 0 : 3;
}
