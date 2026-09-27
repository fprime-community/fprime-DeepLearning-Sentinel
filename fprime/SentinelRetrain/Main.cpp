// ======================================================================
// \title  Main.cpp
// \brief  the retrainer's own process
//
// (!) A SEPARATE OS PROCESS IS A REQUIREMENT AND NOT A CHOICE. D70 consequence
// 2: OCaml 5's minor collector is stop-the-world across all domains, so a
// retrainer sharing a process with the detector would stall it at a GC barrier
// even if the retrainer's own code allocated nothing.
// ======================================================================
#include <SentinelRetrain/Top/SentinelRetrainTopology.hpp>
#include <Os/Os.hpp>
#include <signal.h>
#include <getopt.h>
#include <cstdlib>
#include <Fw/Logger/Logger.hpp>

void print_usage(const char* app) {
    // One call per line: Fw::Logger truncates a long message (found in D85).
    Fw::Logger::log("Usage: ./%s [options]\n", app);
    Fw::Logger::log("-a\thub server hostname/IP address (default 127.0.0.1)\n");
    Fw::Logger::log("-p\thub server port; 0 or absent runs the cycle with no hub\n");
    Fw::Logger::log("-t\tbase tick in microseconds; absent runs at 1 Hz (host runs only)\n");
}

static void signalHandler(int signum) {
    static_cast<void>(signum);
    SentinelRetrain::stopRateGroups();
}

int main(int argc, char* argv[]) {
    I32 option = 0;
    const char* hostname = "127.0.0.1";
    U16 port_number = 0;
    U32 tick_micros = 0;

    Os::init();

    while ((option = getopt(argc, argv, "hp:a:t:")) != -1) {
        switch (option) {
            case 'a':
                hostname = optarg;
                break;
            case 'p':
                port_number = static_cast<U16>(atoi(optarg));
                break;
            // D85: the accelerated clock for the end-to-end loop run.
            case 't':
                tick_micros = static_cast<U32>(strtoul(optarg, nullptr, 10));
                break;
            case 'h':
            case '?':
            default:
                print_usage(argv[0]);
                return (option == 'h') ? 0 : 1;
        }
    }

    SentinelRetrain::TopologyState inputs;
    inputs.hubHostname = hostname;
    inputs.hubPort = port_number;
    inputs.tickMicros = tick_micros;

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    Fw::Logger::log("Hit Ctrl-C to quit\n");

    SentinelRetrain::setupTopology(inputs);
    SentinelRetrain::startRateGroups();
    SentinelRetrain::teardownTopology(inputs);
    Fw::Logger::log("Exiting...\n");
    return 0;
}
