// ======================================================================
// \title  Main.cpp
// \brief main program for the F' application. Intended for CLI-based systems (Linux, macOS)
//
// ======================================================================
// Used to access topology functions
#include <SentinelRef/Top/SentinelRefTopology.hpp>
// OSAL initialization
#include <Os/Os.hpp>
// Used for signal handling shutdown
#include <signal.h>
// Used for command line argument processing
#include <getopt.h>
// Used for atoi
#include <cstdlib>
// Used for logging to the console
#include <Fw/Logger/Logger.hpp>

/**
 * \brief print command line help message
 *
 * This will print a command line help message including the available command line arguments.
 *
 * @param app: name of application
 */
void print_usage(const char* app) {
    // One call per line: Fw::Logger truncates a long message (found in D85).
    Fw::Logger::log("Usage: ./%s [options]\n", app);
    Fw::Logger::log("-a\thostname/IP address\n");
    Fw::Logger::log("-p\tport_number\n");
    Fw::Logger::log("-H\thub server port; 0 or absent runs the detector with no hub\n");
    Fw::Logger::log("-t\tbase tick in microseconds; absent runs at 1 Hz (host runs only)\n");
    Fw::Logger::log("-L\tPowerSim runs the balanced, spun-up plant of docs/MODELS.md 78.3\n");
    Fw::Logger::log("-g\twith -L, the plant also ages at this emissivity loss\n");
}

/**
 * \brief shutdown topology cycling on signal
 *
 * The reference topology allows for a simulated cycling of the rate groups. This simulated cycling needs to be stopped
 * in order for the program to shutdown. This is done via handling signals such that it is performed via Ctrl-C
 *
 * @param signum
 */
static void signalHandler(int signum) {
    SentinelRef::stopRateGroups();
}

/**
 * \brief execute the program
 *
 * This F' program is designed to run in standard environments (e.g. Linux/macOs running on a laptop). Thus it uses
 * command line inputs to specify how to connect.
 *
 * @param argc: argument count supplied to program
 * @param argv: argument values supplied to program
 * @return: 0 on success, something else on failure
 */
int main(int argc, char* argv[]) {
    I32 option = 0;
    CHAR* hostname = nullptr;
    U16 port_number = 0;
    U16 hub_port = 0;
    U32 tick_micros = 0;
    bool loop_plant = false;
    double age_emis = 0.0;

    Os::init();

    // Loop while reading the getopt supplied options
    while ((option = getopt(argc, argv, "hp:a:H:t:Lg:")) != -1) {
        switch (option) {
            // Handle the -a argument for address/hostname
            case 'a':
                hostname = optarg;
                break;
            // Handle the -p port number argument
            case 'p':
                port_number = static_cast<U16>(atoi(optarg));
                break;
            // (!) The HUB's port, separate from the GDS port on purpose: the
            // detector's downlink and the crossing must be able to fail
            // independently, because the downlink is how the crossing is
            // observed (docs/MODELS.md 70).
            case 'H':
                hub_port = static_cast<U16>(atoi(optarg));
                break;
            // D85: the accelerated clock for the end-to-end loop run.
            case 't':
                tick_micros = static_cast<U32>(strtoul(optarg, nullptr, 10));
                break;
            case 'L':
                loop_plant = true;
                break;
            case 'g':
                age_emis = strtod(optarg, nullptr);
                break;
            // Cascade intended: help output
            case 'h':
            // Cascade intended: help output
            case '?':
            // Default case: output help and exit
            default:
                print_usage(argv[0]);
                return (option == 'h') ? 0 : 1;
        }
    }
    // Object for communicating state to the topology
    SentinelRef::TopologyState inputs;
    inputs.hostname = hostname;
    inputs.port = port_number;
    inputs.hubPort = hub_port;
    inputs.tickMicros = tick_micros;
    inputs.loopPlant = loop_plant;
    inputs.ageEmis = age_emis;

    // Setup program shutdown via Ctrl-C
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    Fw::Logger::log("Hit Ctrl-C to quit\n");

    // Setup, cycle, and teardown topology
    SentinelRef::setupTopology(inputs);
    SentinelRef::startRateGroups();
    SentinelRef::teardownTopology(inputs);
    Fw::Logger::log("Exiting...\n");
    return 0;
}
