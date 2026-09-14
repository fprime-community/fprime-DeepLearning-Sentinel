// The physics. Constants chosen by running `PlantTune.cpp` on the host, not by
// estimate: the healthy run has to stay comfortably inside every RED limit for
// longer than the toolkit's data floor, and the seeded degradation has to leave
// every channel in limits while the current-to-temperature relationship breaks,
// then trip one limit. Both are 42's T2, and neither is a property anything can
// be assumed to have.
#include "PowerPlant.hpp"

#include <cmath>

namespace Testbed {

namespace {

// -- the operating point ----------------------------------------------------
constexpr double DT = 1.0;              // one tick, one second of plant time
constexpr double ORBIT = 5400.0;        // ticks per orbit; 90 min at 1 Hz
constexpr double ECLIPSE = 0.35;        // fraction of the orbit in shadow
constexpr double SOLAR_PEAK = 280.0;    // W

constexpr double R_NOMINAL = 0.040;     // ohm, healthy cell
constexpr double V_OC_FULL = 31.5;      // V at SoC 1.0
constexpr double V_OC_SPAN = 4.5;       // V dropped from full to empty

constexpr double CELL_C = 900.0;        // J/K, cell stack
constexpr double RAD_C = 420.0;         // J/K, radiator
constexpr double K_CELL_RAD = 0.15;     // W/K, cell to radiator. (!) SMALL ON PURPOSE:
                                        // at 1.60 the cell is pinned to the radiator and
                                        // ohmic heating moves it by 1 K, so no degradation
                                        // could ever reach a temperature limit. A cell that
                                        // cannot overheat is not a testbed for this.
constexpr double RAD_EMIS = 0.62;       // W/K, radiator to sink
constexpr double T_SINK = -45.0;        // degC, deep space side
constexpr double HEATER_W = 45.0;       // W at full duty
constexpr double RAD_SETPOINT = 6.0;    // degC, bang-bang target
constexpr double RAD_HYST = 2.0;        // degC

constexpr double CAPACITY_AS = 190000.0;  // amp-seconds, 52.8 Ah

//! The bus load: a duty-cycled instrument plus a steady housekeeping draw.
//! Deterministic in the tick, so it repeats exactly.
double loadCurrent(uint32_t seed, uint32_t tick) {
    const double phase = std::fmod(static_cast<double>(tick), 900.0);
    const double instrument = (phase < 420.0) ? 4.10 : 0.55;
    const double housekeeping = 2.35;
    return housekeeping + instrument + 0.06 * plantNoise(seed, tick, 3u);
}

//! Solar input over the orbit, zero through eclipse.
double solarInput(uint32_t seed, uint32_t tick) {
    const double phase = std::fmod(static_cast<double>(tick), ORBIT) / ORBIT;
    if (phase > (1.0 - ECLIPSE)) {
        return 0.0;
    }
    const double lit = phase / (1.0 - ECLIPSE);          // 0 .. 1 across daylight
    const double s = std::sin(lit * 3.14159265358979);
    const double w = SOLAR_PEAK * s * s + 0.8 * plantNoise(seed, tick, 5u);
    return (w > 0.0) ? w : 0.0;      // an array does not generate negative power
}

}  // namespace

void PowerPlant::reset(uint32_t seed) {
    m_seed = seed;
    m_tick = 0u;
    m_r = R_NOMINAL;
    m_soc = 0.86;
    m_cell = 17.0;
    m_rad = 6.0;
    m_duty = 0.0;
    for (uint32_t c = 0u; c < PLANT_CHANNELS; ++c) {
        m_v[c] = 0.0;
    }
}

void PowerPlant::step(bool faultActive, double faultRate) {
    // (!) THE FAULT TOUCHES ONE SCALAR AND NOTHING ELSE. Every other channel
    // moves because the physics couples it. That is what makes the anomaly
    // contextual by construction rather than by annotation, and it is 42's
    // "faults seeded IN THE PHYSICS" read literally.
    if (faultActive) {
        m_r *= (1.0 + faultRate);
    }

    const double solar = solarInput(m_seed, m_tick);
    const double load = loadCurrent(m_seed, m_tick);

    // Open-circuit voltage falls with state of charge; the bus sags under load
    // by the cell's own internal resistance, which is the quantity degrading.
    const double vOc = (V_OC_FULL - V_OC_SPAN) + V_OC_SPAN * m_soc;
    const double bus = vOc - (load * m_r);

    // Charge current is whatever the array can supply beyond the load.
    const double charge = (bus > 1.0) ? ((solar / bus) - load) : -load;

    // Coulomb count, clamped at both rails.
    m_soc += (charge * DT) / CAPACITY_AS;
    if (m_soc > 1.0) { m_soc = 1.0; }
    if (m_soc < 0.0) { m_soc = 0.0; }

    // Ohmic self-heating. The cell carries whichever current is larger in
    // magnitude, so eclipse discharge heats it as charging does.
    const double through = (std::fabs(charge) > load) ? std::fabs(charge) : load;
    const double ohmic = through * through * m_r;

    // Bang-bang survival heater on the radiator.
    if (m_rad < (RAD_SETPOINT - RAD_HYST)) { m_duty = 1.0; }
    if (m_rad > (RAD_SETPOINT + RAD_HYST)) { m_duty = 0.0; }

    const double toRad = K_CELL_RAD * (m_cell - m_rad);
    m_cell += ((ohmic - toRad) * DT) / CELL_C;
    m_rad += ((toRad + (m_duty * HEATER_W) - (RAD_EMIS * (m_rad - T_SINK))) * DT) / RAD_C;

    m_v[CH_SOLAR_INPUT] = solar;
    m_v[CH_CHARGE_CURRENT] = charge;
    m_v[CH_LOAD_CURRENT] = load;
    m_v[CH_BUS_VOLTAGE] = bus;
    m_v[CH_CELL_TEMP] = m_cell;
    m_v[CH_RADIATOR_TEMP] = m_rad;
    m_v[CH_HEATER_DUTY] = m_duty;
    m_v[CH_STATE_OF_CHARGE] = m_soc;

    ++m_tick;
}

uint32_t PowerPlant::firstRedCrossing() const {
    for (uint32_t c = 0u; c < PLANT_CHANNELS; ++c) {
        if ((m_v[c] < PLANT_RED[c].low) || (m_v[c] > PLANT_RED[c].high)) {
            return c;
        }
    }
    return PLANT_CHANNELS;
}

}  // namespace Testbed
