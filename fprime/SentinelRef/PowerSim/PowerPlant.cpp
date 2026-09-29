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
//! `phaseTick` places the instrument's schedule; it is `tick` unless D86b's opt-in
//! phase fault is configured, and the noise always follows the true tick.
double loadCurrent(uint32_t seed, uint32_t tick, uint32_t phaseTick, double housekeeping,
                   double onTicks = 420.0) {
    const double phase = std::fmod(static_cast<double>(phaseTick), 900.0);
    const double instrument = (phase < onTicks) ? 4.10 : 0.55;
    return housekeeping + instrument + 0.06 * plantNoise(seed, tick, 3u);
}

//! Solar input over the orbit, zero through eclipse.
double solarInput(uint32_t seed, uint32_t tick, double peakFactor, double eclipse = ECLIPSE) {
    const double phase = std::fmod(static_cast<double>(tick), ORBIT) / ORBIT;
    if (phase > (1.0 - eclipse)) {
        return 0.0;
    }
    const double lit = phase / (1.0 - eclipse);          // 0 .. 1 across daylight
    const double s = std::sin(lit * 3.14159265358979);
    // D85: `peakFactor` is 1.0 exactly unless healthy ageing is configured, and
    // SOLAR_PEAK * 1.0 is SOLAR_PEAK bit for bit, so the unaged run is unchanged.
    const double w = (SOLAR_PEAK * peakFactor) * s * s + 0.8 * plantNoise(seed, tick, 5u);
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

void PowerPlant::configureLoadShift(uint32_t start, uint32_t shift) {
    m_loadShifted = true;
    m_loadShiftStart = start;
    m_loadShift = shift;
}

void PowerPlant::configureAgeing(uint32_t start, double tau, double solarLoss,
                                 double emisLoss) {
    m_ageing = true;
    m_ageStart = start;
    m_ageTau = (tau > 0.0) ? tau : 1.0;
    m_solarLoss = solarLoss;
    m_emisLoss = emisLoss;
}

void PowerPlant::configureOcvAgeing(uint32_t start, double tau, double loss) {
    m_ocvAgeing = true;
    m_ocvStart = start;
    m_ocvTau = (tau > 0.0) ? tau : 1.0;
    m_ocvLoss = loss;
}

void PowerPlant::configureDrift(Drift which, uint32_t start, double tau, double delta) {
    Seasonal& d = m_drift[static_cast<uint32_t>(which)];
    d.on = true;
    d.start = start;
    d.tau = (tau > 0.0) ? tau : 1.0;
    d.delta = delta;
}

double PowerPlant::drift(Drift which) const {
    const Seasonal& d = m_drift[static_cast<uint32_t>(which)];
    if (!d.on || (m_tick < d.start)) {
        return 0.0;
    }
    return d.delta * (1.0 - std::exp(-static_cast<double>(m_tick - d.start) / d.tau));
}

void PowerPlant::configureBalance(double housekeeping, double shuntSoc) {
    m_balanced = true;
    m_housekeeping = housekeeping;
    m_shuntSoc = shuntSoc;
}

double PowerPlant::ageing() const {
    if (!m_ageing || (m_tick < m_ageStart)) {
        return 0.0;
    }
    return 1.0 - std::exp(-static_cast<double>(m_tick - m_ageStart) / m_ageTau);
}

void PowerPlant::step(bool faultActive, double faultRate) {
    // (!) THE FAULT TOUCHES ONE SCALAR AND NOTHING ELSE. Every other channel
    // moves because the physics couples it. That is what makes the anomaly
    // contextual by construction rather than by annotation, and it is 42's
    // "faults seeded IN THE PHYSICS" read literally.
    if (faultActive) {
        m_r *= (1.0 + faultRate);
    }

    // D85: healthy ageing. Both factors are exactly 1.0 when it is not configured.
    const double g = ageing();
    const double peakFactor = m_ageing ? (1.0 - (m_solarLoss * g)) : 1.0;
    const double emis = m_ageing ? (RAD_EMIS * (1.0 - (m_emisLoss * g))) : RAD_EMIS;

    // D85b 81.9: the survey's healthy drifts. Each is off unless configured, and then the
    // call below passes the pre-D85b constant itself, so the default plant is unchanged.
    const Seasonal& dE = m_drift[static_cast<uint32_t>(Drift::ECLIPSE_FRACTION)];
    const Seasonal& dD = m_drift[static_cast<uint32_t>(Drift::DUTY_ON_TIME)];
    const Seasonal& dH = m_drift[static_cast<uint32_t>(Drift::HOUSEKEEPING)];
    const double eclipse = dE.on ? (ECLIPSE * (1.0 + drift(Drift::ECLIPSE_FRACTION))) : ECLIPSE;
    const double onTicks = dD.on ? (420.0 * (1.0 + drift(Drift::DUTY_ON_TIME))) : 420.0;
    const double solar = solarInput(m_seed, m_tick, peakFactor, eclipse);
    // 2.35 A is the pre-D85 housekeeping draw, and it is what an unbalanced plant uses.
    const uint32_t phaseTick = (m_loadShifted && (m_tick >= m_loadShiftStart))
                                   ? (m_tick + m_loadShift) : m_tick;
    const double baseHk = m_balanced ? m_housekeeping : 2.35;
    const double hk = dH.on ? (baseHk + drift(Drift::HOUSEKEEPING)) : baseHk;
    const double load = loadCurrent(m_seed, m_tick, phaseTick, hk, onTicks);

    // Open-circuit voltage falls with state of charge; the bus sags under load
    // by the cell's own internal resistance, which is the quantity degrading.
    // D85b arm (a): the span falls with OCV ageing. Unconfigured, the expression is
    // exactly the pre-D85b one, so the default plant's bits do not move.
    double span = V_OC_SPAN;
    if (m_ocvAgeing && (m_tick >= m_ocvStart)) {
        const double gOcv = 1.0 - std::exp(-static_cast<double>(m_tick - m_ocvStart) / m_ocvTau);
        span = V_OC_SPAN * (1.0 - (m_ocvLoss * gOcv));
    }
    const double vOc = (V_OC_FULL - V_OC_SPAN) + span * m_soc;
    const double bus = vOc - (load * m_r);

    // Charge current is whatever the array can supply beyond the load.
    double charge = (bus > 1.0) ? ((solar / bus) - load) : -load;

    // D85: the shunt regulator, only on the balanced operating point. Above the
    // shunt threshold a surplus is dumped rather than charged into the battery.
    if (m_balanced && (m_soc >= m_shuntSoc) && (charge > 0.0)) {
        charge = 0.0;
    }

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
    m_rad += ((toRad + (m_duty * HEATER_W) - (emis * (m_rad - T_SINK))) * DT) / RAD_C;

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
