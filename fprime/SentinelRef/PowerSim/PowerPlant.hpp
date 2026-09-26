// A coupled power and thermal subsystem, deterministic and seeded.
//
// (!) NO F' TYPES AND NO F' HEADERS, deliberately. The same separation
// `flight/include/sentinel/Types.hpp` makes for the inference core, and for the
// same reason: the physics can then be compiled, run and TUNED on a host with no
// framework present, which is how its constants were chosen. `PowerSim.cpp` is
// the F' wrapper around it and owns no arithmetic of its own.
//
// Determinism is a requirement rather than a convenience. A warning time is the
// gap between two instants in one run; if the run is not reproducible the gap
// cannot be re-measured, and `docs/MODELS.md` 42's T1 through T5 all rest on
// re-running the same seed.
#ifndef SENTINELREF_POWER_PLANT_HPP
#define SENTINELREF_POWER_PLANT_HPP

#include <cstdint>

namespace Testbed {

//! Published channel order. The model the toolkit trains is fitted on this
//! order and `model.bin`'s CHANNELS record carries these names, so it is a
//! contract and not a convenience.
enum PlantChannel : uint32_t {
    CH_SOLAR_INPUT = 0,
    CH_CHARGE_CURRENT = 1,
    CH_LOAD_CURRENT = 2,
    CH_BUS_VOLTAGE = 3,
    CH_CELL_TEMP = 4,
    CH_RADIATOR_TEMP = 5,
    CH_HEATER_DUTY = 6,
    CH_STATE_OF_CHARGE = 7,
    PLANT_CHANNELS = 8
};

//! RED limits, low and high, in published channel order.
//!
//! (!) THIS IS A SECOND COPY OF `PowerSim.fpp`'s DICTIONARY AND IT IS GUARDED.
//! F' v4.3.0 does not expose telemetry limits to C++ -- they live in the
//! dictionary and "limit checking is performed by the ground system"
//! (`docs/reference/system-functional/telemetry-chan.md:32`). 42.3 departure 1
//! needs them ONBOARD, so that a limit trip and a warning share a time base, and
//! there is no autocoded constant to read. So the numbers are written twice and
//! `tests/test_powersim_limits.py` re-derives this table from the FPP on every
//! run. An unchecked second copy is what that departure forbids; a checked one is
//! the only shape available.
struct RedLimits {
    double low;
    double high;
};

constexpr RedLimits PLANT_RED[PLANT_CHANNELS] = {
    {  -5.0, 320.0 },   // SolarInput
    { -12.0,  12.0 },   // ChargeCurrent
    {   0.0,  11.0 },   // LoadCurrent
    {  26.0,  33.0 },   // BusVoltage
    { -10.0,  45.0 },   // CellTemp
    { -40.0,  35.0 },   // RadiatorTemp
    { -0.01,  1.01 },   // HeaterDuty
    {  0.20,  1.01 }    // StateOfCharge
};

//! Deterministic value noise. A counter-based mix rather than a stateful PRNG,
//! so the sample at tick t does not depend on how many samples were drawn
//! before it -- which keeps a run reproducible even if the caller changes how
//! often it asks. Same technique `flight/test/DeterminismTest.cpp:45` uses for
//! its drive.
inline double plantNoise(uint32_t seed, uint32_t tick, uint32_t stream) {
    uint32_t h = seed ^ (tick * 2654435761u) ^ (stream * 40503u);
    h ^= h >> 16; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    return (static_cast<double>(h) / 4294967295.0) * 2.0 - 1.0;   // [-1, +1]
}

//! The plant. One `step()` per rate-group tick.
class PowerPlant {
  public:
    void reset(uint32_t seed);

    //! Advance one tick. `faultActive` ramps the cell's internal resistance,
    //! which is the ONLY quantity the fault touches -- everything else moves
    //! because the physics couples it, which is what makes the anomaly
    //! contextual rather than injected into a channel.
    void step(bool faultActive, double faultRate);

    //! D85 / `docs/MODELS.md` 78: HEALTHY AGEING, off unless configured.
    //!
    //! Two slow, saturating losses that a healthy spacecraft accumulates and that no
    //! engineer would call a failure: the solar array's output falls (radiation and
    //! UV darkening of the cover glass) and the radiator's emissivity falls
    //! (contamination of its surface). Each follows `1 - exp(-(t - start) / tau)`, so
    //! it approaches a floor and stops -- which is what makes it ageing and not a
    //! trend toward a limit. Neither touches the cell's resistance, which is the
    //! fault's quantity: ageing and failure are different physics by construction.
    //!
    //! With ageing unconfigured, `step()` computes exactly what it did before D85
    //! (`tests/test_powersim_limits.py` pins the traces by hash).
    void configureAgeing(uint32_t start, double tau, double solarLoss, double emisLoss);

    //! The ageing factor g(t) in [0, 1): 0 before `start`, then saturating.
    double ageing() const;

    //! D85 / `docs/MODELS.md` 78: an ORBIT-BALANCED operating point, off unless
    //! configured.
    //!
    //! (!) THE DEFAULT PLANT IS NOT ENERGY-BALANCED, AND 42's RUNS WERE TOO SHORT TO
    //! SEE IT. Over an orbit the array delivers about 3.1 A at the bus against a mean
    //! load of about 4.6 A, so state of charge falls on every orbit and a HEALTHY run
    //! crosses its 0.30 yellow at about tick 74,680 (measured, seeds 1-3). 42.9 ran
    //! 22,000 ticks. A plant whose healthy state is a slow march to a limit cannot
    //! host an ageing arm, and a trend-to-limit rule would rightly refuse all of it.
    //!
    //! This mode lowers the housekeeping draw so the orbit runs a small surplus, and
    //! adds the shunt regulator every real power system has: above `shuntSoc` the
    //! array's excess is dumped rather than charged, so state of charge settles into
    //! a periodic band instead of drifting. With it unconfigured, `step()` is
    //! unchanged.
    void configureBalance(double housekeeping, double shuntSoc);

    double value(uint32_t channel) const { return m_v[channel]; }
    uint32_t tick() const { return m_tick; }
    double resistance() const { return m_r; }

    //! Index of the first channel outside its RED limits, or PLANT_CHANNELS.
    uint32_t firstRedCrossing() const;

  private:
    double m_v[PLANT_CHANNELS] = {0.0};
    double m_r = 0.0;          //!< cell internal resistance, ohms
    double m_soc = 0.0;        //!< state of charge, fraction
    double m_cell = 0.0;       //!< cell temperature, degC
    double m_rad = 0.0;        //!< radiator temperature, degC
    double m_duty = 0.0;       //!< heater duty, fraction
    uint32_t m_seed = 1u;
    uint32_t m_tick = 0u;
    bool m_ageing = false;     //!< D85: healthy ageing configured
    uint32_t m_ageStart = 0u;
    double m_ageTau = 1.0;
    double m_solarLoss = 0.0;  //!< fraction of SOLAR_PEAK lost at saturation
    double m_emisLoss = 0.0;   //!< fraction of RAD_EMIS lost at saturation
    bool m_balanced = false;   //!< D85: orbit-balanced operating point
    double m_housekeeping = 2.35;
    double m_shuntSoc = 1.0;
};

}  // namespace Testbed

#endif
