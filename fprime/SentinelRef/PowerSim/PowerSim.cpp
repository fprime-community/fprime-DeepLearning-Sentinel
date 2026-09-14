#include "SentinelRef/PowerSim/PowerSim.hpp"

#include "SentinelRef/PowerSim/FppConstantsAc.hpp"
#include "Sentinel/Monitor/FppConstantsAc.hpp"

namespace Testbed {

namespace {
//! The published width must fit the vector the Monitor accepts. Checked here
//! rather than trusted, because the two are declared in different FPP files.
static_assert(static_cast<U32>(PLANT_CHANNELS) == static_cast<U32>(POWERSIM_CHANNELS),
              "PowerPlant's channel count and PowerSim.fpp's must agree");
static_assert(static_cast<U32>(PLANT_CHANNELS) <= static_cast<U32>(Sentinel::MAX_CHANNELS),
              "PowerSim publishes more channels than Sentinel.MAX_CHANNELS accepts");
}  // namespace

PowerSim::PowerSim(const char* compName) : PowerSimComponentBase(compName) {
    m_plant.reset(1U);
}

void PowerSim::schedIn_handler(FwIndexType portNum, U32 context) {
    static_cast<void>(portNum);
    static_cast<void>(context);

    // Parameters are read once, on the first tick, so a run is reproducible
    // from the values that were in force when it began. A parameter changing
    // mid-run would make the trace unrepeatable, which is the one property
    // every figure 42 registers depends on.
    if (!m_started) {
        Fw::ParamValid valid = Fw::ParamValid::INVALID;
        const U32 seed = this->paramGet_SEED(valid);
        if (valid == Fw::ParamValid::VALID) { m_seed = seed; }
        const U32 start = this->paramGet_FAULT_START(valid);
        if (valid == Fw::ParamValid::VALID) { m_faultStart = start; }
        const F32 rate = this->paramGet_FAULT_RATE(valid);
        if (valid == Fw::ParamValid::VALID) { m_faultRate = rate; }
        m_plant.reset(m_seed);
        m_started = true;
    }

    Fw::ParamValid modeValid = Fw::ParamValid::INVALID;
    const FaultMode mode = this->paramGet_FAULT_MODE(modeValid);
    const bool wantFault = (modeValid == Fw::ParamValid::VALID) &&
                           (mode == FaultMode::RESISTANCE_RISE);

    const U32 tick = m_plant.tick();
    const bool active = wantFault && (tick >= m_faultStart);
    if (active && !m_faulted) {
        this->log_ACTIVITY_HI_FaultInjected(mode, tick);
        m_faulted = true;
    }

    m_plant.step(active, static_cast<double>(m_faultRate));

    // -- publish, in the model's channel order --------------------------------
    F32 values[POWERSIM_CHANNELS];
    for (U32 c = 0U; c < static_cast<U32>(POWERSIM_CHANNELS); ++c) {
        values[c] = static_cast<F32>(m_plant.value(c));
    }
    this->tlmWrite_SolarInput(values[CH_SOLAR_INPUT]);
    this->tlmWrite_ChargeCurrent(values[CH_CHARGE_CURRENT]);
    this->tlmWrite_LoadCurrent(values[CH_LOAD_CURRENT]);
    this->tlmWrite_BusVoltage(values[CH_BUS_VOLTAGE]);
    this->tlmWrite_CellTemp(values[CH_CELL_TEMP]);
    this->tlmWrite_RadiatorTemp(values[CH_RADIATOR_TEMP]);
    this->tlmWrite_HeaterDuty(values[CH_HEATER_DUTY]);
    this->tlmWrite_StateOfCharge(values[CH_STATE_OF_CHARGE]);
    this->tlmWrite_SimTick(m_plant.tick());

    checkLimits();

    // The watched vector, to whatever is listening. `valid` is true because
    // every channel was measured this tick by construction -- a simulation has
    // no dropouts, and inventing them would be a second fault nobody registered.
    if (this->isConnected_channelOut_OutputPort(0)) {
        Sentinel::ChannelVector vec;
        for (U32 c = 0U; c < static_cast<U32>(POWERSIM_CHANNELS); ++c) {
            vec[static_cast<FwSizeType>(c)] = values[c];
        }
        for (U32 c = static_cast<U32>(POWERSIM_CHANNELS);
             c < static_cast<U32>(Sentinel::MAX_CHANNELS); ++c) {
            vec[static_cast<FwSizeType>(c)] = 0.0F;
        }
        this->channelOut_out(0, vec, true);
    }
}

void PowerSim::checkLimits() {
    // (!) EVALUATED ONBOARD, AGAINST THIS COMPONENT'S OWN DICTIONARY VALUES.
    // F' checks telemetry limits on the ground, so a ground-stamped trip and an
    // onboard-stamped warning sit on different clocks and subtracting one from
    // the other measures the downlink rather than the detector (42.3 departure
    // 1). `PLANT_RED` is the same table PowerSim.fpp declares, and
    // `tests/test_powersim_limits.py` re-derives it from the FPP so the two
    // cannot drift.
    if (m_tripped) {
        return;
    }
    const U32 crossed = m_plant.firstRedCrossing();
    if (crossed < static_cast<U32>(PLANT_CHANNELS)) {
        this->log_WARNING_HI_LimitTripped(crossed,
                                          static_cast<F32>(m_plant.value(crossed)),
                                          m_plant.tick());
        m_tripped = true;
    }
}

}  // namespace Testbed
