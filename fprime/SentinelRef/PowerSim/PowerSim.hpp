// The F' wrapper around `PowerPlant`. It owns no arithmetic: the physics is in
// PowerPlant.cpp, which compiles without F' so its constants could be tuned on
// a host (`docs/MODELS.md` 42).
#ifndef SENTINELREF_POWERSIM_HPP
#define SENTINELREF_POWERSIM_HPP

#include "SentinelRef/PowerSim/PowerSimComponentAc.hpp"
#include "SentinelRef/PowerSim/PowerPlant.hpp"

namespace Testbed {

class PowerSim final : public PowerSimComponentBase {
  public:
    explicit PowerSim(const char* compName);
    ~PowerSim() override = default;

  private:
    void schedIn_handler(FwIndexType portNum, U32 context) override;

    //! Evaluate this component's own dictionary RED limits against the onboard
    //! time base (42.3 departure 1). Emits once, on the first crossing.
    void checkLimits();

    PowerPlant m_plant;
    bool m_started = false;
    bool m_tripped = false;
    bool m_faulted = false;
    U32 m_seed = 1U;
    U32 m_faultStart = 8000U;
    F32 m_faultRate = 2.0e-4F;
};

}  // namespace Testbed

#endif
