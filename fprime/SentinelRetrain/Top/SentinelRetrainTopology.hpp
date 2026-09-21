// ======================================================================
// \title  SentinelRetrainTopology.hpp
// \brief topology instantiation definitions
// ======================================================================
#ifndef SENTINELRETRAIN_SENTINELRETRAINTOPOLOGY_HPP
#define SENTINELRETRAIN_SENTINELRETRAINTOPOLOGY_HPP

#include <SentinelRetrain/Top/SentinelRetrainTopologyDefs.hpp>

namespace SentinelRetrain {

void setupTopology(const TopologyState& state);
void teardownTopology(const TopologyState& state);
void startRateGroups();
void stopRateGroups();

}  // namespace SentinelRetrain
#endif
