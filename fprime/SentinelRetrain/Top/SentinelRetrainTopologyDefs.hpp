// ======================================================================
// \title  SentinelRetrainTopologyDefs.hpp
// \brief definitions the topology autocoder requires
// ======================================================================
#ifndef SENTINELRETRAIN_SENTINELRETRAINTOPOLOGYDEFS_HPP
#define SENTINELRETRAIN_SENTINELRETRAINTOPOLOGYDEFS_HPP

#include <Fw/Types/MallocAllocator.hpp>
#include <Fw/FPrimeBasicTypes.hpp>
#include "SentinelRetrain/Top/FppConstantsAc.hpp"

//! Required by the autocoder. This deployment declares no `health connections`
//! -- nothing in it has a Svc.Ping port -- so the namespace is empty rather
//! than absent.
namespace PingEntries {}

namespace SentinelRetrain {

//! The autocoder requires a type of this name. It is otherwise opaque to it.
//! The hub pool's backing allocator. Declared here because the AUTOCODED
//! topology is where Svc::BufferManager::setup is called from, and that file
//! sees this header and not SentinelRetrainTopology.cpp.
extern Fw::MallocAllocator hubAllocator;

struct TopologyState {
    const char* hubHostname;  //!< where SentinelRef's hub is listening
    U16 hubPort;              //!< 0 means "no hub": run the cycle locally only
};

namespace PingEntries = ::PingEntries;
}  // namespace SentinelRetrain

#endif
