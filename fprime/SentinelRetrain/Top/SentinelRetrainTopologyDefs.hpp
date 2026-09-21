// ======================================================================
// \title  SentinelRetrainTopologyDefs.hpp
// \brief definitions the topology autocoder requires
// ======================================================================
#ifndef SENTINELRETRAIN_SENTINELRETRAINTOPOLOGYDEFS_HPP
#define SENTINELRETRAIN_SENTINELRETRAINTOPOLOGYDEFS_HPP

#include <Fw/FPrimeBasicTypes.hpp>
#include "SentinelRetrain/Top/FppConstantsAc.hpp"

//! Required by the autocoder. This deployment declares no `health connections`
//! -- nothing in it has a Svc.Ping port -- so the namespace is empty rather
//! than absent.
namespace PingEntries {}

namespace SentinelRetrain {

//! The autocoder requires a type of this name. It is otherwise opaque to it.
struct TopologyState {
    const char* hubHostname;  //!< where SentinelRef's hub is listening
    U16 hubPort;              //!< 0 means "no hub": run the cycle locally only
};

namespace PingEntries = ::PingEntries;
}  // namespace SentinelRetrain

#endif
