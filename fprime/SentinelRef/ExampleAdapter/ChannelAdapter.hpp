// F's Passive Adapter Pattern, at its smallest useful size: a mission's typed
// channel values in, one `Sentinel::ChannelVector` per tick out.
//
// (!) EXAMPLE CODE, NOT PRODUCT. `fprime/library.cmake` exports
// `Sentinel/Monitor` and nothing here. A mission COPIES this directory and
// replaces `valueIn` with its own telemetry; it does not link it.
#ifndef SENTINELREF_EXAMPLEADAPTER_HPP
#define SENTINELREF_EXAMPLEADAPTER_HPP

#include "SentinelRef/ExampleAdapter/ChannelAdapterComponentAc.hpp"
#include "Sentinel/Monitor/FppConstantsAc.hpp"

namespace Example {

class ChannelAdapter final : public ChannelAdapterComponentBase {
  public:
    explicit ChannelAdapter(const char* compName);
    ~ChannelAdapter() override = default;

    //! How many channels this adapter publishes. Called once, at init.
    //! CPP-1: storage is fixed at `Sentinel::MAX_CHANNELS` and nothing is
    //! allocated here or afterwards.
    void configure(U32 published);

  private:
    void valueIn_handler(FwIndexType portNum, U32 index, F32 value) override;
    void schedIn_handler(FwIndexType portNum, U32 context) override;

    Sentinel::ChannelVector m_values;
    bool m_seen[Sentinel::MAX_CHANNELS];
    U32 m_published = 0U;
};

}  // namespace Example

#endif
