#include "sentinel/Status.hpp"

namespace Sentinel {

const char* statusName(LoadStatus status) {
    switch (status) {
        case LoadStatus::OK:              return "OK";
        case LoadStatus::BAD_MAGIC:       return "BAD_MAGIC";
        case LoadStatus::BAD_VERSION:     return "BAD_VERSION";
        case LoadStatus::BAD_HEADER_CRC:  return "BAD_HEADER_CRC";
        case LoadStatus::BAD_STATIC_CRC:  return "BAD_STATIC_CRC";
        case LoadStatus::BAD_PARAM_CRC:   return "BAD_PARAM_CRC";
        case LoadStatus::BAD_ARCH:        return "BAD_ARCH";
        case LoadStatus::BAD_GATE_ORDER:  return "BAD_GATE_ORDER";
        case LoadStatus::BAD_SHAPE:       return "BAD_SHAPE";
        case LoadStatus::TOO_LARGE:       return "TOO_LARGE";
        case LoadStatus::TRUNCATED:       return "TRUNCATED";
        case LoadStatus::BAD_NORM_POLICY: return "BAD_NORM_POLICY";
        default:                          return "UNKNOWN";
    }
}

}  // namespace Sentinel
