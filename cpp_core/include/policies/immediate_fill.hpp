#pragma once
#include "../order.hpp"

struct ImmediateFillPolicy {
    bool should_fill(const Order& order) const {
        return true; // Fill immediately
    }
};
