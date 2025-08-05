#pragma once
#include "../order.hpp"

struct DelayedFillPolicy {
    long current_time = 1650000010;

    bool should_fill(const Order& order) const {
        return order.timestamp < current_time;
    }
};
