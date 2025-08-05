#pragma once
#include "../order.hpp"

struct PriceThresholdPolicy {
    double min_price = 100.0;

    bool should_fill(const Order& order) const {
        return order.price >= min_price;
    }
};
