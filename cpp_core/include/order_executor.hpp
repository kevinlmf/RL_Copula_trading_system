#pragma once
#include <vector>
#include "order.hpp"

template <typename ExecutionPolicy>
class OrderExecutor {
public:
    explicit OrderExecutor(ExecutionPolicy policy = ExecutionPolicy())
        : policy_(policy) {}

    void submit_order(const Order& order) {
        order_queue_.push_back(order);
    }

    void simulate_execution() {
        for (const auto& order : order_queue_) {
            if (policy_.should_fill(order)) {
                filled_orders_.push_back(order);
            }
        }
        order_queue_.clear();
    }

    std::vector<Order> get_filled_orders() const {
        return filled_orders_;
    }

private:
    std::vector<Order> order_queue_;
    std::vector<Order> filled_orders_;
    ExecutionPolicy policy_;
};


