#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "order_executor.hpp"

namespace py = pybind11;

void bind_order_executor(py::module_& m) {
    py::enum_<OrderType>(m, "OrderType")
        .value("BUY", OrderType::BUY)
        .value("SELL", OrderType::SELL);

    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("symbol", &Order::symbol)
        .def_readwrite("type", &Order::type)
        .def_readwrite("price", &Order::price)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("timestamp", &Order::timestamp);

    py::class_<OrderExecutor>(m, "OrderExecutor")
        .def(py::init<>())
        .def("submit_order", &OrderExecutor::submit_order)
        .def("simulate_execution", &OrderExecutor::simulate_execution)
        .def("get_filled_orders", &OrderExecutor::get_filled_orders);
}
