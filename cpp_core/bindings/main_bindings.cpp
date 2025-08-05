#include <pybind11/pybind11.h>

// 声明子模块绑定函数（由其他 .cpp 文件提供）
void bind_data_feed(pybind11::module_& m);
void bind_order_executor(pybind11::module_& m);

namespace py = pybind11;

// PyBind11 模块入口
PYBIND11_MODULE(cpp_trading, m) {
    m.doc() = "C++ backend for the Quant Trading System";

    bind_data_feed(m);
    bind_order_executor(m);
}
