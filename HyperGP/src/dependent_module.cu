// 类型列表
using SupportedTypes_1 = std::tuple<bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double>;

template<typename Func, typename... TypeLists>
void register_combinations(Func&& func, TypeLists&&... type_lists){
    std::apply([&](auto... types1) {
        (std::apply([&](auto... types2) {
            (func(types1, types2), ...);
        }, type_lists), ...);
    }, type_lists...);
}

PYBIND11_MODULE(ewise_add, m){
    register_combinations([&m](auto type1, auto type2, auto type3){
        m.def("ewise_add", [](Array<type1>& a, Array<type2>& b, Array<type3>& out, int32_t& offset_a, int32_t& offset_b){
            ewise_compute_2op<type1, type2, type3>(a, b, out, offset_a, offset_b, 0);
        });
    }, SupportedTypes{}, SupportedTypes{}, SupportedTypes{});
}