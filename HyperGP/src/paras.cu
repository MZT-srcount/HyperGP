#include "paras.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

void initialize_global_streams() {
    std::cout << "initialize_global_streams " << std::endl;
    for(int k = 0; k < STREAM_NUM_NDARRAY; ++k){
        cudaStreamCreate(&streams[0][k]);
    }
}

// bool initialized = []() {
//     initialize_global_streams();
//     return true;
// }();

GlobalStreams gstreams;

PYBIND11_MODULE(global_streams, m){
    py::class_<GlobalStreams>(m, std::string("GlobalStreams").c_str());
    initialize_global_streams();
    gstreams.init(streams);
    m.def("get_gstreams", [](){ 
        return gstreams;
    });
}
