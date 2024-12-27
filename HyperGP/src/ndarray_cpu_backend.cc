#include<pybind11/pybind11.h>

namespace HyperGP{
namespace cpu{

}
}

PYBIND11_MODULE(ndarray_cpu_backend, m){
    namespace py = pybind11;
    using namespace HyperGP;
    using namespace cpu;
}