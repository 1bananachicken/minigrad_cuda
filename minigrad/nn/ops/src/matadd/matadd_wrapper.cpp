#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matadd.cuh"
#include <stdio.h>

namespace py = pybind11;

py::array_t<float> py_matAdd2d(py::array_t<float> A, py::array_t<float> B)
{
    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::invalid_argument("Input arrays must be 2D.");

    int N = A.shape(0);
    int C_out = A.shape(1);
    auto A_buf = A.request();
    auto B_buf = B.request();

    float *A_ptr = static_cast<float*>(A_buf.ptr);
    float *B_ptr = static_cast<float*>(B_buf.ptr);
    float *C_ptr = matAdd2d(A_ptr, B_ptr, N, C_out);

    auto result = py::array_t<float>({N, C_out}, {C_out * sizeof(float), sizeof(float)}, C_ptr);

    free(C_ptr);
    return result;
}

py::array_t<float> py_matAdd4d(py::array_t<float> A, py::array_t<float> B)
{
    if (A.ndim() != 4 || B.ndim() != 4)
        throw std::invalid_argument("Input arrays must be 4D.");

    int N = A.shape(0);
    int C_out = A.shape(1);
    int OH = A.shape(2);
    int OW = A.shape(3);
    auto A_buf = A.request();
    auto B_buf = B.request();

    float *A_ptr = static_cast<float*>(A_buf.ptr);
    float *B_ptr = static_cast<float*>(B_buf.ptr);
    float *C_ptr = matAdd4d(A_ptr, B_ptr, N, C_out, OH, OW);

    auto result = py::array_t<float>({N, C_out, OH, OW}, {C_out * OH * OW * sizeof(float), OH * OW * sizeof(float), OW * sizeof(float), sizeof(float)}, C_ptr);

    free(C_ptr);
    return result;
}

PYBIND11_MODULE(matadd, m)
{
    m.def("matAdd2d", &py_matAdd2d, "Matrix add 2D",
        py::arg("A"), py::arg("B"));
    m.def("matAdd4d", &py_matAdd4d, "Matrix add 4D",
        py::arg("A"), py::arg("B"));
}
