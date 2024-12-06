#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matmul.cuh"
#include <stdio.h>

namespace py = pybind11;

py::array_t<float> py_matMul(py::array_t<float> A, py::array_t<float> B)
{
    if (A.shape(1) != B.shape(0))
        throw std::invalid_argument("Inner dimensions of input arrays must agree.");
    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::invalid_argument("Input arrays must be 2D.");

    bool AT = A.flags() == 1282 ? true : false;
    bool BT = B.flags() == 1282 ? true : false;

    if (AT && BT)
        throw std::logic_error("Not implemented error");

    int M = A.shape(0);
    int K = A.shape(1);
    int N = B.shape(1);

    auto A_buf = A.request();
    auto B_buf = B.request();

    float* A_ptr = static_cast<float*>(A_buf.ptr);
    float* B_ptr = static_cast<float*>(B_buf.ptr);
    float* C_ptr = matMul(A_ptr, B_ptr, M, K, N, AT, BT);

    auto result = py::array_t<float>({M*N}, {sizeof(float)}, C_ptr);

    result.resize({M, N});

    return result;
}

PYBIND11_MODULE(matmul, m)
{
    m.def("matmul", &py_matMul, "Matrix Multiplication using CUDA",
        py::arg("A"), py::arg("B"));
}
