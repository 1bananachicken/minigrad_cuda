#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "mat_mul.cuh"
#include <stdio.h>

namespace py = pybind11;

py::array_t<float> py_matMul(py::array_t<float> A, py::array_t<float> B, int M, int K, int N) {
    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::invalid_argument("Input arrays must be 2D.");

    if (A.shape(1) != K || B.shape(0) != K)
        throw std::invalid_argument("Input dimensions are mismatched.");

    auto A_buf = A.request();
    auto B_buf = B.request();

    float* A_ptr = static_cast<float*>(A_buf.ptr);
    float* B_ptr = static_cast<float*>(B_buf.ptr);
    float* C_ptr = matMul(A_ptr, B_ptr, M, K, N);

    auto result = py::array_t<float>({M*N}, {sizeof(float)}, C_ptr);

    result.resize({M, N});

    return result;
}

PYBIND11_MODULE(mat_mul, m) {
    m.def("mat_mul", &py_matMul, "Matrix Multiplication using CUDA",
          py::arg("A"), py::arg("B"), py::arg("M"), py::arg("K"), py::arg("N"));
}
