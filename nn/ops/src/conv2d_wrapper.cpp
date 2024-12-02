#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "conv2d.cuh"
#include <stdio.h>

namespace py = pybind11;

py::array_t<float> py_conv2d(py::array_t<float> X, py::array_t<float> kernel, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride) 
{
    if (X.shape(0) != N || X.shape(1) != C_in || X.shape(2) != H || X.shape(3) != W)
    {
        throw std::runtime_error("X shape don't match!");
    }
    if (kernel.shape(1) != C_out || kernel.shape(1) != C_in || kernel.shape(2) != KH || kernel.shape(3) != KW)
    {
        throw std::runtime_error("kernel shape don't match!");
    }

    auto X_buf = X.request();
    auto kernel_buf = kernel.request();\

    int OH = (H - KH + 1) / stride;
    int OW = (W - KW + 1) / stride;
    int M = C_out;
    int K = C_in * KH * KW;
    int N_O = N * OH * OW;

    float *X_ptr = static_cast<float*>(X_buf.ptr);
    float *kernel_ptr = static_cast<float*>(kernel_buf.ptr);
    float *C_ptr = Conv2d(X_ptr, kernel_ptr, N, H, W, C_in, C_out, KH, KW, stride, M, K, N_O);

    py::array_t<float> result(
        {N, C_out, OH, OW}, 
        {sizeof(float) * C_out * OH * OW, sizeof(float) * OH * OW, sizeof(float) * OW, sizeof(float)},
        C_ptr 
    );

    free(C_ptr);
    
    return result;
}

py::tuple py_conv2d_backward(py::array_t<float> X, py::array_t<float> kernel, py::array_t<float> Dout, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride) 
{
    auto X_buf = X.request();
    auto kernel_buf = kernel.request();
    auto Dout_buf = Dout.request();

    float *X_ptr = static_cast<float*>(X_buf.ptr);
    float *kernel_ptr = static_cast<float*>(kernel_buf.ptr);
    float *Dout_ptr = static_cast<float*>(Dout_buf.ptr);

    int OH = (H - KH + 1) / stride;
    int OW = (W - KW + 1) / stride;
    
    int M = C_in * KH * KW;
    int K = C_out;
    int N_O = N * OH * OW;

    float *DX_ptr = Conv2dBackwardDX(kernel_ptr, Dout_ptr, N, H, W, C_in, C_out, KH, KW, stride, M, K, N_O);
    float *DK_ptr = Conv2dBackwardDK(X_ptr, Dout_ptr, N, H, W, C_in, C_out, KH, KW, stride, K, N_O, M);

    py::array_t<float> DX(
        {N, C_in, H, W}, 
        {sizeof(float) * C_in * H * W, sizeof(float) * H * W, sizeof(float) * W, sizeof(float)},
        DX_ptr 
    );

    py::array_t<float> DK(
        {C_out, C_in, KH, KW}, 
        {sizeof(float) * C_in * KH * KW, sizeof(float) * KH * KW, sizeof(float) * KW, sizeof(float)},
        DK_ptr 
    );

    free(DX_ptr);
    free(DK_ptr);

    return py::make_tuple(DX, DK);
}

PYBIND11_MODULE(conv2d, m)
{
    m.def("conv2d", &py_conv2d, "Conv2d forward",
        py::arg("X"), py::arg("kernel"), py::arg("N"), py::arg("H"), py::arg("W"), py::arg("C_in"), py::arg("C_out"), py::arg("KH"), py::arg("KW"), py::arg("stride"));
    m.def("conv2d_backward", &py_conv2d_backward, "Conv2d backward",
        py::arg("x"), py::arg("kernel"), py::arg("dout"), py::arg("N"), py::arg("H"), py::arg("W"), py::arg("C_in"), py::arg("C_out"), py::arg("KH"), py::arg("KW"), py::arg("stride"));
}
